import time
import random
import math

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Task says 60s. We aim to finish well before 60s.
    # Let's set a hard deadline of 55s to be safe.
    DEADLINE = 55.0

    N = 256

    # 1. Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # 2. Identify vertices with self-loops (2*v is a square)
    # These cannot be in the set.
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)
    
    # Candidates are 1..N minus forbidden
    candidates = [v for v in range(1, N + 1) if v not in forbidden]
    
    # Map candidate values to 0-based indices for bitmask operations
    # This allows us to use Python integers as bitmasks efficiently.
    val_to_idx = {}
    idx_to_val = []
    for v in candidates:
        idx_to_val.append(v)
        val_to_idx[v] = len(idx_to_val) - 1
        
    num_candidates = len(candidates)
    
    # 3. Build adjacency bitmask
    # adj_mask[i] is a bitmask where bit j is set if candidate i and candidate j conflict (sum is square)
    adj_mask = [0] * num_candidates
    
    # To speed up, we can iterate over squares and find pairs
    # But N is small, O(N^2) is fine.
    for i in range(num_candidates):
        vi = idx_to_val[i]
        for j in range(i + 1, num_candidates):
            vj = idx_to_val[j]
            if vi + vj in squares:
                # Set bit j in mask i, and bit i in mask j
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute degrees for greedy initialization
    degrees = [bin(adj_mask[i]).count('1') for i in range(num_candidates)]
    
    # 4. Helper: Check if a bitmask represents a valid independent set
    # This is implicitly handled by the construction logic, but useful for verification.
    
    # 5. Greedy Initial Solution
    # Strategy: Pick vertex with min degree among remaining, add it, remove neighbors.
    def greedy_init():
        # remaining_mask: bits set for vertices still available
        remaining_mask = (1 << num_candidates) - 1
        selected_mask = 0
        
        while remaining_mask:
            # Find vertex in remaining_mask with minimum degree within remaining_mask
            best_idx = -1
            best_deg = float('inf')
            
            # Iterate through set bits in remaining_mask
            m = remaining_mask
            while m:
                # Get index of least significant set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                m ^= lsb
                
                # Count conflicts within remaining_mask
                # adj_mask[idx] & remaining_mask gives neighbors of idx that are still available
                deg = bin(adj_mask[idx] & remaining_mask).count('1')
                
                if deg < best_deg:
                    best_deg = deg
                    best_idx = idx
                    # Early break if degree 0 (isolated vertex)
                    if deg == 0:
                        break
            
            if best_idx == -1:
                break
                
            # Select best_idx
            selected_mask |= (1 << best_idx)
            
            # Remove best_idx and all its neighbors from remaining_mask
            # Neighbors are in adj_mask[best_idx]
            # We remove best_idx itself and its neighbors
            to_remove = (1 << best_idx) | adj_mask[best_idx]
            remaining_mask &= ~to_remove
            
        return selected_mask

    # 6. Branch and Bound Refinement
    # We use a recursive B&B that tries to add more vertices to a given starting set.
    # current_mask: bits set for vertices already in the independent set
    # available_mask: bits set for vertices that can potentially be added (not in current, not conflicting with current)
    
    def branch_and_bound(current_mask, available_mask, current_size, best_size_holder):
        """
        Recursive B&B.
        best_size_holder is a list [size, mask] to allow mutation in recursion.
        """
        # Pruning: If we can't beat the best, stop
        # Upper bound: current_size + number of bits in available_mask
        # A tighter bound could be used, but for N~240, bit count is a decent first pass.
        # Actually, a better bound is the size of the maximum independent set of the subgraph induced by available_mask.
        # For speed, we use a greedy bound or just the count.
        # Let's use a simple greedy upper bound on available_mask to prune better.
        
        # Greedy bound: repeatedly pick min-degree vertex from available, remove neighbors, count picks.
        # This is O(N^2) per node, which might be slow if tree is deep.
        # Given N is small, let's stick to a simpler bound first: popcount(available_mask).
        # If that's too loose, we'll see.
        
        if current_size + bin(available_mask).count('1') <= best_size_holder[0]:
            return

        if available_mask == 0:
            if current_size > best_size_holder[0]:
                best_size_holder[0] = current_size
                best_size_holder[1] = current_mask
            return

        # Heuristic: Pick a vertex to branch on.
        # Choose vertex with max degree in available subgraph? Or min?
        # Max degree usually prunes more because it removes more neighbors.
        best_idx = -1
        max_deg = -1
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            deg = bin(adj_mask[idx] & available_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_idx = idx
        
        if best_idx == -1:
             # Should not happen if available_mask != 0
             return

        # Branch 1: Include best_idx
        # New available: remove best_idx and its neighbors from available_mask
        new_available_include = available_mask & ~( (1 << best_idx) | adj_mask[best_idx] )
        new_current_mask = current_mask | (1 << best_idx)
        branch_and_bound(new_current_mask, new_available_include, current_size + 1, best_size_holder)
        
        # Check time
        if time.time() > DEADLINE:
            return

        # Branch 2: Exclude best_idx
        # New available: remove best_idx from available_mask
        new_available_exclude = available_mask & ~(1 << best_idx)
        branch_and_bound(current_mask, new_available_exclude, current_size, best_size_holder)

    # 7. Local Improvement (Add-only)
    def local_improve(mask):
        # Try to add any vertex from available that doesn't conflict
        # This is a fast polish.
        current_mask = mask
        # Calculate available: all candidates not in current and not conflicting with current
        # Conflicting with current: union of adj_mask[v] for v in current
        # This is expensive to compute from scratch. 
        # Better: iterate through candidates not in current, check if (adj_mask[v] & current_mask) == 0
        improved = True
        while improved:
            improved = False
            for i in range(num_candidates):
                if (current_mask >> i) & 1:
                    continue
                # Check if i conflicts with any in current_mask
                if (adj_mask[i] & current_mask) == 0:
                    current_mask |= (1 << i)
                    improved = True
        return current_mask

    # 8. Main Search Loop
    best_size = 0
    best_mask = 0
    
    # Get initial solution
    init_mask = greedy_init()
    init_mask = local_improve(init_mask)
    best_size = bin(init_mask).count('1')
    best_mask = init_mask
    
    # Prepare holder for B&B
    best_holder = [best_size, best_mask]
    
    # Run B&B from initial solution
    # Available for B&B: vertices not in init_mask and not conflicting with init_mask
    # We need to compute the "available" set correctly.
    # A vertex v is available if v not in current AND (adj_mask[v] & current_mask) == 0
    
    def get_available(current_mask):
        avail = 0
        for i in range(num_candidates):
            if (current_mask >> i) & 1:
                continue
            if (adj_mask[i] & current_mask) == 0:
                avail |= (1 << i)
        return avail

    initial_avail = get_available(best_mask)
    
    # Set recursion limit higher if needed, but N=256 is deep.
    # Python recursion limit is 1000 by default. Depth can be up to N.
    import sys
    sys.setrecursionlimit(10000)
    
    # Run B&B
    # We might want to limit the depth or use iterative deepening if it's too slow.
    # But for N=256, if the graph is sparse enough, it should be fast.
    # If it hangs, we need a timeout.
    
    # To handle timeout in recursion, we check time.time() in the recursive step.
    # However, deep recursion might not hit the check often.
    # Let's try it.
    
    start_bb = time.time()
    try:
        branch_and_bound(best_mask, initial_avail, best_size, best_holder)
    except RecursionError:
        pass
    except Exception:
        pass
        
    best_size = best_holder[0]
    best_mask = best_holder[1]
    
    # Iterated Local Search / B&B Restart
    # If we have time, perturb the best solution and run B&B again.
    while time.time() < DEADLINE - 2:
        
        # Perturb: remove a few random vertices from best_mask
        s_list = [i for i in range(num_candidates) if (best_mask >> i) & 1]
        if not s_list:
            break
            
        perturb_count = max(1, len(s_list) // 10)
        if random.random() < 0.5:
            # Random restart
            curr_mask = greedy_init()
        else:
            # Perturb current best
            curr_mask = best_mask
            for _ in range(perturb_count):
                if not curr_mask:
                    break
                # Pick random set bit
                # Get list of set bits
                bits = [i for i in range(num_candidates) if (curr_mask >> i) & 1]
                if not bits:
                    break
                idx_to_remove = random.choice(bits)
                curr_mask &= ~(1 << idx_to_remove)
        
        # Local improve
        curr_mask = local_improve(curr_mask)
        curr_size = bin(curr_mask).count('1')
        
        # If this perturbed solution is already better, update best
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            continue # Skip B&B for this one? Or run it?
            
        # Run B&B from this state
        avail = get_available(curr_mask)
        holder = [best_size, best_mask]
        
        # Limit B&B time per iteration to ensure we finish
        bb_start = time.time()
        # We can't easily interrupt the recursive B&B with a time limit other than checking inside.
        # The check inside branch_and_bound uses the global DEADLINE.
        # We should ideally pass a deadline to the B&B function.
        # For now, we rely on the global DEADLINE check in branch_and_bound.
        
        try:
            branch_and_bound(curr_mask, avail, curr_size, holder)
        except RecursionError:
            pass
        except Exception:
            pass
            
        best_size = holder[0]
        best_mask = holder[1]
        
        # If no improvement in a while, we might want to stop or restart more aggressively
        # But the loop condition handles the overall time.

    # Final polish
    best_mask = local_improve(best_mask)
    
    # Convert mask back to values
    result = []
    for i in range(num_candidates):
        if (best_mask >> i) & 1:
            result.append(idx_to_val[i])
            
    if not result:
        # Fallback to a single valid element
        # Find any v in 1..256 such that 2v is not square
        for v in range(1, 257):
            if 2*v not in squares:
                return [v]
        return [1] # Should not happen

    return sorted(result)

# EVOLVE-BLOCK-END

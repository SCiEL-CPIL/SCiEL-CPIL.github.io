import time
import random
import math

DEADLINE = 115  # Safety margin for 120s limit

def _compute_score_fast(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles using bitmask intersection.
    adj_masks: list of integers, adj_masks[i] has bit j set if i-j is an edge.
    degs: list of degrees.
    """
    max_sum = 0
    # Iterate over all potential edges (u, v)
    for u in range(n):
        adj_u = adj_masks[u]
        # Only consider v > u to avoid double counting and self-loops
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                # Common neighbors of u and v
                common = adj_u & adj_masks[v]
                # Iterate through set bits in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _initial_construction():
    """Base construction: K_{8,8} plus one internal edge (0,1)."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    edges = []
    
    # Add K_{8,8} edges
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
            edges.append((u, v))
            
    # Add one internal edge to create triangles
    u, v = 0, 1
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    degs[u] += 1
    degs[v] += 1
    edges.append((u, v))
    
    return adj_masks, degs, edges

def _state_to_edges(adj_masks):
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute all pairs for random sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_adj_masks = None
    
    # Run multiple restarts
    # We will alternate between starting from the base construction 
    # and slightly perturbed versions to escape local minima.
    
    num_restarts = 1000 # Upper bound, will be limited by time
    
    for restart_idx in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Initialize state
        adj_masks, degs, edges_list = _initial_construction()
        
        # Optional: Apply small random perturbations to escape the exact same local minimum
        # For the first few restarts, keep it pure. Later, perturb.
        if restart_idx > 5:
             # Swap 2 random edges to perturb
             for _ in range(2):
                 if len(edges_list) < 2: break
                 # Pick random edge to remove
                 rem_idx = random.randint(0, len(edges_list) - 1)
                 rem_u, rem_v = edges_list[rem_idx]
                 # Pick random non-edge to add
                 found = False
                 for _ in range(20):
                     add_u, add_v = random.choice(all_pairs)
                     if not ((adj_masks[add_u] >> add_v) & 1):
                         found = True
                         break
                 if not found: continue
                 
                 # Perform swap
                 adj_masks[rem_u] &= ~(1 << rem_v)
                 adj_masks[rem_v] &= ~(1 << rem_u)
                 degs[rem_u] -= 1
                 degs[rem_v] -= 1
                 edges_list.pop(rem_idx)
                 
                 adj_masks[add_u] |= (1 << add_v)
                 adj_masks[add_v] |= (1 << add_u)
                 degs[add_u] += 1
                 degs[add_v] += 1
                 edges_list.append((add_u, add_v))
        
        current_score = _compute_score_fast(n, adj_masks, degs)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = adj_masks[:]
            
        if best_score <= 23:
            # Likely optimal or near-optimal, can break early if we want, 
            # but let's keep searching for a bit to be sure
            pass

        # Local Search
        # We use a simple hill climbing with random restarts (the outer loop handles restarts).
        # To be more effective, we allow some "sideways" moves or use simulated annealing?
        # Given the discrete nature and small n, strict hill climbing with good restarts is often sufficient.
        # However, to escape shallow minima, we can try to prioritize removing edges in high-score triangles.
        
        stuck_count = 0
        max_stuck = 500 # If we don't improve in 500 steps, break and restart
        
        while time.time() - start_time < DEADLINE:
            # If we have reached the theoretical lower bound or very close, we might stop
            if current_score <= 23:
                # Let's do a few more steps just in case, but mostly stop
                if random.random() < 0.1:
                    break
                # Otherwise continue to try to find 22? Unlikely.
            
            # Select edge to remove
            # Bias: Pick an edge that is part of a high-scoring triangle
            # For simplicity and speed, we just pick a random edge.
            # If we want to be smarter:
            # 1. Find all triangles.
            # 2. Find the max score triangle.
            # 3. Pick one of its edges to remove.
            # This requires O(E*k) to find triangles, which is what _compute_score_fast does.
            # Let's just pick a random edge to keep the inner loop fast.
            
            rem_idx = random.randint(0, len(edges_list) - 1)
            rem_u, rem_v = edges_list[rem_idx]
            
            # Select edge to add
            # Pick a random non-edge
            add_u, add_v = None, None
            for _ in range(10): # Retry a few times if we hit an edge
                cand_u, cand_v = random.choice(all_pairs)
                if not ((adj_masks[cand_u] >> cand_v) & 1):
                    add_u, add_v = cand_u, cand_v
                    break
            
            if add_u is None:
                # Graph is complete? Impossible with 65 edges on 16 nodes (max 120)
                continue

            # Perform Swap
            # 1. Remove (rem_u, rem_v)
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edges_list.pop(rem_idx)
            
            # 2. Add (add_u, add_v)
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edges_list.append((add_u, add_v))
            
            # Evaluate
            new_score = _compute_score_fast(n, adj_masks, degs)
            
            if new_score < current_score:
                current_score = new_score
                stuck_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj_masks = adj_masks[:]
            else:
                # Revert
                # Remove (add_u, add_v)
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edges_list.pop()
                
                # Add (rem_u, rem_v)
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edges_list.append((rem_u, rem_v))
                
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
                    
    # Return the best result found
    if best_adj_masks is None:
        # Fallback: return the base construction
        adj_masks, degs, edges_list = _initial_construction()
        best_adj_masks = adj_masks
        
    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}

import math
import random

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 1. Identify self-conflicting nodes (2*x is a square)
    self_conflict = [False] * (n_max + 1)
    for i in range(1, n_max + 1):
        if is_perfect_square(2 * i):
            self_conflict[i] = True

    # Precompute conflict masks
    conflict_mask = [0] * (n_max + 1)
    for i in range(1, n_max + 1):
        if self_conflict[i]:
            continue
        mask = 0
        for j in range(1, n_max + 1):
            if i != j and is_perfect_square(i + j):
                mask |= (1 << j)
        conflict_mask[i] = mask

    valid_nodes = [i for i in range(1, n_max + 1) if not self_conflict[i]]
    num_valid = len(valid_nodes)
    
    # Helper to get size of mask
    def popcount(x):
        return bin(x).count('1')

    # Greedy MIS
    def greedy_mis():
        remaining = set(valid_nodes)
        selected_mask = 0
        while remaining:
            # Find node with min degree in remaining
            best_node = None
            min_deg = float('inf')
            rem_mask = 0
            for v in remaining:
                rem_mask |= (1 << v)
            
            for v in remaining:
                neigh = conflict_mask[v] & rem_mask
                deg = popcount(neigh)
                if deg < min_deg:
                    min_deg = deg
                    best_node = v
            
            if best_node is None:
                break
            selected_mask |= (1 << best_node)
            # Remove best_node and its neighbors
            neighbors = conflict_mask[best_node] | (1 << best_node)
            new_remaining = set()
            for v in remaining:
                if not (neighbors & (1 << v)):
                    new_remaining.add(v)
            remaining = new_remaining
        return selected_mask

    # Local Search with kicks
    def local_search(start_mask, max_iter=2000):
        current_mask = start_mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        for _ in range(max_iter):
            improved = False
            
            # 1. Try to add a node
            for v in valid_nodes:
                if (current_mask & (1 << v)) == 0:
                    if (conflict_mask[v] & current_mask) == 0:
                        current_mask |= (1 << v)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        improved = True
                        break 
                        
            if improved:
                continue
                
            # 2. Try 1-swap
            found_swap = False
            for v in valid_nodes:
                if (current_mask & (1 << v)) == 0:
                    conflicts_with_S = conflict_mask[v] & current_mask
                    if conflicts_with_S == 0:
                        continue
                    if conflicts_with_S & (conflicts_with_S - 1) == 0:
                        u_bit = conflicts_with_S
                        current_mask = (current_mask ^ u_bit) | (1 << v)
                        found_swap = True
                        break
                        
            if found_swap:
                # Size didn't change, but configuration did. 
                # We don't update best_size here unless we later improve.
                # But we should continue to see if we can add more.
                # However, the loop restarts, so it will try to add again.
                continue

            # 3. Kick: Remove 1-3 random nodes, then greedily fill
            # This helps escape local minima
            bits = []
            temp = current_mask
            while temp:
                lsb = temp & -temp
                bits.append(lsb)
                temp ^= lsb
                
            if bits:
                # Remove a random subset of size 1 to 3
                kick_size = min(len(bits), random.randint(1, 3))
                to_remove = random.sample(bits, kick_size)
                new_mask = current_mask
                for b in to_remove:
                    new_mask ^= b
                new_size = popcount(new_mask)
                
                # Greedily add back
                # Sort valid nodes by some heuristic? Or just iterate?
                # Iterating in fixed order is fast.
                for v in valid_nodes:
                    if (new_mask & (1 << v)) == 0:
                        if (conflict_mask[v] & new_mask) == 0:
                            new_mask |= (1 << v)
                
                new_size = popcount(new_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                
                # Accept the kick if it doesn't reduce size, or with some probability if it does?
                # For simplicity and speed, accept if size is same or better.
                if new_size >= current_size:
                    current_mask = new_mask
                    current_size = new_size
                else:
                    # If size decreased, revert to previous state to avoid getting worse
                    pass
            else:
                break
                
        return best_mask

    # Run multiple restarts
    best_overall_mask = 0
    best_overall_size = 0
    
    # Initial greedy
    initial_mask = greedy_mis()
    initial_size = popcount(initial_mask)
    if initial_size > best_overall_size:
        best_overall_size = initial_size
        best_overall_mask = initial_mask

    # Try a few random restarts of local search from the greedy start
    # Or perturb the greedy start
    for _ in range(10):
        # Perturb: remove a few random nodes from greedy mask
        mask = initial_mask
        bits = []
        temp = mask
        while temp:
            lsb = temp & -temp
            bits.append(lsb)
            temp ^= lsb
        if bits:
            # Remove 2-5 random nodes
            kick_size = min(len(bits), random.randint(2, 5))
            to_remove = random.sample(bits, kick_size)
            for b in to_remove:
                mask ^= b
        
        result_mask = local_search(mask, max_iter=5000)
        res_size = popcount(result_mask)
        if res_size > best_overall_size:
            best_overall_size = res_size
            best_overall_mask = result_mask

    # Convert bitmask to list
    result = []
    for i in range(1, n_max + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    # Final verification
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True

    if not is_valid(result):
        # Fallback to simple greedy
        result = []
        for i in range(1, n_max + 1):
            if self_conflict[i]: continue
            ok = True
            for j in result:
                if is_perfect_square(i + j):
                    ok = False
                    break
            if ok:
                result.append(i)
                
    return sorted(result)

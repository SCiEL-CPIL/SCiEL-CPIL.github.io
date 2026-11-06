"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill
    
    n = 256
    # Build conflict graph: vertex i (0-indexed, representing value i+1)
    # conflicts with vertex j if (i+1) + (j+1) is a perfect square
    # Also self-conflict if 2*(i+1) is a perfect square
    
    # Precompute which values are "self-conflicting" (2a is perfect square)
    self_conflict = [False] * n
    for i in range(n):
        val = 2 * (i + 1)
        r = int(math.isqrt(val))
        if r * r == val:
            self_conflict[i] = True
    
    # Build adjacency as bitsets (using Python integers as bitsets)
    # adj[i] is a bitmask of all vertices j that conflict with i
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            s = (i + 1) + (j + 1)
            r = int(math.isqrt(s))
            if r * r == s:
                mask |= (1 << j)
        adj[i] = mask
    
    # Remove self-conflicting vertices from consideration
    # They can never be in any valid set
    available_mask = 0
    for i in range(n):
        if not self_conflict[i]:
            available_mask |= (1 << i)
    
    # Branch and bound for maximum independent set
    # We use a greedy upper bound: repeatedly pick the vertex with minimum degree
    # in the remaining subgraph
    
    best_set = [0] * n  # bitset of best solution found
    best_size = 0
    
    def count_bits(x):
        return bin(x).count('1')
    
    def greedy_upper_bound(candidates, adj_list):
        """Greedy independent set size as upper bound"""
        if candidates == 0:
            return 0
        remaining = candidates
        count = 0
        while remaining:
            # Find vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                # Degree of v in remaining subgraph
                deg = count_bits(adj_list[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lb
            if best_v == -1:
                break
            count += 1
            # Remove best_v and all its neighbors from remaining
            remaining &= ~(adj[best_v] | (1 << best_v))
        return count
    
    def branch_and_bound(candidates, current_set, current_size, depth):
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Upper bound: current_size + greedy upper bound of remaining
        ub = current_size + greedy_upper_bound(candidates, adj)
        if ub <= best_size:
            return
        
        # Choose vertex to branch on: minimum degree in candidates
        best_v = -1
        best_deg = n + 1
        temp = candidates
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = count_bits(adj[v] & candidates)
            if deg < best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_candidates = candidates & ~(adj[best_v] | (1 << best_v))
        current_set[best_v] = 1
        branch_and_bound(new_candidates, current_set, current_size + 1, depth + 1)
        current_set[best_v] = 0
        
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        branch_and_bound(new_candidates, current_set, current_size, depth + 1)
    
    # Initialize with greedy solution to get a good lower bound
    def greedy_init():
        remaining = available_mask
        sol = [0] * n
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = count_bits(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lb
            if best_v == -1:
                break
            sol[best_v] = 1
            remaining &= ~(adj[best_v] | (1 << best_v))
        return sol
    
    init_sol = greedy_init()
    best_size = count_bits(sum(1 << i for i in range(n) if init_sol[i]))
    best_set = init_sol[:]
    
    # Run branch and bound
    branch_and_bound(available_mask, [0] * n, 0, 0)
    
    # Extract solution
    result = []
    for i in range(n):
        if best_set[i]:
            result.append(i + 1)
    
    if not result:
        # Fallback: return something valid
        result = [i + 1 for i in range(n) if not self_conflict[i]]
    
    return result
# EVOLVE-BLOCK-END

import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute clique and independent set status for all masks
    # is_clique[m]: True if mask induces a clique
    # is_indie[m]: True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_indie = [False] * (1 << n)
    
    is_clique[0] = True
    is_indie[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        # v connected to all in rest means (adj[v] & rest) == rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check indie: v must NOT be connected to any in rest, and rest must be indie
        # v not connected to any in rest means (adj[v] & rest) == 0
        if is_indie[rest] and (adj[v] & rest) == 0:
            is_indie[mask] = True

    # Memoization for (mask, k)
    # Value: None if not visited, True if solvable, False if not solvable
    # We also need to store the partition choice for reconstruction
    memo = {}
    choice = {} # (mask, k) -> subset S used in the first step
    
    def can_cover(mask, k):
        if (mask, k) in memo:
            return memo[(mask, k)]
        
        if time.time() > deadline:
            # Fallback to a simple greedy if we run out of time?
            # For now, just return False to stop search, main loop will handle timeout
            return False

        if mask == 0:
            memo[(mask, k)] = True
            return True
        if k == 0:
            memo[(mask, k)] = False
            return False
            
        # If the remaining mask itself is a clique or indie, we can cover it with 1 part (if k>=1)
        if k >= 1 and (is_clique[mask] or is_indie[mask]):
            memo[(mask, k)] = True
            choice[(mask, k)] = mask
            return True
            
        # Pick a vertex v in mask to anchor the search
        # Heuristic: pick vertex with highest degree in induced subgraph?
        # Or just lowest index for simplicity/symmetry
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # We need to find a valid subset S containing v such that S is clique or indie
        # and can_cover(mask ^ S, k-1) is True
        
        # Iterate over all subsets S of mask that contain v
        # S must be a subset of mask
        # Let rest_mask = mask ^ lsb
        # S = lsb | sub, where sub is a subset of rest_mask
        rest_mask = mask ^ lsb
        
        # To improve pruning, we can iterate sub in a specific order
        # But for n=14, 2^13 = 8192 subsets is small enough
        
        sub_iter = rest_mask
        while True:
            S = lsb | sub_iter
            if is_clique[S] or is_indie[S]:
                if can_cover(mask ^ S, k - 1):
                    memo[(mask, k)] = True
                    choice[(mask, k)] = S
                    return True
            if sub_iter == 0:
                break
            sub_iter = (sub_iter - 1) & rest_mask
            
        memo[(mask, k)] = False
        return False

    # Find minimal K
    # Lower bound: at least 1
    # Upper bound: n (each vertex its own part, which is a clique of size 1)
    best_k = n
    found = False
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_cover(full_mask, k):
            best_k = k
            found = True
            break
            
    if not found:
        # Fallback to greedy if exact search failed/timed out
        # Greedy: assign each vertex to first compatible part or new part
        parts = []
        for v in range(n):
            placed = False
            for i, part in enumerate(parts):
                # Check if v can be added to part
                # part is a list of vertices
                # Check clique or indie property
                new_part = part + [v]
                m = 0
                for u in new_part:
                    m |= (1 << u)
                if is_clique[m] or is_indie[m]:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    # Reconstruct partition
    parts = []
    current_mask = full_mask
    remaining_k = best_k
    
    while current_mask != 0:
        S = choice[(current_mask, remaining_k)]
        # Convert S bitmask to list of vertices
        vertices = [i for i in range(n) if (S >> i) & 1]
        parts.append(vertices)
        current_mask ^= S
        remaining_k -= 1
        
    # The parts are in reverse order of selection, but order doesn't matter for validity
    # Ensure types are Python int and list
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])
        
    return final_parts

if __name__ == "__main__":
    # This block is not executed by the evaluator, but good for local testing
    res = construct()
    print(res)

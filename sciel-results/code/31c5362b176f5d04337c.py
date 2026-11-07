import time
import sys

# Increase recursion limit if necessary, though n=14 is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit
    
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Precompute: for each vertex, its neighbors and non-neighbors (excluding self)
    # This might help in quick checks, but adjacency matrix is O(1).

    # Calculate Lower Bound
    # Max Independent Set (approx or exact for small n)
    # Max Clique (approx or exact for small n)
    # Since n=14, we can find exact Max Independent Set and Max Clique easily.
    
    def max_independent_set_size():
        # Bron-Kerbosch variant or simple DFS
        # For n=14, simple bitmask DP or DFS is fine.
        # Let's use a simple recursive search for max independent set size
        best = 0
        def dfs_is(idx, current_size, chosen):
            nonlocal best
            if idx == n:
                if current_size > best:
                    best = current_size
                return
            # Option 1: Don't choose idx
            dfs_is(idx + 1, current_size, chosen)
            # Option 2: Choose idx if compatible
            compatible = True
            for v in chosen:
                if adj[idx][v]:
                    compatible = False
                    break
            if compatible:
                dfs_is(idx + 1, current_size + 1, chosen + [idx])
        
        dfs_is(0, 0, [])
        return best

    def max_clique_size():
        best = 0
        def dfs_cq(idx, current_size, chosen):
            nonlocal best
            if idx == n:
                if current_size > best:
                    best = current_size
                return
            # Option 1: Don't choose idx
            dfs_cq(idx + 1, current_size, chosen)
            # Option 2: Choose idx if compatible
            compatible = True
            for v in chosen:
                if not adj[idx][v]:
                    compatible = False
                    break
            if compatible:
                dfs_cq(idx + 1, current_size + 1, chosen + [idx])
        
        dfs_cq(0, 0, [])
        return best

    alpha = max_independent_set_size()
    omega = max_clique_size()
    
    lb = max((n + alpha - 1) // alpha, (n + omega - 1) // omega)
    
    # Also, trivial lower bound is 1.
    # Another bound: If graph is complete, z=1. If empty, z=1.
    
    # Try k from lb to n
    for k in range(lb, n + 1):
        if time.time() > deadline:
            # Fallback to greedy if timeout
            return greedy_partition()
            
        # State for DFS
        # parts: list of lists of vertices
        # part_types: list of 'C', 'I', or 'M' (Mixed/Invalid - should not happen)
        # Actually, if a part has size 1, type is 'A' (Any).
        # If size > 1, type is 'C' or 'I'.
        
        # To prune, we can track which vertices are used.
        
        def can_add(v, part, ptype):
            if not part:
                return True
            if len(part) == 1:
                # Can be either, so always True for now, type will be set
                return True
            if ptype == 'C':
                # v must be adjacent to all in part
                for u in part:
                    if not adj[v][u]:
                        return False
                return True
            elif ptype == 'I':
                # v must be non-adjacent to all in part
                for u in part:
                    if adj[v][u]:
                        return False
                return True
            else:
                return False

        def get_type(part):
            if not part:
                return 'A'
            if len(part) == 1:
                return 'A'
            # Check if clique
            is_c = True
            is_i = True
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if not adj[part[i]][part[j]]:
                        is_c = False
                    if adj[part[i]][part[j]]:
                        is_i = False
            if is_c:
                return 'C'
            elif is_i:
                return 'I'
            else:
                return 'M' # Invalid

        # DFS variables
        parts = [[] for _ in range(k)]
        # We will fill parts in order.
        # To break symmetry, we enforce that part i is "filled" before part i+1 starts?
        # No, standard symmetry breaking: 
        # 1. The first vertex assigned to part i must be smaller than the first vertex assigned to part i+1? 
        #    This is hard to implement dynamically.
        # 2. Simpler: Only allow assigning vertex v to part j if:
        #    - j < k-1 and part j is not empty? No.
        #    - We can just iterate j from 0 to k-1.
        #    - To avoid permutations, we can require that if we assign v to part j, 
        #      and part j was empty, then all parts j+1..k-1 must also be empty.
        #      This ensures parts are created in order 0, 1, 2...
        
        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # All vertices assigned
                # Check if all parts are valid (non-empty? No, problem says "nonempty lists" in contract? 
                # "partition of vertices 0..13 into nonempty lists"
                # So we must use exactly k parts, and all must be non-empty.
                # My DFS creates k slots. If some are empty, it's invalid.
                # However, if I allow empty parts, I can just return the non-empty ones.
                # But the objective is to MINIMIZE number of parts.
                # If I find a solution with k slots, but some are empty, the actual number of parts is < k.
                # Since I'm iterating k upwards, if I find a solution with k slots, 
                # and some are empty, it means a solution with < k parts exists.
                # But I would have found that in a previous iteration of k.
                # So if I reach here with k slots, I should check if all are non-empty?
                # Actually, if I allow empty slots, I might find a "solution" for k=5 
                # that only uses 3 parts. But I would have found that at k=3.
                # So, to be safe and efficient, I should require all k parts to be non-empty 
                # OR just return the non-empty parts and let the caller check count.
                # Given I iterate k from lb, the first valid configuration (even with empty slots)
                # implies the true min is <= k.
                # But I want the exact min.
                # Let's enforce: All k parts must be non-empty.
                # This prunes branches where we "waste" a part.
                for p in parts:
                    if not p:
                        return False
                return True

            v = idx
            # Try assigning v to each part j
            for j in range(k):
                # Symmetry breaking: 
                # If part j is empty, we can only use it if all subsequent parts are also empty.
                # This prevents (part0: [1], part1: []) and (part0: [], part1: [1]) both being explored as distinct.
                # Actually, standard trick: 
                # Only assign to part j if:
                # 1. j is the first empty part (i.e., parts[j] is empty, and parts[j+1..] are empty)
                #    OR
                # 2. parts[j] is not empty.
                
                if not parts[j]:
                    # Check if this is the first empty part
                    is_first_empty = True
                    for m in range(j+1, k):
                        if parts[m]:
                            is_first_empty = False
                            break
                    if not is_first_empty:
                        continue
                    # Also, if we are opening a new part, we can do so.
                
                if can_add(v, parts[j], get_type(parts[j])):
                    # Add v to part j
                    parts[j].append(v)
                    
                    # Check if part j is now invalid (Mixed)
                    if get_type(parts[j]) != 'M':
                        if dfs(idx + 1):
                            return True
                    
                    parts[j].pop()
            
            return False

        # Run DFS
        if dfs(0):
            # Extract non-empty parts
            result = [p for p in parts if p]
            # Verify validity
            if len(result) == k:
                return result
            # If result has fewer parts, it's still valid, but we should have found it earlier?
            # If we enforce non-empty, len(result) must be k.
            return result

    # If no solution found (should not happen), fallback to greedy
    return greedy_partition()

def greedy_partition():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

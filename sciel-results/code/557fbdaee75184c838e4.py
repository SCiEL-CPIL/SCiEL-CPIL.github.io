import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_n = 17
    best_edges = _paley_17_edges()
    
    # Try to find larger graphs
    for n in range(35, best_n, -1):
        if time.time() > deadline - 10:
            break
        result = _local_search(n, deadline)
        if result:
            best_n = n
            best_edges = result
            # If we found a large one, we can stop or continue to see if we can go higher?
            # The task says maximize n. So if we find 35, that's the max possible known?
            # Actually, if we find 35, we should check if 36 is possible?
            # But 36 is unlikely.
            # Let's just break if we find a high one, or continue if we have time.
            # Given the difficulty, finding 35 is a big win.
            # Let's try to find the max possible in the remaining time.
            # But local search for larger n is harder.
            # Let's just return the first valid one found for a high n?
            # No, we should try to maximize.
            # If we find n=20, we should try n=21, etc?
            # It's better to try high n first. If it fails, try lower.
            # So the loop `for n in range(35, best_n, -1)` is good.
            # If we find a valid graph for n=35, we break.
            # If we don't find 35, we try 34, etc.
            # We should update best_n and best_edges whenever we find a valid one.
            # And continue to try higher n? No, we are going down.
            # So if we find 34, we have a valid 34. We break because we are looking for max?
            # No, if we start at 35 and fail, then try 34 and succeed, we have 34.
            # We should not try 33 because 34 is better.
            # So we break as soon as we find a valid graph for the current n.
            break

    return {'n': best_n, 'edges': best_edges}

def _paley_17_edges():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return edges

def _local_search(n, deadline):
    # Try multiple restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search
        for step in range(10000):
            if time.time() > deadline:
                return None
            
            # Check for violations
            viol = _find_violation(adj, n)
            if viol is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return edges
            
            # Flip an edge in the violation
            # viol is a tuple of vertices forming the bad subgraph
            # Pick a random edge from the subgraph to flip
            verts = viol
            # For K4, any edge removal works. For Ind6, any edge addition works.
            # We need to know which type it was.
            # Let's return (type, verts)
            # If type is 'K4', remove edge. If 'IND6', add edge.
            pass
            
        # If we didn't find a solution in this restart, try another
    return None

def _find_violation(adj, n):
    # Check for K4
    # Iterate over pairs (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            # Common neighbors
            common = adj[u] & adj[v]
            # Check if common has any edge
            # Iterate over bits in common
            c1 = common & (common - 1)
            if c1:
                # There are at least 2 common neighbors
                # Check if any two are connected
                # Let w be the first bit
                w = (c1 & -c1).bit_length() - 1
                # Check if w is connected to any other in common
                # Actually, we just need to find an edge in common.
                # Let's iterate
                mask = common
                while mask:
                    bit = mask & -mask
                    w = bit.bit_length() - 1
                    mask ^= bit
                    # Check if w is connected to any other in common
                    # common & adj[w]
                    # If (common & adj[w]) != 0, then there is a K4
                    # But we must ensure we don't pick w itself?
                    # adj[w] doesn't have w.
                    # So if common & adj[w] is non-zero, there is a neighbor x of w in common.
                    # Then u, v, w, x form a K4.
                    if common & adj[w]:
                        # Find x
                        x_mask = common & adj[w]
                        x_bit = x_mask & -x_mask
                        x = x_bit.bit_length() - 1
                        return ('K4', [u, v, w, x])
    # Check for Independent Set of 6
    # This is harder.
    # We can use a simple backtracking to find an independent set of size 6.
    # Or, we can check the complement for K6.
    # Let's try a greedy approach first to find a candidate, then verify?
    # No, we need to be exact.
    # Backtracking for Ind 6:
    # Order vertices by degree?
    # Simple DFS
    def dfs(ind_set, candidates):
        if len(ind_set) == 6:
            return ind_set
        if not candidates:
            return None
        # Pick a vertex from candidates
        # To be efficient, pick one with fewest connections to other candidates?
        # Or just iterate
        for i in range(len(candidates)):
            v = candidates[i]
            # Check if v is independent from ind_set
            # v must not be connected to any in ind_set
            ok = True
            for u in ind_set:
                if adj[v] & (1 << u):
                    ok = False
                    break
            if not ok:
                continue
            
            # New candidates are those in candidates[i+1:] that are not connected to v
            new_cands = []
            for j in range(i + 1, len(candidates)):
                u = candidates[j]
                if not (adj[v] & (1 << u)):
                    new_cands.append(u)
            
            res = dfs(ind_set + [v], new_cands)
            if res:
                return res
        return None
    
    res = dfs([], list(range(n)))
    if res:
        return ('IND6', res)
        
    return None

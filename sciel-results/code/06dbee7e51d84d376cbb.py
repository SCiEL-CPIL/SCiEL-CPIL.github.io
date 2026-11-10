import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
            
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_path_3(adj, s, t):
        # Check if there is a path s-u-v-t in adj
        # s != t
        if s == t:
            return False
        # u must be neighbor of s
        for u in adj[s]:
            if u == t:
                continue
            # v must be neighbor of u and t
            # and v != s
            for v in adj[u]:
                if v == s or v == t:
                    continue
                if t in adj[v]:
                    return True
        return False

    def is_valid_incremental(curr_deg, curr_adj, new_edge):
        # curr_deg, curr_adj are BEFORE adding new_edge
        u, v = new_edge
        # 1. Calculate new degrees
        new_deg = curr_deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        # 2. Identify pairs with equal new degrees
        # We only need to check pairs that might have paths.
        # Since graph was valid before, any pair with equal OLD degrees had no path.
        # But their degrees might have changed?
        # Only u, v changed.
        # So pairs not involving u, v have same old degrees.
        # If they had equal old degrees, they had no path.
        # Did we add a path?
        # Yes, if new edge is part of a path.
        # So we MUST check all pairs with equal NEW degrees.
        
        # Group by new degree
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[new_deg[i]].append(i)
            
        # Build new adj (for path check)
        # We can just simulate the check using curr_adj + new_edge
        # To avoid copying, we can pass a flag or just create a small wrapper.
        # But creating new_adj is cheap (15 sets).
        new_adj = [s.copy() for s in curr_adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        
        for deg_val, verts in groups.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s, t = verts[i], verts[j]
                    if has_path_3(new_adj, s, t):
                        return False
        return True

    # Initial state
    best_edges = set(base_edges)
    best_deg = get_degrees(best_edges)
    best_adj = build_adj(best_edges)
    best_count = len(best_edges)
    
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append(tuple(sorted((A[i], A[j]))))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append(tuple(sorted((B[i], B[j]))))
            
    # Greedy Addition
    curr = set(base_edges)
    curr_deg = get_degrees(curr)
    curr_adj = build_adj(curr)
    
    # Sort candidates to prioritize those that might be safer?
    # Randomize or fixed order.
    random.shuffle(candidates)
    
    for c in candidates:
        if c in curr:
            continue
        if is_valid_incremental(curr_deg, curr_adj, c):
            u, v = c
            curr.add(c)
            curr_deg[u] += 1
            curr_deg[v] += 1
            curr_adj[u].add(v)
            curr_adj[v].add(u)
            
    if len(curr) > best_count:
        best_edges = curr
        best_deg = curr_deg
        best_adj = curr_adj
        best_count = len(curr)
        
    # Local Search: Add one, Remove one
    # Use incremental check for the "Add" part?
    # Removing an edge changes degrees of its endpoints.
    # This breaks the "incremental add" assumption slightly.
    # But we can still use the full check or a modified incremental check.
    # Given time constraints, let's stick to a faster full check or optimized incremental.
    
    def is_valid_full(edges, deg, adj):
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Optimized Local Search
    max_iter = 2000 # Can afford more if fast
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # Try to add an edge
        for add_e in candidates:
            if add_e in best_edges:
                continue
            # Check if adding is valid
            # We need to check if best_edges + add_e is valid.
            # Use incremental check?
            # best_edges is the current state.
            if is_valid_incremental(best_deg, best_adj, add_e):
                # Try to remove an edge to keep count same?
                # No, we want to INCREASE count.
                # So if we can add without removing, we do it.
                u, v = add_e
                best_edges.add(add_e)
                best_deg[u] += 1
                best_deg[v] += 1
                best_adj[u].add(v)
                best_adj[v].add(u)
                best_count += 1
                improved = True
                break
        if improved:
            continue
            
        # If no pure add, try swap (add one, remove one)
        # This is expensive.
        # Try random swaps?
        if time.time() > deadline - 5:
            break
            
        # Pick a random candidate to add
        add_e = random.choice(candidates)
        if add_e in best_edges:
            continue
            
        # Pick a random edge to remove
        rem_e = random.choice(list(best_edges))
        
        # Check if (best_edges - rem_e + add_e) is valid
        # We must check full validity because degrees of 4 vertices change.
        test_edges = (best_edges | {add_e}) - {rem_e}
        test_deg = get_degrees(test_edges)
        test_adj = build_adj(test_edges)
        if is_valid_full(test_edges, test_deg, test_adj):
            best_edges = test_edges
            best_deg = test_deg
            best_adj = test_adj
            # best_count unchanged
            
    result = [list(e) for e in best_edges]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    return result

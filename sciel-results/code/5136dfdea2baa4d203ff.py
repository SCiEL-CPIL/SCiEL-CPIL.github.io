import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Vertices 0..14
    # Base graph: K_{7,8}
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
    
    def is_valid_fast(edges, deg, adj):
        # Check all paths s-u-v-t using precomputed adj and deg
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

    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Initial best solution
    best_edges = set(base_edges)
    best_deg = get_degrees(best_edges)
    best_adj = build_adj(best_edges)
    best_count = len(best_edges)
    
    # Candidate edges to add: Internal to A, Internal to B
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append(tuple(sorted((A[i], A[j]))))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append(tuple(sorted((B[i], B[j]))))
    
    # Greedy addition with fast validity check
    curr = set(base_edges)
    curr_deg = get_degrees(curr)
    curr_adj = build_adj(curr)
    
    for c in candidates:
        if c in curr:
            continue
        u, v = c
        # Check if adding c is valid
        # New degrees
        new_deg = curr_deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        # New adj
        new_adj = [s.copy() for s in curr_adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        # Check only paths involving the new edge or affected vertices
        # Simplified: just check all paths for now
        test_edges = curr | {c}
        if is_valid_fast(test_edges, new_deg, new_adj):
            curr = test_edges
            curr_deg = new_deg
            curr_adj = new_adj
            
    if len(curr) > best_count:
        best_edges = curr
        best_deg = curr_deg
        best_adj = curr_adj
        best_count = len(curr)
    
    # Local search: 2-opt moves (add one, remove one)
    # Limit iterations for time safety
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for add_e in candidates:
            if add_e in best_edges:
                continue
            for rem_e in list(best_edges):
                test = (best_edges | {add_e}) - {rem_e}
                if len(test) <= len(best_edges):
                    continue
                test_deg = get_degrees(test)
                test_adj = build_adj(test)
                if is_valid_fast(test, test_deg, test_adj):
                    best_edges = test
                    best_deg = test_deg
                    best_adj = test_adj
                    best_count = len(test)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final conversion
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are int
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result

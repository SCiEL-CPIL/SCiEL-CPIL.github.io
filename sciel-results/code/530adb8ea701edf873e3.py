import time
import random
import numpy as np
from itertools import combinations

def find_k4(adj):
    n = adj.shape[0]
    # Iterate over triangles
    # Optimization: Only check if i<j<k
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]: continue
            for k in range(j+1, n):
                if not adj[i, k] or not adj[j, k]: continue
                # Triangle i,j,k found. Check for 4th vertex
                # Candidates: vertices connected to i, j, k
                # We can check all other vertices
                for x in range(n):
                    if x == i or x == j or x == k: continue
                    if adj[i, x] and adj[j, x] and adj[k, x]:
                        return (i, j, k, x)
    return None

def find_i6(adj):
    n = adj.shape[0]
    # Greedy independent set finder
    # Heuristic: pick vertex with min degree in remaining graph
    remaining = list(range(n))
    indep_set = []
    
    # We need a set of size 6.
    # If we can't find 6, return None.
    # This greedy might fail even if alpha >= 6, but it's a good heuristic.
    # To be more robust, we can randomize the choice among min-degree vertices.
    
    while remaining:
        # Find min degree in induced subgraph of 'remaining'
        min_deg = n
        min_v = -1
        candidates = []
        for v in remaining:
            deg = 0
            for u in remaining:
                if v != u and adj[v, u]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_v = v
                candidates = [v]
            elif deg == min_deg:
                candidates.append(v)
        
        # Pick random from candidates
        v = random.choice(candidates)
        indep_set.append(v)
        
        if len(indep_set) == 6:
            return tuple(indep_set)
        
        # Remove v and its neighbors from remaining
        to_remove = set()
        to_remove.add(v)
        for u in remaining:
            if adj[v, u]:
                to_remove.add(u)
        
        remaining = [x for x in remaining if x not in to_remove]
        
        if len(indep_set) + len(remaining) < 6:
            return None
            
    return None if len(indep_set) < 6 else tuple(indep_set)

def is_valid(adj):
    n = adj.shape[0]
    # Check K4
    if find_k4(adj) is not None:
        return False
    # Check I6
    if find_i6(adj) is not None:
        return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Try n from 35 down to 17
    best_graph = None
    
    for n in range(35, 16, -1):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        # Time budget for this n
        time_budget = 10.0
        if n > 30:
            time_budget = 15.0
        elif n > 20:
            time_budget = 10.0
        else:
            time_budget = 5.0
            
        start_n = time.time()
        found = False
        
        # Local Search
        # Initialize random graph
        # Use p=0.5
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Check initial
        if is_valid(adj):
            found = True
        else:
            # Conflict Driven Local Search
            # We will loop until time runs out
            while time.time() - start_n < time_budget:
                # Find violation
                k4 = find_k4(adj)
                if k4:
                    # Break K4 by removing an edge
                    # Randomly pick an edge from the K4
                    u, v, w, x = k4
                    edges_to_remove = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    e = random.choice(edges_to_remove)
                    adj[e[0], e[1]] = 0
                    adj[e[1], e[0]] = 0
                    continue
                
                i6 = find_i6(adj)
                if i6:
                    # Break I6 by adding an edge
                    a, b, c, d, e, f = i6
                    verts = [a, b, c, d, e, f]
                    # Pick two random distinct vertices from the set
                    u, v = random.sample(verts, 2)
                    adj[u, v] = 1
                    adj[v, u] = 1
                    continue
                
                # No violations found?
                # This means is_valid should return True
                # But find_i6 is greedy, might miss I6?
                # find_k4 is exact.
                # If find_i6 returns None, it's a heuristic.
                # We should do a more rigorous check if we suspect validity.
                # For now, assume if find_k4 and find_i6 return None, it's likely valid.
                # To be safe, we can do a full check every 100 steps.
                # But let's assume the heuristic is good enough for "found" candidate.
                # We will verify before returning.
                
                # If we get here, we think it's valid.
                # Let's do a final check.
                # Note: find_i6 might be false negative.
                # Let's implement a more robust check for I6 if needed?
                # For now, let's try to return it.
                
                # Verify
                # We need a definitive check.
                # find_k4 is definitive.
                # For I6, we can use a backtracking search for I6?
                # Or just use the greedy one as a proxy and hope.
                # Better: If we find a candidate, run a definitive check.
                
                # Definitive check for I6:
                # Backtracking search for independent set of size 6.
                # This is expensive but we only do it when we think we have a solution.
                
                def check_i6_definitive(adj):
                    n = adj.shape[0]
                    def dfs(current, candidates):
                        if len(current) == 6:
                            return True
                        if len(current) + len(candidates) < 6:
                            return False
                        for i, v in enumerate(candidates):
                            # Check if v is independent from current
                            ok = True
                            for u in current:
                                if adj[u, v]:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            
                            new_current = current + [v]
                            new_candidates = []
                            for j in range(i+1, len(candidates)):
                                u2 = candidates[j]
                                if not adj[v, u2]:
                                    new_candidates.append(u2)
                            
                            if dfs(new_current, new_candidates):
                                return True
                        return False
                    return dfs([], list(range(n)))
                
                if not check_i6_definitive(adj):
                    # Valid!
                    found = True
                    break
                else:
                    # Still has I6, but greedy missed it?
                    # This implies our greedy find_i6 is not complete.
                    # We should restart or do a random flip to escape local opt.
                    # Random flip
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        adj[u, v] = 1 - adj[u, v]
                        adj[v, u] = adj[u, v]

        if found:
            # Convert adj to edges list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a known small valid graph
    # Paley 17
    q = 17
    edges = []
    for i in range(q):
        for j in range(i+1, q):
            diff = (j - i) % q
            if diff != 0 and pow(diff, (q-1)//2, q) == 1:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

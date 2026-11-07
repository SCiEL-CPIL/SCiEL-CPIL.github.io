"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def _is_valid(edges, n):
    """Check if the graph has no 3-edge path with equal-degree endpoints."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # For each pair (s,t) with deg(s)==deg(t), check if there's a 3-edge path s-u-v-t
    for s in range(n):
        for t in range(s+1, n):
            if deg[s] != deg[t]:
                continue
            # Check if there exists u,v: s~u, u~v, v~t, all distinct, u!=t, v!=s
            found = False
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[t]:
                    if v == s:
                        continue
                    if u == v:
                        continue
                    if v in adj[u]:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def _count_edges(edges):
    return len(edges)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = P['n']
    
    # Start with K_{7,8}
    A = list(range(7))   # part A, size 7
    B = list(range(7, 15))  # part B, size 8
    
    edges = []
    for a in A:
        for b in B:
            edges.append([a, b])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try adding intra-A edges
    intra_A = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            intra_A.append((A[i], A[j]))
    
    # Try adding intra-B edges
    intra_B = []
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            intra_B.append((B[i], B[j]))
    
    # Greedy: try adding each intra edge if it maintains validity
    candidate = list(best_edges)
    for e in intra_A + intra_B:
        u, v = e
        new_edges = candidate + [[u, v]]
        if _is_valid(new_edges, n):
            candidate = new_edges
    
    if len(candidate) > best_count:
        best_edges = candidate
        best_count = best_count if best_count > len(candidate) else len(candidate)
        best_count = len(candidate)
    
    # Local search: try removing an edge and re-adding different ones
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best_edges)
        current_set = set()
        for e in current:
            current_set.add((min(e[0],e[1]), max(e[0],e[1])))
        
        # Try removing each edge and see if we can add more
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:idx] + current[idx+1:]
            trial_set = set()
            for e in trial:
                trial_set.add((min(e[0],e[1]), max(e[0],e[1])))
            
            # Try to add back edges
            added_any = False
            for u in range(n):
                for v in range(u+1, n):
                    if (u,v) in trial_set:
                        continue
                    new_e = [u, v]
                    new_trial = trial + [new_e]
                    if _is_valid(new_trial, n):
                        trial = new_trial
                        trial_set.add((u,v))
                        added_any = True
            
            if len(trial) > len(best_edges):
                best_edges = trial
                best_count = len(trial)
                improved = True
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END

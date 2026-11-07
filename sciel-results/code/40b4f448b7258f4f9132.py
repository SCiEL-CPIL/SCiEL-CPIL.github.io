"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {(v, u): i for i, (u, v) in edge_map.items()}
    edge_index.update({(u, v): i for i, (u, v) in edge_map.items()})

    # Structural Initialization: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    current_edges = set()
    for u in A:
        for v in B:
            current_edges.add(edge_index[(u, v)])
            
    def is_valid(edge_set):
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all edges (u, v) for violating paths s-u-v-t
        for e in edge_set:
            u, v = edge_map[e]
            
            s_by_deg = {}
            for s in adj[u]:
                if s == v: continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u: continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            # Check intersection of degrees
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # We need s in S_set, t in T_set such that s != t.
                    # This fails ONLY if S_set == T_set == {x} for some x.
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    # Verify initial state
    if not is_valid(current_edges):
        star_edges = set()
        for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
        return [[u, v] for e in star_edges for u, v in [edge_map[e]]]

    best_edges = current_edges.copy()
    best_score = len(current_edges)
    
    # Generate list of missing internal edges
    missing_internal = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            u, v = A[i], A[j]
            idx = edge_index[(u, v)]
            if idx not in current_edges:
                missing_internal.append(idx)
                
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            idx = edge_index[(u, v)]
            if idx not in current_edges:
                missing_internal.append(idx)
                
    random.shuffle(missing_internal)
    
    # Greedy Add
    for idx in missing_internal:
        if time.time() > deadline:
            break
        new_edges = current_edges | {idx}
        if is_valid(new_edges):
            current_edges = new_edges
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            
    # Now try to add remaining missing edges
    all_missing = [i for i in range(total_edges) if i not in current_edges]
    random.shuffle(all_missing)
    
    for idx in all_missing:
        if time.time() > deadline:
            break
        new_edges = current_edges | {idx}
        if is_valid(new_edges):
            current_edges = new_edges
            best_edges = current_edges.copy()
            best_score = len(current_edges)

    # Local search with 1-swap and 2-swap moves
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try 1-swap moves: remove edge e1, add edge e2
        existing_edges = list(current_edges)
        missing_edges = [i for i in range(total_edges) if i not in current_edges]
        
        random.shuffle(existing_edges)
        random.shuffle(missing_edges)
        
        for e1 in existing_edges:
            if time.time() > deadline - 5:
                break
            for e2 in missing_edges:
                if time.time() > deadline - 5:
                    break
                trial_edges = current_edges - {e1}
                trial_edges.add(e2)
                
                if is_valid(trial_edges):
                    current_edges = trial_edges
                    best_edges = current_edges.copy()
                    best_score = len(current_edges)
                    improved = True
                    missing_edges = [i for i in range(total_edges) if i not in current_edges]
                    break
        
        # If no 1-swap improvement, try 2-swap: remove e1, e2, add e3, e4
        if not improved:
            existing_edges = list(current_edges)
            missing_edges = [i for i in range(total_edges) if i not in current_edges]
            random.shuffle(existing_edges)
            random.shuffle(missing_edges)
            
            # Limit 2-swap search to avoid explosion
            limit_e1 = min(20, len(existing_edges))
            limit_e3 = min(20, len(missing_edges))
            
            for i in range(limit_e1):
                if time.time() > deadline - 5:
                    break
                e1 = existing_edges[i]
                for j in range(i+1, limit_e1):
                    if time.time() > deadline - 5:
                        break
                    e2 = existing_edges[j]
                    for k in range(limit_e3):
                        if time.time() > deadline - 5:
                            break
                        e3 = missing_edges[k]
                        for l in range(k+1, limit_e3):
                            if time.time() > deadline - 5:
                                break
                            e4 = missing_edges[l]
                            
                            trial_edges = current_edges - {e1, e2}
                            trial_edges.add(e3)
                            trial_edges.add(e4)
                            
                            if is_valid(trial_edges):
                                current_edges = trial_edges
                                best_edges = current_edges.copy()
                                best_score = len(current_edges)
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break

        # If still no improvement, try random restarts of greedy add
        if not improved:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                if current_edges:
                    e_to_remove = random.choice(list(current_edges))
                    perturbed = current_edges - {e_to_remove}
                    
                    missing = [i for i in range(total_edges) if i not in perturbed]
                    random.shuffle(missing)
                    
                    for idx in missing:
                        if time.time() > deadline - 5:
                            break
                        new_edges = perturbed | {idx}
                        if is_valid(new_edges):
                            perturbed = new_edges
                    
                    if len(perturbed) > best_score:
                        current_edges = perturbed
                        best_edges = current_edges.copy()
                        best_score = len(current_edges)
                        improved = True

    # Final validation
    if not is_valid(best_edges):
         # Fallback
         star_edges = set()
         for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
         best_edges = star_edges

    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
        
    return result
# EVOLVE-BLOCK-END

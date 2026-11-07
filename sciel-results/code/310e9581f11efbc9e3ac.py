import time
import random

# EVOLVE-BLOCK-START

N = 15
MAX_DEG = 14
# Precompute all 3-edge simple paths (s, u, v, t) with s != t, and all distinct.
# A path is s-u-v-t where (s,u), (u,v), (v,t) are edges in the graph.
# s != t, and all four vertices are distinct (simple path of length 3).
# The constraint: for every such path present in the graph, deg(s) != deg(t).

# Precompute all possible 3-paths as tuples (s, u, v, t)
ALL_3_PATHS = []
for s in range(N):
    for u in range(N):
        if u == s: continue
        for v in range(N):
            if v == s or v == u: continue
            for t in range(N):
                if t == s or t == u or t == v: continue
                ALL_3_PATHS.append((s, u, v, t))

def compute_degrees(edges_set):
    deg = [0] * N
    for (u, v) in edges_set:
        deg[u] += 1
        deg[v] += 1
    return deg

def is_valid(edges_set):
    deg = compute_degrees(edges_set)
    for (s, u, v, t) in ALL_3_PATHS:
        if (s, u) in edges_set and (u, v) in edges_set and (v, t) in edges_set:
            if deg[s] == deg[t]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Start with a star graph (14 edges) - known valid
    best_edges = [(0, v) for v in range(1, N)]
    best_set = set(best_edges)
    best_count = len(best_edges)
    
    # Try to add edges greedily, then do local search
    # First, try all pairs in some order
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Try adding edges one by one in a smart order
    # Sort by some heuristic - maybe prefer edges between high-degree nodes?
    # Actually, let's just try all and pick the best valid one
    
    # Greedy: try adding each edge if it maintains validity
    current_edges = list(best_edges)
    current_set = set(best_edges)
    
    # Try adding edges in random order, multiple restarts
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        trial_edges = list(best_edges)
        trial_set = set(best_edges)
        
        # Shuffle the candidate edges to add
        candidates = [p for p in all_pairs if p not in trial_set]
        random.shuffle(candidates)
        
        for edge in candidates:
            u, v = edge
            new_set = trial_set | {edge}
            if is_valid(new_set):
                trial_set = new_set
                trial_edges.append(edge)
        
        if len(trial_edges) > best_count:
            best_edges = trial_edges
            best_set = trial_set
            best_count = len(best_edges)
    
    # Local search: try to add edges, or swap
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_set = set(best_edges)
        
        # Try adding any single edge
        for edge in all_pairs:
            if edge in current_set:
                continue
            new_set = current_set | {edge}
            if is_valid(new_set):
                best_edges.append(edge)
                current_set = new_set
                best_count += 1
                improved = True
                break
        
        if time.time() > deadline - 10:
            break
    
    # Try 1-move swaps: remove one edge, add another
    if time.time() < deadline - 5:
        current_set = set(best_edges)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try removing one edge and adding one or two others
            for rem in list(current_set):
                if time.time() > deadline - 5:
                    break
                new_set = current_set - {rem}
                # Try adding one edge
                for add in all_pairs:
                    if add in new_set:
                        continue
                    cand = new_set | {add}
                    if len(cand) > len(current_set) and is_valid(cand):
                        current_set = cand
                        improved = True
                        break
                if improved:
                    break
                # Try adding two edges
                for add1 in all_pairs:
                    if add1 in new_set:
                        continue
                    for add2 in all_pairs:
                        if add2 in new_set or add2 <= add1:
                            continue
                        cand = new_set | {add1, add2}
                        if len(cand) > len(current_set) and is_valid(cand):
                            current_set = cand
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    best_edges = list(current_set)
    best_count = len(best_edges)
    
    # Convert to proper format
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END

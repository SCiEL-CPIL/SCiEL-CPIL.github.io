import itertools
import time
import random

# Precompute all 8-vertex subsets
SUBSETS_8 = list(itertools.combinations(range(16), 8))

# Canonical Q3 adjacency list
CANON_Q3 = [
    [1, 2, 4],
    [0, 3, 5],
    [0, 6, 7],
    [1, 2, 7],
    [0, 5, 6],
    [1, 4, 7],
    [2, 4, 5],
    [3, 5, 6]
]

def check_subset_for_q3(edges_set, subset):
    """
    Check if the induced subgraph on 'subset' contains Q3 as a subgraph.
    """
    s = set(subset)
    adj_subset = {v: set() for v in subset}
    edge_count = 0
    for u, v in edges_set:
        if u in s and v in s:
            edge_count += 1
            adj_subset[u].add(v)
            adj_subset[v].add(u)
            
    if edge_count < 12:
        return False
        
    # Degree check: all vertices must have degree >= 3
    for v in subset:
        if len(adj_subset[v]) < 3:
            return False
            
    # Backtracking search
    verts = list(subset)
    index_map = {v: i for i, v in enumerate(verts)}
    
    adj_mat = [[False]*8 for _ in range(8)]
    for u in subset:
        i = index_map[u]
        for v in adj_subset[u]:
            j = index_map[v]
            adj_mat[i][j] = True
            adj_mat[j][i] = True
            
    mapping = [-1] * 8
    used = [False] * 8
    
    def backtrack(q3_idx):
        if q3_idx == 8:
            return True
        for sub_idx in range(8):
            if used[sub_idx]:
                continue
            ok = True
            for neighbor in CANON_Q3[q3_idx]:
                prev_map = mapping[neighbor]
                if prev_map != -1:
                    if not adj_mat[sub_idx][prev_map]:
                        ok = False
                        break
            if ok:
                mapping[q3_idx] = sub_idx
                used[sub_idx] = True
                if backtrack(q3_idx + 1):
                    return True
                mapping[q3_idx] = -1
                used[sub_idx] = False
        return False

    for start_sub in range(8):
        mapping[0] = start_sub
        used[start_sub] = True
        if backtrack(1):
            return True
        mapping[0] = -1
        used[start_sub] = False
        
    return False

def has_q3(edges_set):
    for subset in SUBSETS_8:
        if check_subset_for_q3(edges_set, subset):
            return True
    return False

def construct():
    deadline = time.time() + 110
    best_edges = set()
    best_count = 0
    
    all_edges = list(itertools.combinations(range(16), 2))
    
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        current_edges = set()
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            current_edges.add((u, v))
            if not has_q3(current_edges):
                pass
            else:
                current_edges.discard((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    return list(best_edges)

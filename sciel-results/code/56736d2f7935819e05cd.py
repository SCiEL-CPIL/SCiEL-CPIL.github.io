"""
Dense Q3-free graph on 16 vertices.
Strategy: bitmask backtracking with greedy seed and aggressive pruning.
"""
import time
import random
from itertools import combinations, permutations

def construct():
    START = time.time()
    DEADLINE = 115  # safety margin before 120s kill
    
    N = 16
    ALL_PAIRS = list(combinations(range(N), 2))
    PAIR_IDX = {}
    for i, (u, v) in enumerate(ALL_PAIRS):
        PAIR_IDX[(u, v)] = i
    
    TOTAL_PAIRS = len(ALL_PAIRS)  # 120
    
    # Precompute all Q3s in K16 as frozensets of edge indices
    q3_edge_sets = []
    for verts in combinations(range(N), 8):
        vset = set(verts)
        # Build adjacency: in K16, all pairs are connected
        adj_local = {v: set() for v in verts}
        for i, u in enumerate(verts):
            for v in verts[i+1:]:
                adj_local[u].add(v)
                adj_local[v].add(u)
        
        # Q3 is 3-regular
        if any(len(adj_local[v]) != 3 for v in verts):
            continue
        
        v0 = verts[0]
        neighbors0 = list(adj_local[v0])
        if len(neighbors0) != 3:
            continue
        
        found = False
        for perm in permutations(neighbors0):
            label = {v0: (0,0,0), perm[0]: (1,0,0), perm[1]: (0,1,0), perm[2]: (0,0,1)}
            
            common01 = adj_local[perm[0]] & adj_local[perm[1]] - {v0}
            if len(common01) != 1:
                continue
            v110 = list(common01)[0]
            label[v110] = (1,1,0)
            
            common02 = adj_local[perm[0]] & adj_local[perm[2]] - {v0}
            if len(common02) != 1:
                continue
            v101 = list(common02)[0]
            label[v101] = (1,0,1)
            
            common12 = adj_local[perm[1]] & adj_local[perm[2]] - {v0}
            if len(common12) != 1:
                continue
            v011 = list(common12)[0]
            label[v011] = (0,1,1)
            
            remaining = set(verts) - set(label.keys())
            if len(remaining) != 1:
                continue
            v111 = list(remaining)[0]
            label[v111] = (1,1,1)
            
            valid = True
            for u in verts:
                for v in adj_local[u]:
                    if u < v:
                        du = label[u]
                        dv = label[v]
                        dist = sum(1 for a, b in zip(du, dv) if a != b)
                        if dist != 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                found = True
                break
        
        if found:
            edges_in_q3 = []
            for u in verts:
                for v in adj_local[u]:
                    if u < v:
                        idx = PAIR_IDX[(u, v)]
                        edges_in_q3.append(idx)
            q3_edge_sets.append(frozenset(edges_in_q3))
    
    q3_edge_sets = list(set(q3_edge_sets))
    
    # For each edge, which Q3s contain it?
    q3s_by_edge = [[] for _ in range(TOTAL_PAIRS)]
    for qi, q3 in enumerate(q3_edge_sets):
        for e in q3:
            q3s_by_edge[e].append(qi)
    
    def would_create_q3(edge_idx, current_edges_mask):
        for qi in q3s_by_edge[edge_idx]:
            q3 = q3_edge_sets[qi]
            others = q3 - {edge_idx}
            all_present = True
            for e in others:
                if not (current_edges_mask >> e) & 1:
                    all_present = False
                    break
            if all_present:
                return True
        return False
    
    best_edges = []
    best_count = 0
    
    for trial in range(500):
        if time.time() > DEADLINE:
            break
        
        random.seed(trial)
        edge_order = list(range(TOTAL_PAIRS))
        random.shuffle(edge_order)
        
        current_mask = 0
        current_edges = []
        
        for e in edge_order:
            if time.time() > DEADLINE:
                break
            if not would_create_q3(e, current_mask):
                current_mask |= (1 << e)
                current_edges.append(e)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    result = [list(ALL_PAIRS[i]) for i in best_edges]
    return result

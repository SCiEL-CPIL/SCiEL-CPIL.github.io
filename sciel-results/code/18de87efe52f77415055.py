import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edges_for_values(values):
        edges = []
        edges_set = set()
        sums = set()
        prods = set()
        
        edge_sums = {}
        edge_prods = {}
        for e in all_edges:
            u, v = e
            edge_sums[e] = values[u] + values[v]
            edge_prods[e] = values[u] * values[v]
        
        remaining = list(all_edges)
        random.shuffle(remaining)
        
        for _ in range(m):
            if time.time() > deadline - 5.0:
                break
            best_edge = None
            best_sc = float('inf')
            for e in remaining:
                if e in edges_set:
                    continue
                s = edge_sums[e]
                p = edge_prods[e]
                ns = len(sums | {s})
                np_ = len(prods | {p})
                sc = max(ns, np_)
                if sc < best_sc:
                    best_sc = sc
                    best_edge = e
            if best_edge is None:
                best_edge = remaining[0]
            edges.append(best_edge)
            edges_set.add(best_edge)
            sums.add(edge_sums[best_edge])
            prods.add(edge_prods[best_edge])
            remaining.remove(best_edge)
        
        return edges, score_of(values, edges)

    def local_search(values, edges, budget_iters=30000):
        vals = list(values)
        edgs = list(edges)
        edges_set = set(edgs)
        
        cur_score = score_of(vals, edgs)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edgs)
        
        T = 8.0
        T_min = 0.01
        cooling_rate = 0.99997
        
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            
            r = random.random()
            if r < 0.55:
                i = random.randint(0, m - 1)
                e_out = edgs[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edgs = list(edgs)
                new_edgs[i] = e_in
                new_score = score_of(vals, new_edgs)
                
                delta = new_score - cur_score
                if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    edgs = new_edgs
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
            else:
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                
                if random.random() < 0.4:
                    j = random.randint(0, n - 1)
                    if j == idx:
                        continue
                    vals[idx], vals[j] = vals[j], vals[idx]
                    new_score = score_of(vals, edgs)
                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        cur_score = new_score
                    else:
                        vals[idx], vals[j] = vals[j], vals[idx]
                else:
                    candidates = []
                    for dv in [-2, -1, 1, 2, 3, 5]:
                        nv = old_val + dv
                        if 1 <= nv <= 10000 and nv != old_val:
                            candidates.append(nv)
                    candidates.append(random.randint(1, 25))
                    
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    if any(v == new_val for k, v in enumerate(vals) if k != idx):
                        continue
                    
                    vals[idx] = new_val
                    new_score = score_of(vals, edgs)
                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        cur_score = new_score
                    else:
                        vals[idx] = old_val

            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = list(edgs)
            
            T = max(T_min, T * cooling_rate)

        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    candidate_val_sets = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 21],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
    ]
    
    for _ in range(4):
        cv = random.sample(range(1, 35), n)
        candidate_val_sets.append(cv)
    
    for vals in candidate_val_sets:
        if time.time() > deadline - 20.0:
            break
        
        g_edges, g_score = greedy_edges_for_values(vals)
        
        ls_score, ls_vals, ls_edges = local_search(vals, g_edges, budget_iters=25000)
        
        if ls_score < best_score:
            best_score = ls_score
            best_vals = ls_vals
            best_edges = ls_edges
    
    if best_vals is not None and time.time() < deadline - 8.0:
        fs_score, fs_vals, fs_edges = local_search(best_vals, best_edges, budget_iters=40000)
        if fs_score < best_score:
            best_score = fs_score
            best_vals = fs_vals
            best_edges = fs_edges
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}

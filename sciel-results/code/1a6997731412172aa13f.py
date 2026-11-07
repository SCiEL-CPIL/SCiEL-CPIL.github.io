import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def fast_score_with_counts(values, edge_list):
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p)), len(s), len(p)

    def optimize_edges_for_values(values):
        # Precompute sums and products for all edges
        edge_props = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_props.append((u, v, s, p))
            
        best_edges = None
        best_score = float('inf')
        
        # Multiple initializations
        inits = []
        
        # 1. Smallest sums
        sorted_by_sum = sorted(edge_props, key=lambda x: x[2])
        inits.append([e[:2] for e in sorted_by_sum[:m]])
        
        # 2. Smallest products
        sorted_by_prod = sorted(edge_props, key=lambda x: x[3])
        inits.append([e[:2] for e in sorted_by_prod[:m]])
        
        # 3. Random
        inits.append([e[:2] for e in random.sample(edge_props, m)])
        
        # 4. Mix
        inits.append([e[:2] for e in random.sample(edge_props, m)])
        
        for init in inits:
            edges_set = set(init)
            elist = list(init)
            cur_score, cur_sc, cur_pc = fast_score_with_counts(values, elist)
            
            no_improve = 0
            it = 0
            max_iters = 2000 # Reduced to allow more value sets
            while it < max_iters and time.time() < deadline - 5.0:
                it += 1
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_elist = list(elist)
                new_elist[i] = e_in
                new_score, new_sc, new_pc = fast_score_with_counts(values, new_elist)
                
                if new_score < cur_score:
                    elist = new_elist
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score, cur_sc, cur_pc = new_score, new_sc, new_pc
                    no_improve = 0
                elif new_score == cur_score:
                    if random.random() < 0.5:
                        elist = new_elist
                        edges_set.discard(e_out)
                        edges_set.add(e_in)
                        cur_score, cur_sc, cur_pc = new_score, new_sc, new_pc
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    if no_improve > 1000:
                        break
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = list(elist)
                    
        return best_score, best_edges

    best_global_score = float('inf')
    best_global_vals = None
    best_global_edges = None
    
    # Value Pools
    pools = [
        list(range(1, 25)),
        list(range(2, 30)),
        list(range(1, 20)),
        list(range(1, 35)),
        list(range(3, 40)),
    ]
    
    # Add random pools
    for _ in range(10):
        pools.append(random.sample(range(1, 80), 40))
        
    for pool in pools:
        if time.time() > deadline - 15.0:
            break
        for _ in range(10):
            if time.time() > deadline - 15.0:
                break
            if len(pool) < n:
                continue
            vals = random.sample(pool, n)
            sc, edges = optimize_edges_for_values(vals)
            if sc < best_global_score:
                best_global_score = sc
                best_global_vals = vals
                best_global_edges = edges
                
    # Final Polish
    if best_global_vals is not None:
        vals = best_global_vals
        edges = best_global_edges
        edges_set = set(edges)
        elist = list(edges)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(vals, elist)
        
        while time.time() < deadline - 2.0:
            r = random.random()
            if r < 0.7:
                # Edge move
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                new_elist = list(elist)
                new_elist[i] = e_in
                new_score, new_sc, new_pc = fast_score_with_counts(vals, new_elist)
                if new_score < cur_score:
                    elist = new_elist
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score, cur_sc, cur_pc = new_score, new_sc, new_pc
            else:
                # Value move
                idx = random.randint(0, n - 1)
                new_val = vals[idx] + random.randint(-10, 10)
                if not (1 <= new_val <= 10000) or new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                new_score, new_sc, new_pc = fast_score_with_counts(new_vals, elist)
                if new_score < cur_score:
                    vals = new_vals
                    cur_score, cur_sc, cur_pc = new_score, new_sc, new_pc
            
            if cur_score < best_global_score:
                best_global_score = cur_score
                best_global_vals = vals
                best_global_edges = list(elist)
                
    if best_global_vals is None:
        best_global_vals = list(range(1, 15))
        best_global_edges = [list(e) for e in all_edges[:m]]
        
    return {
        'values': [int(v) for v in best_global_vals],
        'edges': [[int(u), int(v)] for u, v in best_global_edges]
    }

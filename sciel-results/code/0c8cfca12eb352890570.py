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

    def simulated_annealing(init_vals, init_edges, budget_iters=200000):
        vals = list(init_vals)
        edges = list(init_edges)
        edges_set = set(edges)
        
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        it = 0
        while it < budget_iters and time.time() < deadline - 2.0:
            it += 1
            
            if random.random() < 0.6:
                # Edge Swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                delta = new_score - cur_score
                if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
            else:
                # Value Swap or Perturb
                if random.random() < 0.5:
                    # Swap two values
                    i, j = random.sample(range(n), 2)
                    vals[i], vals[j] = vals[j], vals[i]
                    new_score = score_of(vals, edges)
                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        cur_score = new_score
                    else:
                        vals[i], vals[j] = vals[j], vals[i]
                else:
                    # Perturb one value
                    idx = random.randint(0, n - 1)
                    old_val = vals[idx]
                    # Sample from a distribution that favors small numbers
                    new_val = random.randint(1, 50)
                    if new_val == old_val:
                        continue
                    if any(v == new_val for k, v in enumerate(vals) if k != idx):
                        continue
                    
                    vals[idx] = new_val
                    new_score = score_of(vals, edges)
                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        cur_score = new_score
                    else:
                        vals[idx] = old_val

            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = list(edges)
            
            T = max(T_min, T * cooling_rate)

        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Candidate 1: 1..14
    vals1 = list(range(1, 15))
    edges1 = random.sample(all_edges, m)
    sc1, v1, e1 = simulated_annealing(vals1, edges1, budget_iters=50000)
    if sc1 < best_score:
        best_score, best_vals, best_edges = sc1, v1, e1

    # Candidate 2: Divisors of 72
    vals2 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72]
    edges2 = random.sample(all_edges, m)
    sc2, v2, e2 = simulated_annealing(vals2, edges2, budget_iters=50000)
    if sc2 < best_score:
        best_score, best_vals, best_edges = sc2, v2, e2

    # Candidate 3: Random small values
    vals3 = random.sample(range(1, 30), n)
    edges3 = random.sample(all_edges, m)
    sc3, v3, e3 = simulated_annealing(vals3, edges3, budget_iters=50000)
    if sc3 < best_score:
        best_score, best_vals, best_edges = sc3, v3, e3

    # Final polish with more iterations on the best found
    if time.time() < deadline - 10.0:
        sc4, v4, e4 = simulated_annealing(best_vals, best_edges, budget_iters=100000)
        if sc4 < best_score:
            best_score, best_vals, best_edges = sc4, v4, e4

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}

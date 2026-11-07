import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 90s limit, leave buffer for overhead
    
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

    def optimize_edges(values, initial_edges=None, budget_iters=30000):
        """
        Find 35 edges for fixed values minimizing score.
        Uses greedy init then local search with 2-edge swaps.
        """
        if initial_edges:
            edges = list(initial_edges)
        else:
            # Greedy: pick edges that minimize increase in distinct sums/prods
            edges = []
            edge_set = set()
            sums = set()
            prods = set()
            
            # Sort edges by sum then product to encourage collisions
            sorted_edges = sorted(all_edges, key=lambda e: (values[e[0]]+values[e[1]], values[e[0]]*values[e[1]]))
            
            for e in sorted_edges:
                if len(edges) == m:
                    break
                edges.append(e)
                edge_set.add(e)
                sums.add(values[e[0]] + values[e[1]])
                prods.add(values[e[0]] * values[e[1]])
        
        edges_set = set(edges)
        cur_score = score_of(values, edges)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            move_type = random.random()
            
            if move_type < 0.6:
                # Single edge swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(values, new_edges)
                
                if new_score <= cur_score:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Two-edge swap
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 == i2:
                    continue
                e_out1 = edges[i1]
                e_out2 = edges[i2]
                e_in1 = random.choice(all_edges)
                e_in2 = random.choice(all_edges)
                if e_in1 in edges_set or e_in2 in edges_set or e_in1 == e_in2:
                    continue
                
                new_edges = list(edges)
                new_edges[i1] = e_in1
                new_edges[i2] = e_in2
                new_score = score_of(values, new_edges)
                
                if new_score <= cur_score:
                    edges = new_edges
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    cur_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > 2000 and random.random() < 0.02:
                no_improve = 0

        return edges, cur_score

    def optimize_values(values, edges, budget_iters=30000):
        """
        Optimize values for fixed edges.
        """
        vals = list(values)
        cur_score = score_of(vals, edges)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Try various perturbation strategies
            r = random.random()
            if r < 0.3:
                new_val = old_val + random.randint(-10, 10)
            elif r < 0.5:
                new_val = old_val * random.choice([2, 3])
            elif r < 0.7:
                new_val = old_val + random.randint(-50, 50)
            else:
                new_val = random.randint(1, 200)
            
            if new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
            
            new_vals = list(vals)
            new_vals[idx] = new_val
            new_score = score_of(new_vals, edges)
            
            if new_score <= cur_score:
                vals = new_vals
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 3000 and random.random() < 0.01:
                no_improve = 0

        return vals, cur_score

    def joint_search(init_vals, init_edges, budget_iters=50000):
        """
        Joint local search on values and edges.
        """
        vals = list(init_vals)
        edges = list(init_edges)
        edges_set = set(edges)
        
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        no_improve = 0
        it = 0
        
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            
            if random.random() < 0.6:
                # Edge move
                i = random.randint(0, m - 1)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                if new_score <= cur_score:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Value move
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                r = random.random()
                if r < 0.4:
                    new_val = old_val + random.randint(-8, 8)
                elif r < 0.6:
                    new_val = random.randint(1, 100)
                elif r < 0.8:
                    new_val = old_val + random.randint(-30, 30)
                else:
                    new_val = random.randint(1, 500)
                
                if new_val <= 0 or new_val > 10000:
                    continue
                if any(v == new_val for j, v in enumerate(vals) if j != idx):
                    continue
                
                new_vals = list(vals)
                new_vals[idx] = new_val
                new_score = score_of(new_vals, edges)
                
                if new_score <= cur_score:
                    vals = new_vals
                    cur_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
            
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = list(edges)
            
            if no_improve > 5000 and random.random() < 0.005:
                no_improve = 0

        return best_score, best_vals, best_edges

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Generate diverse candidate value sets
    candidates = []
    
    # 1. Small consecutive
    candidates.append(list(range(1, 15)))
    
    # 2. Powers of 2
    candidates.append([1 << i for i in range(14)])
    
    # 3. Arithmetic progression with small difference
    candidates.append([2 + i for i in range(14)])
    
    # 4. Clustered small values
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 5. Divisors of 60
    candidates.append([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 7, 11])
    
    # 6. Random small values
    random.seed(123)
    for _ in range(5):
        rv = random.sample(range(1, 30), n)
        candidates.append(rv)
    
    # 7. Geometric progression
    candidates.append([int(2 ** (i * 0.5)) for i in range(14)])
    # Make distinct
    seen = set()
    unique_cand = []
    for v in candidates[-1]:
        if v not in seen and 1 <= v <= 10000:
            seen.add(v)
            unique_cand.append(v)
    while len(unique_cand) < n:
        unique_cand.append(len(unique_cand) + 100)
    candidates[-1] = unique_cand[:n]
    
    # 8. Fibonacci-like
    fib = [1, 2]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    candidates.append(fib[:n])
    
    # For each candidate, run edge optimization then joint search
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        
        # First optimize edges for this value set
        edges_c, score_c = optimize_edges(cand, budget_iters=10000)
        
        if score_c < best_score:
            best_score = score_c
            best_vals = list(cand)
            best_edges = edges_c
        
        # Then joint search from this starting point
        if time.time() > deadline - 15.0:
            break
        
        sc, vals, edges = joint_search(cand, edges_c, budget_iters=20000)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges
    
    # Final polish: edge optimization on best values
    if time.time() < deadline - 3.0 and best_vals is not None:
        edges_final, score_final = optimize_edges(best_vals, best_edges, budget_iters=15000)
        if score_final < best_score:
            best_score = score_final
            best_edges = edges_final
    
    # Value optimization on best edges
    if time.time() < deadline - 3.0 and best_vals is not None:
        vals_final, score_final = optimize_values(best_vals, best_edges, budget_iters=15000)
        if score_final < best_score:
            best_score = score_final
            best_vals = vals_final
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    # Validate and convert
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Ensure validity
    assert len(final_values) == 14
    assert len(set(final_values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == 35
    edge_set_check = set()
    for e in final_edges:
        assert isinstance(e, list) and len(e) == 2
        assert isinstance(e[0], int) and isinstance(e[1], int)
        assert 0 <= e[0] < 14 and 0 <= e[1] < 14
        assert e[0] != e[1]
        key = tuple(sorted(e))
        assert key not in edge_set_check
        edge_set_check.add(key)
    
    return {'values': final_values, 'edges': final_edges}

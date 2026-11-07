import random
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    # Generate candidate graphs
    graphs = []
    from itertools import combinations
    seed_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    graphs.append(seed_edges)
    
    possible_bip_edges = [(i, j) for i in range(7) for j in range(7, 14)]
    for _ in range(30):
        random.shuffle(possible_bip_edges)
        graphs.append([list(e) for e in possible_bip_edges[:m]])
        
    # Local search function
    def local_search(edges, init_vals, max_steps, T_start):
        current_vals = list(init_vals)
        current_score = evaluate(current_vals, edges)
        best_local_score = current_score
        best_local_vals = list(current_vals)
        
        T = T_start
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            u = random.randint(0, n-1)
            old_val = current_vals[u]
            
            # Generate new value
            # Mix of small and random
            if random.random() < 0.7:
                new_val = random.randint(1, 60)
            else:
                new_val = random.randint(1, 10000)
                
            if new_val == old_val:
                continue
                
            # Check distinctness
            # Optimization: check if new_val is in current_vals excluding u
            # Using a set for current_vals is faster for lookup but needs update
            # n=14 is small, linear scan is fine and avoids set maintenance overhead
            is_distinct = True
            for k in range(n):
                if k != u and current_vals[k] == new_val:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue

            current_vals[u] = new_val
            new_score = evaluate(current_vals, edges)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_vals = list(current_vals)
                T = max(T * 0.95, 0.1)
            else:
                delta = new_score - current_score
                if delta > 0:
                    p = pow(2.718, -delta / T)
                    if random.random() < p:
                        current_score = new_score
                    else:
                        current_vals[u] = old_val
                else:
                    current_score = new_score
            T = max(T * 0.999, 0.1)
            
        return best_local_score, best_local_vals

    best_global_score = float('inf')
    best_global_result = None

    # Initial value sets to try
    init_sets = [
        list(range(1, n+1)),
        random.sample(range(1, 101), n),
        random.sample(range(1, 201), n),
        random.sample(range(1, 1001), n)
    ]
    
    # Shuffle graphs to diversify
    random.shuffle(graphs)
    
    for g_edges in graphs:
        if time.time() > deadline:
            break
        
        # Try different initializations for this graph
        # Limit steps per init to allow multiple inits
        steps_per_init = 1000
        for init_vals in init_sets:
            if time.time() > deadline:
                break
            score, vals = local_search(g_edges, init_vals, steps_per_init, 10.0)
            if score < best_global_score:
                best_global_score = score
                best_global_result = {'values': vals, 'edges': [list(e) for e in g_edges]}
                # If we get very low, we can stop early or tighten search
                if best_global_score <= 18:
                    # Refine with more steps
                    score2, vals2 = local_search(g_edges, vals, 5000, 5.0)
                    if score2 < best_global_score:
                        best_global_score = score2
                        best_global_result = {'values': vals2, 'edges': [list(e) for e in g_edges]}

    if best_global_result is None:
        best_global_result = {'values': list(range(1, n+1)), 'edges': seed_edges}
        
    best_global_result['values'] = [int(v) for v in best_global_result['values']]
    best_global_result['edges'] = [[int(i), int(j)] for i, j in best_global_result['edges']]
    
    return best_global_result

# EVOLVE-BLOCK-END

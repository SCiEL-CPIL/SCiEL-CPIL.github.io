import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to list of lists for faster iteration
    adj_lists = [list(s) for s in adj]
    
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0: break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    def simulated_annealing(initial_set, max_steps, T_start, cooling):
        S = initial_set.copy()
        current_size = len(S)
        best_local = S.copy()
        best_local_size = current_size
        T = T_start
        
        for step in range(max_steps):
            v = random.randint(0, n_sup - 1)
            
            if v in S:
                continue
            
            N = adj[v].intersection(S)
            
            if len(N) == 0:
                S.add(v)
                current_size += 1
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local = S.copy()
            else:
                delta = 1 - len(N)
                
                if delta > 0:
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = S.copy()
                else:
                    if delta == 0:
                        if random.random() < 0.5:
                            S.difference_update(N)
                            S.add(v)
                    else:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S.difference_update(N)
                            S.add(v)
                            current_size += delta
            
            T *= cooling
        
        return best_local, best_local_size

    def local_search(S, max_iter=1000):
        """Try to improve by adding vertices that can be added without conflict"""
        improved = True
        while improved:
            improved = False
            for v in range(n_sup):
                if v not in S:
                    if not adj[v].intersection(S):
                        S.add(v)
                        improved = True
        return S

    # Multiple restarts with different annealing parameters
    start_time = time.time()
    best_set = set()
    best_size = 0
    
    # Run multiple SA instances
    num_restarts = 20
    steps_per_run = 20000
    
    for restart in range(num_restarts):
        if time.time() - start_time > 110:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            init = greedy_is()
        elif restart % 3 == 1:
            # Random greedy
            remaining = set(range(n_sup))
            init = set()
            while remaining:
                v = random.choice(list(remaining))
                init.add(v)
                to_remove = {v}.union(adj[v].intersection(remaining))
                remaining.difference_update(to_remove)
        else:
            # Start with small random set
            init = set(random.sample(range(n_sup), min(10, n_sup)))
        
        # Local search to make maximal
        init = local_search(init)
        
        # Simulated annealing with varying parameters
        T_start = 0.5 + 0.5 * (restart / num_restarts)
        cooling = 0.9995
        
        result_set, result_size = simulated_annealing(init, steps_per_run, T_start, cooling)
        
        # Final local search
        result_set = local_search(result_set)
        result_size = len(result_set)
        
        if result_size > best_size:
            best_size = result_size
            best_set = result_set
    
    # Construct vectors from best_set
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

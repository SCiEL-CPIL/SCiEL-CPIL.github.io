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
                
    # QUBO Formulation:
    # Minimize E = - sum(x_i) + lambda * sum_{(i,j) in E} x_i x_j
    # lambda = 1.5 ensures constraints are respected (penalty > gain)
    LAMBDA = 1.5
    
    # Precompute diagonal and adjacency for fast energy calc
    # Q_ii = -1
    # Q_ij = LAMBDA if edge else 0
    
    # 1. Greedy Initialization
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            # Pick vertex with min degree in remaining
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

    best_set = greedy_is()
    x = np.zeros(n_sup, dtype=np.int8)
    for idx in best_set:
        x[idx] = 1
        
    # Calculate initial energy
    # E = -sum(x) + LAMBDA * sum_{edges} x_i x_j
    current_energy = -np.sum(x)
    for i in range(n_sup):
        if x[i] == 1:
            for j in adj[i]:
                if x[j] == 1:
                    current_energy += LAMBDA
    # Note: The loop above counts each edge twice if we are not careful?
    # adj[i] contains j. If i<j, j is in adj[i].
    # In the loop `for i in range(n_sup)`, if x[i]=1, we check all j in adj[i].
    # If x[j]=1, we add LAMBDA.
    # Since graph is undirected, edge (i,j) is counted when i is processed (if x[i]=1, x[j]=1) 
    # and when j is processed (if x[j]=1, x[i]=1).
    # So we added 2 * LAMBDA for each edge.
    # We want sum_{(i,j) in E} x_i x_j. This is the number of edges with both ends selected.
    # So we should divide by 2 or only count i < j.
    # Let's recalculate cleanly to be safe.
    current_energy = -np.sum(x)
    edge_count = 0
    for i in range(n_sup):
        if x[i] == 1:
            for j in adj[i]:
                if j > i and x[j] == 1:
                    edge_count += 1
    current_energy += LAMBDA * edge_count
    
    best_energy = current_energy
    best_x = x.copy()
    
    # 2. Simulated Quantum Annealing (SQA)
    T = 1.0
    Gamma = 5.0
    T_end = 0.01
    Gamma_end = 0.01
    steps = 100000
    start_time = time.time()
    time_limit = 110.0
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        k = random.randint(0, n_sup - 1)
        
        # Calculate local field h_k = sum_{j != k} x_j Q_kj
        # Q_kj = LAMBDA if j in adj[k] else 0
        neighbor_sum = 0
        for j in adj[k]:
            neighbor_sum += x[j]
            
        # Delta E = (1 - 2*x_k) * (Q_kk + h_k)
        # Q_kk = -1
        local_term = -1.0 + LAMBDA * neighbor_sum
        delta_e = (1 - 2 * x[k]) * local_term
        
        # Acceptance probability
        # Effective temperature for SQA
        effective_T = T + Gamma
        if delta_e < 0:
            accept = True
        else:
            if effective_T < 1e-6:
                accept = False
            else:
                # P = 1 / (1 + exp(delta_e / effective_T))
                # Avoid overflow
                ratio = delta_e / effective_T
                if ratio > 50:
                    p = 0.0
                else:
                    p = 1.0 / (1.0 + np.exp(ratio))
                accept = (random.random() < p)
                
        if accept:
            x[k] = 1 - x[k]
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                
        # Anneal
        T *= 0.999
        Gamma *= 0.999
        
    # Construct vectors from best_x
    vectors = []
    for idx in range(n_sup):
        if best_x[idx] == 1:
            sup = supports[idx]
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = 1
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)

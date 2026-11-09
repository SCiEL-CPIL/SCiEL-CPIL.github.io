import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 120.0
    margin = 5.0
    
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        # 1. Generate Graph
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques (Bron-Kerbosch with pivot)
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            if not P:
                return
            
            # Pivot selection: pick u in P U X maximizing |P \cap N(u)|
            # For small n, simple heuristic is fine
            max_deg = -1
            pivot = None
            # Check P first, then X
            for u in P:
                deg = len(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            for u in X:
                deg = len(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot is None:
                candidates = list(P)
            else:
                candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bron_kerbosch(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
                
        P_initial = frozenset(range(n))
        bron_kerbosch(frozenset(), P_initial, frozenset())
        
        cliques = [list(c) for c in maximal_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Precompute vertex to clique indices
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        num_cliques = len(cliques)
        
        # 4. Greedy Initialization
        S = set()
        cover_count = [0] * num_cliques
        uncovered = num_cliques
        
        while uncovered > 0:
            best_v = -1
            best_gain = -1
            for v in range(n):
                if v in S:
                    continue
                gain = 0
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v == -1:
                break
                
            S.add(best_v)
            for c_idx in vertex_to_cliques[best_v]:
                if cover_count[c_idx] == 0:
                    uncovered -= 1
                cover_count[c_idx] += 1
        
        # 5. Simulated Annealing
        best_S = set(S)
        best_size = len(S)
        
        current_time = time.time()
        if time_limit - (current_time - start_time) - margin <= 0:
            results.append(sorted(best_S))
            continue
            
        rng_sa = random.Random(seed + 1000)
        T = 1.0
        T_min = 0.001
        M = 100 # Penalty for invalid solution
        
        # Helper to calculate delta_uncovered without modifying state
        def calc_delta_uncov(v, is_add):
            delta = 0
            if is_add:
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 0:
                        delta -= 1
            else:
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 1:
                        delta += 1
            return delta

        while time.time() - start_time < time_limit - margin:
            v = rng_sa.randint(0, n - 1)
            
            if v in S:
                delta_uncov = calc_delta_uncov(v, False)
                if delta_uncov == 0:
                    # Valid removal
                    S.remove(v)
                    for c_idx in vertex_to_cliques[v]:
                        cover_count[c_idx] -= 1
                    # uncovered unchanged
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    delta_E = -1 + M * delta_uncov
                    if delta_E < 0 or rng_sa.random() < pow(2.718281828, -delta_E / T):
                        S.remove(v)
                        for c_idx in vertex_to_cliques[v]:
                            cover_count[c_idx] -= 1
                        uncovered += delta_uncov
            else:
                delta_uncov = calc_delta_uncov(v, True)
                delta_E = 1 + M * delta_uncov
                if delta_E < 0 or rng_sa.random() < pow(2.718281828, -delta_E / T):
                    S.add(v)
                    for c_idx in vertex_to_cliques[v]:
                        if cover_count[c_idx] == 0:
                            uncovered -= 1
                        cover_count[c_idx] += 1
                    if uncovered == 0 and len(S) < best_size:
                        best_size = len(S)
                        best_S = set(S)
                        
            T *= 0.999
            if T < T_min:
                T = T_min
                
        results.append(sorted(best_S))
        
    return results

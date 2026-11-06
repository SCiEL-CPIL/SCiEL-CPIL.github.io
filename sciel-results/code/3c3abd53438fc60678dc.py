import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for overhead
    
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return empty lists or best effort? 
            # The task says "Minimize total size". Empty is invalid if cliques exist.
            # We must return a valid transversal.
            # If we run out of time, we should return a valid greedy solution.
            pass
            
        # 1. Generate Graph
        # Use random.Random for reproducibility
        import random
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques
        # Bron-Kerbosch with pivot
        cliques = []
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(tuple(sorted(R)))
                return
            # Pivot selection
            if P:
                # Choose pivot u in P U X with max degree in P
                # For simplicity and speed on small n, just pick first or random?
                # Standard: max |N(u) \cap P|
                # Let's just iterate. For n=40, this is fast.
                pivot_candidates = list(P | X)
                if not pivot_candidates: return
                # Pick u with max neighbors in P
                u = max(pivot_candidates, key=lambda x: len(adj[x] & P))
                # Iterate over P \ N(u)
                candidates = P - adj[u]
            else:
                candidates = P
                
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), set(range(n)), set())
        
        # Filter cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            results.append([])
            continue
            
        # 3. Solve ILP to find max Y such that no clique C is subset of Y
        # Maximize sum(y_i)
        # Constraints: sum_{v in C} y_v <= |C| - 1 for all C
        # y_i in {0, 1}
        
        # If time is running out, use greedy
        if time.time() > deadline - 5:
             # Greedy Set Cover
             S = set()
             uncovered = [set(c) for c in relevant_cliques]
             # Simple greedy: pick vertex in most uncovered cliques
             # We need to cover all cliques.
             # Actually, S must hit every clique.
             # Greedy:
             remaining_cliques = [set(c) for c in relevant_cliques]
             while remaining_cliques:
                 # Find vertex that appears in most remaining cliques
                 counts = {}
                 for c in remaining_cliques:
                     for v in c:
                         counts[v] = counts.get(v, 0) + 1
                 if not counts: break
                 v = max(counts, key=counts.get)
                 S.add(v)
                 # Remove all cliques containing v
                 remaining_cliques = [c for c in remaining_cliques if v not in c]
             results.append(sorted(S))
             continue
             
        # Build ILP
        # Variables: y_0 ... y_{n-1}
        # Objective: c = -1 (minimize -sum(y) => maximize sum(y))
        # Actually milp minimizes c.T @ x. So c = -1.
        
        A_ub = np.zeros((len(relevant_cliques), n))
        b_ub = np.zeros(len(relevant_cliques))
        
        for i, c in enumerate(relevant_cliques):
            for v in c:
                A_ub[i, v] = 1
            b_ub[i] = len(c) - 1
            
        # Bounds: 0 <= y_i <= 1
        bounds = Bounds(0, 1)
        integrality = np.ones(n) # All integer
        
        res = milp(
            c=-np.ones(n),
            constraints=LinearConstraint(A_ub, None, b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            y = res.x
            # y_i is 1 if vertex i is in Y (kept)
            # S is vertices NOT in Y
            S = [i for i in range(n) if y[i] < 0.5]
            results.append(sorted(S))
        else:
            # Fallback to greedy if ILP fails
            S = set()
            remaining_cliques = [set(c) for c in relevant_cliques]
            while remaining_cliques:
                counts = {}
                for c in remaining_cliques:
                    for v in c:
                        counts[v] = counts.get(v, 0) + 1
                if not counts: break
                v = max(counts, key=counts.get)
                S.add(v)
                remaining_cliques = [c for c in remaining_cliques if v not in c]
            results.append(sorted(S))
            
    return results

import random
import numpy as np
from scipy.optimize import linprog
from collections import defaultdict

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        pivot = None
        max_deg = -1
        for v in P.union(X):
            deg = len(adj[v].intersection(P))
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
            
        for v in list(P):
            if v not in adj[pivot]:
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        valid_cliques = [c for c in cliques if len(c) >= 2]
        if not valid_cliques:
            return []
            
        M = len(valid_cliques)
        # A[v][c] = 1 if v is in clique c
        # We want min sum x_v s.t. sum_{v in c} x_v >= 1 for all c
        # x_v in {0,1}
        
        # LP Relaxation
        # Minimize c^T x
        # Subject to A^T x >= 1
        # 0 <= x <= 1
        
        c = np.ones(n)
        
        # A matrix: M x n
        A = np.zeros((M, n))
        for i, clique in enumerate(valid_cliques):
            for v in clique:
                A[i, v] = 1
                
        # Inequality constraints: A^T x >= 1  =>  -A^T x <= -1
        A_ub = -A.T
        b_ub = -np.ones(M)
        
        bounds = [(0, 1)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            # Fallback to greedy
            return greedy_fallback(n, adj, valid_cliques)
            
        x = res.x
        
        # Rounding Strategy:
        # 1. Sort vertices by x_v descending.
        # 2. Greedily add vertices with x_v > 0.5? Or just top k?
        # Better: Greedy on the LP solution.
        # Start with empty set.
        # While uncovered cliques remain:
        #   Pick vertex v with max x_v among those covering uncovered cliques?
        #   Or simply pick all v with x_v > 0.
        
        # Let's try a simple rounding:
        # S = {v | x_v > 0.5}
        # If not valid, add vertices greedily to cover remaining.
        
        S = [v for v in range(n) if x[v] > 0.5]
        
        # Check validity
        def is_valid(S):
            S_set = set(S)
            for c in valid_cliques:
                if not any(v in S_set for v in c):
                    return False
            return True
            
        if not is_valid(S):
            # Greedy completion
            uncovered = set(range(M))
            for i, c in enumerate(valid_cliques):
                if not any(v in S for v in c):
                    uncovered.add(i)
            
            while uncovered:
                best_v = -1
                best_gain = -1
                for v in range(n):
                    if v in S:
                        continue
                    gain = 0
                    for i in uncovered:
                        if v in valid_cliques[i]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1:
                    break
                S.append(best_v)
                # Update uncovered
                new_uncovered = set()
                for i in uncovered:
                    if best_v in valid_cliques[i]:
                        pass # covered
                    else:
                        new_uncovered.add(i)
                uncovered = new_uncovered
                
        # Local Search to minimize size
        # Try to remove vertices
        changed = True
        while changed:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                v = S[i]
                # Check if v is essential
                essential = False
                for c in valid_cliques:
                    if v in c:
                        # Is there another vertex in S in c?
                        if not any(u in S for u in c if u != v):
                            essential = True
                            break
                if not essential:
                    S.pop(i)
                    changed = True
        
        return sorted(S)
        
    def greedy_fallback(n, adj, valid_cliques):
        # Same as original but simplified
        M = len(valid_cliques)
        masks = [0] * n
        for i, clique in enumerate(valid_cliques):
            bit = 1 << i
            for v in clique:
                masks[v] |= bit
                
        all_cliques_mask = (1 << M) - 1
        uncovered = all_cliques_mask
        selected = []
        
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if not (uncovered & masks[v]):
                    continue
                count = (uncovered & masks[v]).bit_count()
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            uncovered &= ~masks[best_v]
            
        return sorted(selected)
        
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results

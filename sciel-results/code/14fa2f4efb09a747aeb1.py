import random
from collections import defaultdict
import numpy as np

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        if P:
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)
        else:
             pass

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def greedy_transversal(cliques):
        if not cliques:
            return []
        
        remaining = list(cliques)
        T = set()
        
        while remaining:
            counts = defaultdict(int)
            for c in remaining:
                for v in c:
                    counts[v] += 1
            
            best_v = max(counts, key=counts.get)
            T.add(best_v)
            
            remaining = [c for c in remaining if best_v not in c]
        
        final_T = set(T)
        for v in list(final_T):
            candidate_T = final_T - {v}
            if all(any(x in candidate_T for x in c) for c in cliques):
                final_T = candidate_T
        
        return sorted(final_T)

    def solve_with_ilp(n, cliques):
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import coo_matrix
            
            if not cliques:
                return []
            
            c = np.ones(n)
            rows = []
            cols = []
            vals = []
            
            for i, clique in enumerate(cliques):
                for v in clique:
                    rows.append(i)
                    cols.append(v)
                    vals.append(1)
            
            if not rows:
                return []
                
            A = coo_matrix((vals, (rows, cols)), shape=(len(cliques), n)).to_csr()
            constraints = LinearConstraint(A, 1, np.inf)
            integrality = np.ones(n)
            bounds = Bounds(0, 1)
            
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                x = np.round(res.x).astype(int)
                return [i for i in range(n) if x[i] == 1]
            else:
                return greedy_transversal(cliques)
        except Exception:
            return greedy_transversal(cliques)

    def local_search_improve(n, cliques, initial_T):
        T = set(initial_T)
        if not cliques:
            return []
        
        # 1. Remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(T):
                if all(any(x in T for x in c) for c in cliques if v not in c or any(x in T for x in c if x != v)):
                    # Check if T - {v} is valid
                    if all(any(x in T - {v} for x in c) for c in cliques):
                        T.remove(v)
                        changed = True
                        break
        
        # 2. Try to swap: Replace 2 vertices with 1
        # This is heuristic and might be slow if not careful, but n is small.
        # We only do this if T is not empty.
        # To avoid infinite loops or excessive time, we limit iterations.
        for _ in range(100): # Max 100 improvement steps
            improved = False
            # Try to find u not in T that allows removal of v1, v2 in T
            # This is O(n * |T|^2 * |cliques|) which is 40 * 1600 * 100 ~ 6.4M ops. Acceptable.
            candidates_not_in_T = [v for v in range(n) if v not in T]
            for u in candidates_not_in_T:
                # If we add u, which vertices in T become redundant?
                # A vertex v in T is redundant if all cliques it hits are also hit by (T - {v}) or by u.
                # Actually, we want to remove a set S of vertices from T and add u.
                # Let's try removing pairs {v1, v2}.
                T_list = list(T)
                for i in range(len(T_list)):
                    for j in range(i+1, len(T_list)):
                        v1, v2 = T_list[i], T_list[j]
                        new_T = T - {v1, v2} | {u}
                        if all(any(x in new_T for x in c) for c in cliques):
                            T = new_T
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try removing 3 vertices and adding 1? Or just stop.
                # For now, stop if no pair swap works.
                break
                
        return sorted(T)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        # Try ILP
        transversal = solve_with_ilp(n, relevant_cliques)
        
        # Verify and improve with local search
        if not all(any(v in transversal for v in c) for c in relevant_cliques):
            transversal = greedy_transversal(relevant_cliques)
            
        transversal = local_search_improve(n, relevant_cliques, transversal)
        results.append(transversal)
        
    return results

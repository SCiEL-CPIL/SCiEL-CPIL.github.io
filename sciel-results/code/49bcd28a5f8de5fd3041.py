"""Clique partition of split graphs using ILP for optimal neighborhood assignment."""
import time
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # 1. Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # C vertices: 0..k-1
        # I vertices: k..n-1
        
        # Get cross neighbors for each I vertex
        # N[i] is the set of C vertices connected to I vertex i
        N = [set() for _ in range(n)]
        for i in range(k, n):
            N[i] = {c for c in range(k) if c in adj[i]}
        
        # 2. Generate candidate subsets for each I vertex
        # A candidate subset S must be a subset of N[i] and must be a clique in the C-part
        # (Since C is a clique, any subset is a clique. Wait, C is the first k vertices, which form a clique.)
        # The constraint is that S must be a subset of N[i].
        # Also, we want to cover edges efficiently.
        # The "big clique" strategy suggests we want large S.
        # However, we must ensure that the union of all chosen S_i (plus edge cliques) covers all edges exactly once.
        
        # Actually, the "exactly once" constraint is the hard part.
        # If we just pick S_i, an edge (u,v) in C is covered by:
        # 1. The big clique {0, ..., k-1} ? No, that would cover all C-edges.
        # If we include the big clique, we cannot use any other clique that contains a C-edge.
        # So if we use the big clique, all S_i must be empty? No, S_i are subsets of N[i].
        # If S_i is non-empty, the clique {i} U S_i contains edges (i, c) and edges (c1, c2) for c1,c2 in S_i.
        # The edges (c1, c2) are C-edges.
        # If the big clique covers ALL C-edges, then no other clique can cover any C-edge.
        # Thus, if we use the big clique, S_i must be such that it does NOT contain any C-edge.
        # Since S_i is a subset of vertices, it contains C-edges if |S_i| >= 2.
        # So if we use the big clique, all |S_i| <= 1.
        # This reduces to: 1 big clique + sum(|S_i|) cliques of size 2 (if |S_i|=1) or 0 (if |S_i|=0).
        # But wait, if |S_i|=1, the clique is {i, c}. This covers the cross edge (i,c).
        # The C-edge (c, c') is covered by the big clique.
        # This is the "1 + E_cross" strategy mentioned in the knowledge.
        
        # Is there a better strategy?
        # If we don't use the big clique, we must cover C-edges using the S_i's and explicit C-edge cliques.
        # An edge (u,v) in C is covered by:
        # - Any S_i containing both u and v.
        # - An explicit clique {u, v} if not covered by any S_i.
        # To minimize count, we want to maximize the number of C-edges covered by S_i's.
        # But each S_i incurs a cost of 1 clique.
        # If S_i has size m, it covers m(m-1)/2 C-edges and m cross edges.
        # If we didn't use S_i, we would use m cliques of size 2 for cross edges (cost m)
        # and the C-edges would need to be covered by other S_j's or explicit cliques.
        
        # Let's formulate as an assignment problem.
        # For each i in I, we choose a subset S_i of N[i].
        # Cost: 1 if S_i is not empty, 0 otherwise.
        # Constraint: For each C-edge e=(u,v), the number of i such that {u,v} subset of S_i must be >= 1.
        # Wait, "exactly one listed clique" contains every edge.
        # If two S_i's both contain {u,v}, then edge (u,v) is in two cliques. Invalid.
        # So for each C-edge e, the number of i such that {u,v} subset of S_i must be EXACTLY 1.
        
        # This is a strict constraint.
        # Let x_{i, S} be 1 if we choose subset S for vertex i.
        # Sum_S x_{i, S} = 1 for all i.
        # For each C-edge e: Sum_i Sum_{S superset e} x_{i, S} = 1.
        # Minimize Sum_i Sum_{S non-empty} x_{i, S}.
        
        # The number of possible subsets S for each i is 2^{|N[i]|}.
        # |N[i]| is around p*k. For k=30, p=0.7, |N[i]| ~ 21. 2^21 is 2 million. Too many.
        
        # We need to reduce the candidate set.
        # Observation: If S is a candidate, any subset of S is also a candidate (structurally).
        # But we only care about subsets that are "maximal" in some sense or specific structures.
        # Actually, if we choose S, we cover all pairs in S.
        # If we choose S1 and S2 where S1 subset S2, choosing S2 is always at least as good for covering C-edges,
        # but it might cause conflicts (covering an edge twice).
        # However, since the constraint is EXACTLY 1, we can't just pick maximal sets.
        
        # Alternative approach:
        # Since k is small (10, 20, 30), the number of C-edges is small (45, 190, 435).
        # We can model this as an ILP over the C-edges?
        
        # Let's try a different perspective.
        # The "1 + E_cross" strategy gives a score of 1.336.
        # The current score is 1.193.
        # The ILP approach aims to beat 1.193.
        
        # Let's restrict candidates for each i to:
        # 1. Empty set
        # 2. Singletons {c} for c in N[i]
        # 3. Pairs {c1, c2} for c1, c2 in N[i]
        # 4. Triples? Maybe.
        # If we limit to size <= 2, the number of candidates is 1 + |N[i]| + |N[i]|^2/2.
        # For |N[i]|=21, that's 1 + 21 + 210 = 232 candidates per i.
        # Total variables: n_i * 232. For n=90, k=30, n_i=60. 60 * 232 = 13920 variables.
        # Constraints: 1 per i (60) + 1 per C-edge (435). Total ~500 constraints.
        # This is a very small ILP!
        
        # Does limiting to size <= 2 lose optimality?
        # A clique of size 3 in S_i covers 3 C-edges.
        # Three singletons cover 0 C-edges (just cross edges).
        # One pair covers 1 C-edge.
        # If we use a triple, we save 2 C-edge cliques (compared to covering them explicitly) 
        # but we use 1 clique instead of 3 (for cross edges)? No, 1 clique of size 4 covers 3 cross edges and 3 C-edges.
        # 3 cliques of size 2 cover 3 cross edges and 0 C-edges.
        # So a triple is better if the 3 C-edges would otherwise need explicit cliques.
        # But if those C-edges are covered by other triples, we have a conflict.
        
        # Let's include triples in the candidate set.
        # Size <= 3: 1 + 21 + 210 + 1330 = 1562 candidates per i.
        # 60 * 1562 = 93720 variables.
        # Still manageable for SciPy's HiGHS solver.
        
        candidates = []
        for i in range(k, n):
            N_i = list(N[i])
            m = len(N_i)
            cand = [()] # Empty set
            
            # Singletons
            for c in N_i:
                cand.append((c,))
            
            # Pairs
            for u_idx in range(m):
                u = N_i[u_idx]
                for v_idx in range(u_idx + 1, m):
                    v = N_i[v_idx]
                    cand.append((u, v))
            
            # Triples
            for u_idx in range(m):
                u = N_i[u_idx]
                for v_idx in range(u_idx + 1, m):
                    v = N_i[v_idx]
                    for w_idx in range(v_idx + 1, m):
                        w = N_i[w_idx]
                        cand.append((u, v, w))
                        
            candidates.append(cand)
            
        # Total variables
        num_vars = sum(len(c) for c in candidates)
        
        # Objective: Minimize number of non-empty cliques
        # c_j = 1 if candidate j is non-empty, else 0
        c_obj = np.zeros(num_vars)
        var_idx = 0
        for i in range(k, n):
            for S in candidates[i]:
                if len(S) > 0:
                    c_obj[var_idx] = 1
                var_idx += 1
                
        # Constraints:
        # 1. Sum of x for each i = 1
        # 2. For each C-edge e, Sum of x for candidates containing e = 1
        
        A_rows = []
        b_lows = []
        b_ups = []
        
        # Constraint 1: Assignment
        for i in range(k, n):
            row = np.zeros(num_vars)
            start_idx = sum(len(candidates[j]) for j in range(i - k))
            for j, S in enumerate(candidates[i]):
                row[start_idx + j] = 1
            A_rows.append(row)
            b_lows.append(1)
            b_ups.append(1)
            
        # Constraint 2: Edge coverage
        # Map edge to index
        c_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edge_to_idx = {e: idx for idx, e in enumerate(c_edges)}
        
        for e in c_edges:
            row = np.zeros(num_vars)
            u, v = e
            for i in range(k, n):
                start_idx = sum(len(candidates[j]) for j in range(i - k))
                for j, S in enumerate(candidates[i]):
                    if u in S and v in S:
                        row[start_idx + j] = 1
            A_rows.append(row)
            b_lows.append(1)
            b_ups.append(1)
            
        A = np.array(A_rows)
        b_lower = np.array(b_lows)
        b_upper = np.array(b_ups)
        
        integrality = np.ones(num_vars)
        
        # Solve ILP
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A, b_lower, b_upper),
                integrality=integrality,
                bounds=Bounds(0, 1)
            )
            
            if res.status != 0:
                # Fallback to greedy if ILP fails
                return _fallback_partition(n, k, edges)
                
            x = res.x
            var_idx = 0
            chosen_S = {}
            for i in range(k, n):
                for j, S in enumerate(candidates[i]):
                    if x[var_idx] > 0.5:
                        chosen_S[i] = S
                        break
                    var_idx += 1
                else:
                    chosen_S[i] = ()
                    
            # Construct cliques
            cliques = []
            # 1. Big clique for C? No, we didn't assume that.
            # We only added cliques for I vertices and explicit C-edges if needed.
            # Wait, the ILP constraint says every C-edge is covered EXACTLY ONCE by some S_i.
            # So we don't need explicit C-edge cliques.
            
            for i in range(k, n):
                S = chosen_S[i]
                if S:
                    clique = [i] + list(S)
                    cliques.append(clique)
                    
            return [[int(x) for x in c] for c in cliques]
            
        except Exception:
            return _fallback_partition(n, k, edges)

    def _fallback_partition(n, k, edges):
        # Fallback: 1 big clique + 2-cliques for cross edges
        cliques = [[int(c) for c in range(k)]]
        for (u, v) in edges:
            if u >= k or v >= k:
                cliques.append([int(u), int(v)])
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append(_fallback_partition(spec[0], spec[1], edges))
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        results.append(cliques)
        
    return results

if __name__ == "__main__":
    construct()

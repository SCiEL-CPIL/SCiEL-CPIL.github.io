import time
import sys
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        # K edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # K is 0..k-1, I is k..n-1
        K_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        # 1. Find a clique cover of K
        # Greedy: repeatedly pick the largest clique in the remaining graph of K
        remaining_edges = set(edges)
        k_edges = [e for e in edges if e[0] < k and e[1] < k]
        remaining_k_edges = set(k_edges)
        
        cliques_K = []
        # Precompute all cliques in K? k is small.
        # Or just greedy.
        while remaining_k_edges:
            # Find largest clique
            max_size = 2
            best_clique = None
            # Try sizes from k down to 2
            for size in range(k, 1, -1):
                found = False
                for combo in combinations(K_vertices, size):
                    # Check if all edges in combo are in remaining_k_edges
                    ok = True
                    for i in range(size):
                        for j in range(i + 1, size):
                            u, v = combo[i], combo[j]
                            if u > v: u, v = v, u
                            if (u, v) not in remaining_k_edges:
                                ok = False
                                break
                        if not ok: break
                    if ok:
                        best_clique = combo
                        found = True
                        break
                if found:
                    break
            if best_clique is None:
                # Should not happen, but fallback to edge
                e = next(iter(remaining_k_edges))
                best_clique = (e[0], e[1])
            
            cliques_K.append(best_clique)
            # Remove edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v: u, v = v, u
                    remaining_k_edges.remove((u, v))
        
        # 2. MIP for I vertices
        m = len(cliques_K)
        # x[v_idx, c_idx] = 1 if I_vertex v_idx is assigned to clique c_idx
        # Constraints:
        # 1. If x[v, c] = 1, then clique c must be subset of N_K(v)
        # 2. Sum_v x[v, c] <= 1 for each c
        # 3. For each v, if x[v, c1] = 1 and x[v, c2] = 1, then clique c1 and c2 must be vertex-disjoint.
        
        # Pre-check compatibility
        # comp[v_idx][c_idx] = True if clique c_idx is subset of N_K(I_vertex v_idx)
        comp = [[False] * m for _ in range(len(I_vertices))]
        for vi, v in enumerate(I_vertices):
            for ci, c in enumerate(cliques_K):
                ok = True
                for u in c:
                    if not adj[v][u]:
                        ok = False
                        break
                comp[vi][ci] = ok
        
        # Check disjointness
        # disjoint[c1][c2] = True if c1 and c2 are vertex-disjoint
        disjoint = [[True] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i == j:
                    disjoint[i][j] = False # Can't assign same clique twice to same v? 
                    # Actually, x[v, c] is binary. So v can't be assigned to c twice.
                    # But the constraint is on pairs c1, c2 assigned to SAME v.
                    # If c1 == c2, it's the same variable.
                    # The constraint x[v, c1] * x[v, c2] = 0 applies if c1 != c2 and not disjoint.
                else:
                    if set(cliques_K[i]).intersection(set(cliques_K[j])):
                        disjoint[i][j] = False
        
        # Variables: x[vi, ci]
        n_vars = len(I_vertices) * m
        c = np.zeros(n_vars)
        # Objective: Minimize sum of uncovered edges for I
        # Uncovered edges for v = |N_K(v)| - sum(ci where x[vi,ci]=1) * |clique ci|
        # Since cliques assigned to v are disjoint, sum of sizes is size of union.
        for vi, v in enumerate(I_vertices):
            deg = sum(1 for u in K_vertices if adj[v][u])
            for ci in range(m):
                if comp[vi][ci]:
                    # Coefficient for x[vi, ci]
                    # We want to MINIMIZE uncovered, so MAXIMIZE covered.
                    # Covered = sum |clique ci| * x[vi, ci]
                    # So objective term is -|clique ci| * x[vi, ci]
                    c[vi * m + ci] = -len(cliques_K[ci])
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # 1. Sum_v x[v, c] <= 1
        for ci in range(m):
            row = np.zeros(n_vars)
            for vi in range(len(I_vertices)):
                row[vi * m + ci] = 1
            A_ub.append(row)
            b_ub.append(1)
            
        # 2. Disjointness for each v
        # x[vi, ci] * x[vi, cj] <= 1 - disjoint[ci][cj] ?
        # If not disjoint, x[vi, ci] + x[vi, cj] <= 1
        for vi in range(len(I_vertices)):
            for ci in range(m):
                for cj in range(ci + 1, m):
                    if not disjoint[ci][cj]:
                        if comp[vi][ci] and comp[vi][cj]:
                            row = np.zeros(n_vars)
                            row[vi * m + ci] = 1
                            row[vi * m + cj] = 1
                            A_ub.append(row)
                            b_ub.append(1)
        
        # If no constraints, handle separately
        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        else:
            constraints = LinearConstraint(np.zeros((0, n_vars)), -np.inf, np.zeros(0))
            
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
        except Exception:
            res = None
        
        # Construct solution
        cliques = []
        # Add K cliques
        for ci, c_k in enumerate(cliques_K):
            # Check if any v is assigned to it
            assigned_v = None
            if res and res.status == 0:
                for vi, v in enumerate(I_vertices):
                    if res.x[vi * m + ci] > 0.5:
                        assigned_v = v
                        break
            if assigned_v is not None:
                cliques.append(list(c_k) + [assigned_v])
            else:
                cliques.append(list(c_k))
        
        # Add remaining I edges
        for vi, v in enumerate(I_vertices):
            covered_neighbors = set()
            if res and res.status == 0:
                for ci in range(m):
                    if res.x[vi * m + ci] > 0.5:
                        covered_neighbors.update(cliques_K[ci])
            
            for u in K_vertices:
                if adj[v][u] and u not in covered_neighbors:
                    cliques.append([u, v])
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like behavior if timeout
            results.append([]) # Invalid, but prevents crash. Better to return best effort.
            break
        try:
            cliques = solve_split_graph(*spec)
            results.append(cliques)
        except Exception as e:
            # Fallback to simple edge list
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            
    return results

# EVOLVE-BLOCK-END

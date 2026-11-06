import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(vertices, adj):
        """Bron-Kerbosch to find all maximal cliques"""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            if not P:
                return
            
            # Choose pivot to minimize branching
            pivot = max(P | X, key=lambda u: len(adj.get(u, set()) & P))
            
            for v in list(P - adj.get(pivot, set())):
                bk(R + [v], P & adj.get(v, set()), X & adj.get(v, set()))
                P.remove(v)
                X.add(v)
        
        bk([], set(vertices), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = {u: set() for u in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        all_cliques = find_maximal_cliques(list(range(n)), adj)
        
        # Filter: keep only cliques that are actual cliques in the edge set
        edge_set = set(edges)
        valid_cliques = []
        for clique in all_cliques:
            is_valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid and len(clique) >= 2:
                valid_cliques.append(tuple(clique))
        
        # If no valid cliques found, fallback to edges
        if not valid_cliques:
            return [[u, v] for u, v in edges]
        
        # Create edge index
        edge_list = list(edge_set)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # For each clique, find which edges it covers
        clique_edges = []
        for clique in valid_cliques:
            covered = []
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u,v), max(u,v))
                    if e in edge_to_idx:
                        covered.append(edge_to_idx[e])
            clique_edges.append(covered)
        
        # ILP: minimize sum(x_i) subject to each edge covered exactly once
        m = len(valid_cliques)
        E = len(edge_list)
        
        if m == 0:
            return [[u, v] for u, v in edges]
        
        # Objective: minimize number of cliques
        c = [1.0] * m
        
        # Constraints: for each edge, sum of cliques covering it = 1
        # A[i, j] = 1 if clique j covers edge i
        A_rows = []
        for e_idx in range(E):
            row = [0.0] * m
            for c_idx, covered in enumerate(clique_edges):
                if e_idx in covered:
                    row[c_idx] = 1.0
            A_rows.append(row)
        
        # Build constraint matrix
        import numpy as np
        A = np.array(A_rows)
        b = np.ones(E)
        
        # Constraint: A @ x == b
        constraint = LinearConstraint(A, b, b)
        
        # Bounds: x in [0, 1]
        bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
        
        # Integrality: all binary
        integrality = np.ones(m)
        
        try:
            result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if result.status == 0 and result.x is not None:
                # Extract solution
                solution_cliques = []
                for i, x in enumerate(result.x):
                    if x > 0.5:
                        solution_cliques.append(list(valid_cliques[i]))
                
                if solution_cliques:
                    return solution_cliques
        except Exception:
            pass
        
        # Fallback: greedy
        remaining_edges = set(edge_list)
        partition = []
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find largest clique from valid_cliques that covers the most remaining edges
            best_clique = None
            best_count = -1
            for clique in valid_cliques:
                covered = 0
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        e = (min(u,v), max(u,v))
                        if e in remaining_edges:
                            covered += 1
                if covered > best_count:
                    best_count = covered
                    best_clique = clique
            
            if best_clique is None or best_count <= 0:
                # Pick any remaining edge
                e = next(iter(remaining_edges))
                best_clique = [e[0], e[1]]
            
            partition.append(list(best_clique))
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    remaining_edges.discard((min(u,v), max(u,v)))
        
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]

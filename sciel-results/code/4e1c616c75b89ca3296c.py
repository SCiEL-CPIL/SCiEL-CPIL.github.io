import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                mask ^= lsb
        return edges

    def find_maximal_cliques(adj, n):
        """
        Bron-Kerbosch with pivot to find all maximal cliques.
        Returns a list of cliques, each clique is a list of vertices.
        """
        cliques = []
        
        # Convert adj to set of neighbors for easier manipulation in BK
        # But bitmasks are faster for intersection.
        
        def bk(R, P, X):
            if time_up():
                return
            if P == 0 and X == 0:
                if R:
                    cliques.append(list(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            # We want a vertex with many neighbors in P to reduce P
            pivot = None
            max_deg = -1
            union = P | X
            m = union
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                m ^= lsb
            
            if pivot is None:
                # No pivot found, iterate all P
                candidates = P
            else:
                # Iterate P \ N(pivot)
                candidates = P & ~(adj[pivot])
            
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                
                P ^= (1 << v)
                X |= (1 << v)
                
                if time_up():
                    return

        bk([], (1 << n) - 1, 0)
        return cliques

    def solve_milp(cliques, edges, n):
        """
        Solve the exact cover problem using MILP.
        cliques: list of lists of vertices
        edges: list of (u, v)
        """
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        num_edges = len(edges)
        
        # Precompute which edges are in which clique
        # edge_index: dict (u,v) -> index in edges list
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # A is sparse matrix (num_edges, num_cliques)
        # We can build it as a list of (row, col, value) for scipy.sparse
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for c_idx, clique in enumerate(cliques):
            # For each pair in clique, if it's an edge, add to matrix
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v: u, v = v, u
                    if (u, v) in edge_index:
                        e_idx = edge_index[(u, v)]
                        rows.append(e_idx)
                        cols.append(c_idx)
                        data.append(1)
        
        if not rows:
            return []
            
        A = sp.csc_matrix((data, (rows, cols)), shape=(num_edges, num_cliques))
        
        # Objective: min sum(x)
        c = np.ones(num_cliques)
        
        # Constraints: A x = 1
        # x in {0, 1}
        # scipy.optimize.milp requires linear constraints in form lb <= A x <= ub
        # Here A x = 1, so lb = 1, ub = 1
        
        # Note: scipy milp uses (A_ub, b_ub) for <= and (A_eq, b_eq) for =?
        # Actually, scipy.optimize.milp(c, constraints, integrality, bounds)
        # constraints: LinearConstraint(A, lb, ub)
        
        try:
            constraint = LinearConstraint(A, 1, 1)
            integrality = np.ones(num_cliques) # All integer
            bounds = Bounds(0, 1)
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds, options={'time_limit': 10})
            
            if res.status == 0: # Optimal
                # Extract solution
                x = res.x
                # Threshold at 0.5
                selected = [i for i in range(num_cliques) if x[i] > 0.5]
                return [cliques[i] for i in selected]
            else:
                return None
        except Exception:
            return None

    def solve_greedy(adj, n, edges):
        """
        Fallback greedy: repeatedly find max clique in remaining graph.
        """
        rem = adj[:]
        cliques = []
        active = (1 << n) - 1
        
        # Precompute edge masks for quick removal?
        # We just update rem bitmasks.
        
        while active and not time_up():
            # Find max clique in subgraph induced by active
            # Use simple BK to find one max clique
            best_clique = []
            best_size = 0
            
            def bk_max(R, P, X):
                nonlocal best_clique, best_size
                if P == 0 and X == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                if time_up():
                    return
                # Simple pivot
                m = P | X
                pivot = None
                max_deg = -1
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    deg = (rem[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                    m ^= lsb
                if pivot is None:
                    candidates = P
                else:
                    candidates = P & ~(rem[pivot])
                
                m = candidates
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    R.append(v)
                    bk_max(R, P & rem[v], X & rem[v])
                    R.pop()
                    P ^= (1 << v)
                    X |= (1 << v)
                    if time_up():
                        return

            bk_max([], active, 0)
            
            if not best_clique:
                # Fallback to single edge
                found = False
                m = active
                while m and not found:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    nb = rem[u] & active
                    if nb:
                        lsb2 = nb & -nb
                        v = lsb2.bit_length() - 1
                        best_clique = [u, v]
                        found = True
                    m ^= lsb
                if not found:
                    break

            cliques.append(best_clique)
            
            # Remove edges of best_clique from rem
            for u in best_clique:
                for v in best_clique:
                    if u != v:
                        rem[u] &= ~(1 << v)
            
            # Update active
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

        # Cover remaining edges with pairs
        if active:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        edges = get_edges(adj, n)
        
        # Try MILP first
        cliques = find_maximal_cliques(adj, n)
        if cliques and not time_up():
            res = solve_milp(cliques, edges, n)
            if res is not None:
                return res
        
        # Fallback to Greedy
        return solve_greedy(adj, n, edges)

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]

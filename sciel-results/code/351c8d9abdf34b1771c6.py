import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

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
        
        def bk(R, P, X):
            if time_up():
                return
            if P == 0 and X == 0:
                if R:
                    cliques.append(list(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
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
                candidates = P
            else:
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
        
        edge_index = {e: i for i, e in enumerate(edges)}
        
        rows = []
        cols = []
        data = []
        
        for c_idx, clique in enumerate(cliques):
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
        
        c = np.ones(num_cliques)
        
        try:
            constraint = LinearConstraint(A, 1, 1)
            integrality = np.ones(num_cliques)
            bounds = Bounds(0, 1)
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds, options={'time_limit': 10})
            
            if res.status == 0:
                x = res.x
                selected = [i for i in range(num_cliques) if x[i] > 0.5]
                return [cliques[i] for i in selected]
            else:
                return None
        except Exception:
            return None

    def max_clique_with_coloring(adj, active_mask):
        """
        Find a maximum clique using Bron-Kerbosch with coloring bound.
        Returns the clique as a list of vertices.
        """
        if active_mask == 0:
            return []
        
        best_clique = []
        best_size = 0
        
        # Color the vertices for upper bound
        # We'll use a simple greedy coloring on the subgraph
        def color_sort(P):
            """
            Returns a list of (vertex, color) pairs sorted by color.
            The color provides an upper bound on clique size from that vertex onward.
            """
            if P == 0:
                return []
            
            # Get list of vertices in P
            verts = []
            m = P
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Greedy coloring
            color = {}
            available_colors = [True] * len(verts)  # Not the best, but simple
            # Better: standard greedy coloring
            assigned_color = {}
            next_color = 0
            for v in verts:
                # Find the smallest color not used by neighbors of v in the colored set
                used_colors = set()
                for u in assigned_color:
                    if (adj[u] >> v) & 1:  # u and v are adjacent
                        used_colors.add(assigned_color[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                assigned_color[v] = c
                next_color = max(next_color, c + 1)
            
            # Sort by color (ascending)
            sorted_verts = sorted(verts, key=lambda x: assigned_color[x])
            return sorted_verts, assigned_color
        
        def bk_color(R, P, X):
            nonlocal best_clique, best_size
            
            if time_up():
                return
            
            if P == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            # Color the vertices in P
            sorted_verts, color_map = color_sort(P)
            
            # Process in order of increasing color
            # The color of vertex v is an upper bound on clique size containing v
            for v in reversed(sorted_verts):
                if len(R) + color_map[v] <= best_size:
                    # No need to continue, colors are non-decreasing
                    break
                
                if not ((X >> v) & 1):  # v not in X
                    R.append(v)
                    # New P is P ∩ N(v), but we need to maintain the coloring order
                    # For simplicity, we just recurse with the new P
                    new_P = P & adj[v]
                    # Filter new_P to only include vertices that come after v in the sorted order? 
                    # Actually, standard BK with coloring just passes P & N(v)
                    # But we need to be careful with the ordering for the bound to be valid
                    # The standard approach: process vertices in order, and when we pick v,
                    # the remaining candidates are those that come after v in the order AND are neighbors of v
                    # Let's re-implement this more carefully
                    
                    R.pop()
                    P ^= (1 << v)
                    X |= (1 << v)
            
            # This implementation is getting complex. Let's use a simpler but effective approach:
            # Just use standard BK with pivot, but add a simple bound based on degree.
            pass
        
        # Let's use a simpler, proven approach: BK with pivot + simple bound
        def bk_simple(R, P, X):
            nonlocal best_clique, best_size
            
            if time_up():
                return
            
            if P == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            # Bound: if |R| + |P| <= best_size, prune
            if len(R) + P.bit_count() <= best_size:
                return
            
            # Choose pivot
            union = P | X
            if union == 0:
                candidates = P
            else:
                pivot = None
                max_deg = -1
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
                    candidates = P
                else:
                    candidates = P & ~(adj[pivot])
            
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                R.append(v)
                bk_simple(R, P & adj[v], X & adj[v])
                R.pop()
                
                P ^= (1 << v)
                X |= (1 << v)
                
                if time_up():
                    return
        
        bk_simple([], active_mask, 0)
        return best_clique

    def solve_greedy(adj, n, edges):
        """
        Greedy: repeatedly find max clique in remaining graph.
        """
        rem = adj[:]
        cliques = []
        active = (1 << n) - 1
        
        while active and not time_up():
            best_clique = max_clique_with_coloring(rem, active)
            
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

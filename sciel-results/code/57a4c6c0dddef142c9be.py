import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def bron_kerbosch_max_cliques(adj, n):
        """
        Find all maximal cliques using Bron-Kerbosch with pivoting.
        adj: list of bitsets for the graph
        Returns: list of cliques, each clique is a bitset
        """
        cliques = []
        
        def expand(R, P, X):
            if time.time() > deadline:
                return
            if P == 0 and X == 0:
                if R != 0:
                    cliques.append(R)
                return
            
            if P == 0:
                return
            
            # Choose pivot: vertex in P|X with most neighbors in P
            pivot_set = P | X
            best_pivot = -1
            best_count = -1
            ps = pivot_set
            while ps:
                v = (ps & -ps).bit_length() - 1
                count = (P & adj[v]).bit_count()
                if count > best_count:
                    best_count = count
                    best_pivot = v
                ps &= ps - 1
            
            if best_pivot >= 0:
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
            
            cs = candidates
            while cs:
                v = (cs & -cs).bit_length() - 1
                v_bit = 1 << v
                new_R = R | v_bit
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P &= ~v_bit
                X |= v_bit
                cs = candidates & ~v_bit  # recompute remaining candidates? No, candidates was fixed
                # Actually, we need to iterate through the original candidates minus v
                # Let me fix this: iterate through candidates properly
                break  # This loop structure is wrong, let me rewrite
            
            # Actually, let me rewrite the main loop properly
            cs = candidates
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                new_R = R | v_bit
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P &= ~v_bit
                X |= v_bit
                cs &= ~v_bit
        
        all_vertices = (1 << n) - 1
        expand(0, all_vertices, 0)
        return cliques
    
    def greedy_hitting_set(cliques, n):
        """
        Greedy hitting set: repeatedly pick the vertex that covers the most uncovered cliques.
        """
        if not cliques:
            return []
        
        # Filter cliques to only those with size >= 2
        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return []
        
        remaining = set(range(len(cliques)))
        hitting_set = set()
        
        # For each vertex, count how many cliques it hits
        # Greedy: pick vertex hitting most remaining cliques
        
        # Precompute for each vertex which cliques it's in
        vertex_to_cliques = [[] for _ in range(n)]
        for ci, clique in enumerate(cliques):
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v].append(ci)
                cs &= ~v_bit
        
        # Track which cliques are still uncovered
        uncovered = set(range(len(cliques)))
        
        while uncovered:
            # Find vertex that hits the most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in hitting_set:
                    continue
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            hitting_set.add(best_v)
            # Remove all cliques containing best_v from uncovered
            for ci in vertex_to_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)
            
            if not uncovered:
                break
        
        return sorted(hitting_set)
    
    def branch_and_bound_hitting_set(cliques, n, time_limit):
        """
        Branch and bound for minimum hitting set.
        """
        if not cliques:
            return []
        
        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return []
        
        # Sort cliques by size (smaller first for better pruning)
        cliques.sort(key=lambda c: c.bit_count())
        
        best_solution = None
        best_size = n + 1
        
        # Precompute vertex-to-cliques mapping
        vertex_to_cliques = [[] for _ in range(n)]
        for ci, clique in enumerate(cliques):
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v].append(ci)
                cs &= ~v_bit
        
        # Count occurrences
        vertex_count = [0] * n
        for v in range(n):
            vertex_count[v] = len(vertex_to_cliques[v])
        
        def solve(remaining_cliques, current_set, current_size):
            nonlocal best_solution, best_size
            
            if time.time() > time_limit:
                return
            
            if not remaining_cliques:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_set)
                return
            
            if current_size >= best_size:
                return
            
            # Pick the most constrained clique (smallest, or most frequent vertices)
            # Find the clique with the minimum number of available vertices
            best_ci = -1
            best_min_count = n + 1
            for ci in remaining_cliques:
                clique = cliques[ci]
                count = 0
                cs = clique
                while cs:
                    v_bit = cs & -cs
                    v = v_bit.bit_length() - 1
                    if v not in current_set:
                        count += 1
                    cs &= ~v_bit
                if count == 0:
                    return  # No valid completion
                if count < best_min_count:
                    best_min_count = count
                    best_ci = ci
            
            if best_ci == -1:
                return
            
            clique = cliques[best_ci]
            cs = clique
            vertices_in_clique = []
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                if v not in current_set:
                    vertices_in_clique.append(v)
                cs &= ~v_bit
            
            # Sort vertices by frequency (most frequent first)
            vertices_in_clique.sort(key=lambda v: -vertex_count[v])
            
            for v in vertices_in_clique:
                if time.time() > time_limit:
                    return
                if current_size >= best_size:
                    return
                
                current_set.add(v)
                # Remove all cliques containing v from remaining
                new_remaining = [ci for ci in remaining_cliques if v not in set(c for c in cliques[ci].bit_length() and [])]
                # Fix: properly filter
                new_remaining = []
                for ci in remaining_cliques:
                    c = cliques[ci]
                    if c & (1 << v):
                        continue
                    new_remaining.append(ci)
                
                solve(new_remaining, current_set, current_size + 1)
                current_set.discard(v)
        
        solve(list(range(len(cliques))), set(), 0)
        
        if best_solution is not None:
            return best_solution
        return greedy_hitting_set(cliques, n)
    
    def local_search_improve(hitting_set, cliques, n, max_iters=100):
        """
        2-opt local search: try swapping one vertex in with one out.
        """
        if not cliques:
            return hitting_set
        
        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return hitting_set
        
        current = set(hitting_set)
        
        # Verify validity
        def is_valid(hs):
            for c in cliques:
                if not (c & _bitset_from_set(hs)):
                    return False
            return True
        
        def _bitset_from_set(s):
            bs = 0
            for v in s:
                bs |= (1 << v)
            return bs
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removing each vertex in current
            for v in list(current):
                new_current = current - {v}
                if is_valid(new_current):
                    current = new_current
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adding + removing pairs
            for v_out in list(current):
                for v_in in range(n):
                    if v_in in current:
                        continue
                    new_current = (current - {v_out}) | {v_in}
                    if is_valid(new_current) and len(new_current) < len(current):
                        current = new_current
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(current)
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                if adj[u]:
                    touched.add(u)
            results.append(sorted(touched))
            continue
        
        adj_orig = generate_graph(n, p, seed)
        
        # Build complement graph
        adj_comp = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_orig[u] & (1 << v)):
                    adj_comp[u] |= (1 << v)
                    adj_comp[v] |= (1 << u)
        
        # Find all maximal cliques in original graph using Bron-Kerbosch
        # Maximal cliques in G correspond to maximal independent sets in complement
        # But we want cliques in G, so we run BK on adj_orig
        maximal_cliques = bron_kerbosch_max_cliques(adj_orig, n)
        
        # Filter to cliques with at least 2 vertices
        maximal_cliques = [c for c in maximal_cliques if c.bit_count() >= 2]
        
        if not maximal_cliques:
            results.append([])
            continue
        
        # Try branch and bound with time limit
        time_limit = min(deadline - (time.time() - start_time), 120.0)
        if time_limit < 1.0:
            time_limit = 1.0
        
        # First get a greedy solution as upper bound
        greedy_solution = greedy_hitting_set(maximal_cliques, n)
        
        # Try exact B&B with time limit
        exact_solution = branch_and_bound_hitting_set(maximal_cliques, n, time_limit)
        
        if exact_solution is not None and len(exact_solution) < len(greedy_solution):
            best = exact_solution
        else:
            best = greedy_solution
        
        # Local search improvement
        improved = local_search_improve(best, maximal_cliques, n, max_iters=50)
        
        results.append(improved)
    
    return results

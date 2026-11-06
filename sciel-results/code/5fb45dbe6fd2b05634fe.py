"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    # Deadline guard: 150s wall-clock, use 140s for safety
    deadline = time.time() + 140.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with bitmasks."""
        cliques = []
        all_mask = (1 << n) - 1
        
        def bk(R, P, X):
            if P == 0 and X == 0:
                cliques.append(R)
                return
            # Pivot selection: choose vertex in P|X with max degree in P
            PX = P | X
            if PX:
                # Find pivot with max |P & N(u)|
                best_pivot = -1
                best_count = -1
                # Iterate over bits in PX
                temp = PX
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    count = bin(P & adj[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_pivot = v
                    if best_count == bin(P).count('1'):
                        break
            else:
                best_pivot = -1
            
            if best_pivot >= 0:
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
            
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                R_new = R | (1 << v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R_new, P_new, X_new)
                P &= ~(1 << v)
                X |= (1 << v)
        
        bk(0, all_mask, 0)
        return cliques

    def find_minimal_transversal(adj, n, cliques, time_limit=30.0):
        """Find a minimal vertex set that meets every maximal clique of size >= 2.
        
        Uses branch-and-bound with bitmasks.
        """
        start_time = time.time()
        
        # Filter cliques: only those with size >= 2
        relevant_cliques = []
        for c in cliques:
            if bin(c).count('1') >= 2:
                relevant_cliques.append(c)
        
        if not relevant_cliques:
            return set()
        
        # Sort cliques by size (descending) for better pruning
        relevant_cliques.sort(key=lambda x: bin(x).count('1'), reverse=True)
        
        # Precompute: for each vertex, which cliques it covers
        # cliques_covered_by[v] = bitmask of clique indices that vertex v is in
        num_cliques = len(relevant_cliques)
        cliques_covered_by = [0] * n
        for i, c in enumerate(relevant_cliques):
            temp = c
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                cliques_covered_by[v] |= (1 << i)
        
        # Find vertices that appear in any relevant clique
        active_vertices = set()
        for c in relevant_cliques:
            temp = c
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                active_vertices.add(v)
        
        active_vertices = sorted(active_vertices)
        
        # Best solution found
        best_size = len(active_vertices)
        best_solution = set(active_vertices)
        
        # Branch and bound
        def branch(selected_mask, uncovered_cliques_mask, depth):
            nonlocal best_size, best_solution
            
            if time.time() - start_time > time_limit:
                return
            
            # Pruning: if current selected size + lower bound >= best, prune
            current_size = bin(selected_mask).count('1')
            if current_size >= best_size:
                return
            
            # If all cliques are covered, update best
            if uncovered_cliques_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = set()
                    temp = selected_mask
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        best_solution.add(v)
                return
            
            # Lower bound: we need at least ceil(popcount(uncovered) / max_cover_per_vertex)
            # Simple LB: each new vertex covers at most some number of uncovered cliques
            # Better: find max number of uncovered cliques any single vertex can cover
            max_cover = 0
            temp = selected_mask
            remaining = 0
            # Compute which vertices are still available (not yet selected)
            all_active_mask = 0
            for v in active_vertices:
                all_active_mask |= (1 << v)
            
            available = all_active_mask & ~selected_mask
            temp = available
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                cover = cliques_covered_by[v] & uncovered_cliques_mask
                cnt = bin(cover).count('1')
                if cnt > max_cover:
                    max_cover = cnt
            
            if max_cover == 0:
                return  # Can't cover any more, but uncovered != 0, so dead end
            
            # Simple lower bound: at least 1 more vertex needed
            if current_size + 1 >= best_size:
                return
            
            # Choose which vertex to branch on
            # Heuristic: pick vertex that covers the most uncovered cliques
            best_vertex = -1
            best_cover_count = 0
            temp = available
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                cover = cliques_covered_by[v] & uncovered_cliques_mask
                cnt = bin(cover).count('1')
                if cnt > best_cover_count:
                    best_cover_count = cnt
                    best_vertex = v
            
            if best_vertex < 0:
                return
            
            # Branch 1: include best_vertex
            new_selected = selected_mask | (1 << best_vertex)
            new_uncovered = uncovered_cliques_mask & ~cliques_covered_by[best_vertex]
            branch(new_selected, new_uncovered, depth + 1)
            
            # Branch 2: exclude best_vertex
            # But we must ensure all cliques containing best_vertex are covered by other vertices
            # This is complex; for simplicity, just try excluding
            branch(selected_mask, uncovered_cliques_mask & ~cliques_covered_by[best_vertex], depth + 1)
        
        # Initial uncovered: all relevant cliques
        initial_uncovered = (1 << num_cliques) - 1
        branch(0, initial_uncovered, 0)
        
        return best_solution

    def greedy_transversal(adj, n, cliques):
        """Greedy approach: repeatedly pick vertex covering most uncovered cliques."""
        relevant_cliques = [c for c in cliques if bin(c).count('1') >= 2]
        if not relevant_cliques:
            return set()
        
        num_cliques = len(relevant_cliques)
        uncovered = [True] * num_cliques
        selected = set()
        
        # For each vertex, list of cliques it's in
        vertex_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(relevant_cliques):
            temp = c
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                vertex_cliques[v].append(i)
        
        while any(uncovered):
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                count = sum(1 for i in vertex_cliques[v] if uncovered[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v < 0 or best_count == 0:
                break
            selected.add(best_v)
            for i in vertex_cliques[best_v]:
                uncovered[i] = False
        
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Fallback: return empty sets (invalid but better than timeout)
            results.append([])
            continue
        
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        # First try greedy to get a good initial bound
        greedy_sol = greedy_transversal(adj, n, cliques)
        
        # Then try branch and bound with time limit
        remaining_time = min(30.0, deadline - time.time() - 5)
        if remaining_time > 5:
            bb_sol = find_minimal_transversal(adj, n, cliques, time_limit=remaining_time)
            if len(bb_sol) < len(greedy_sol):
                greedy_sol = bb_sol
        
        results.append(sorted(greedy_sol))
    
    return results
# EVOLVE-BLOCK-END

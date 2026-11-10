import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """
        Yields all maximal cliques in the graph.
        Uses the pivot heuristic.
        """
        n = len(adj)
        
        def bk(R, P, X):
            if not P and not X:
                yield R
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            
            # Find pivot u in P ∪ X that maximizes |P ∩ N(u)|
            u = None
            max_inter = -1
            for v in pivot_candidates:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    u = v
            
            # Branch on vertices in P that are NOT neighbors of u
            candidates = P - adj[u]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                yield from bk(new_R, new_P, new_X)
                
                P.remove(v)
                X.add(v)
        
        yield from bk(set(), set(range(n)), set())

    def solve_hit_set(cliques, n):
        """
        Finds a minimum hitting set for the given list of cliques.
        Each clique is a frozenset of vertices.
        Returns the set of vertices.
        """
        if not cliques:
            return set()
        
        # Convert cliques to bitmasks for fast intersection checks
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        n_cliques = len(clique_masks)
        
        # Precompute: for each vertex, which cliques it covers
        vertex_covers = [0] * n  # bitmask of cliques covered by vertex v
        for v in range(n):
            bit_v = 1 << v
            covers = 0
            for i, m in enumerate(clique_masks):
                if m & bit_v:
                    covers |= (1 << i)
            vertex_covers[v] = covers
        
        # Precompute clique sizes
        clique_sizes = [m.bit_count() for m in clique_masks]
        
        def lower_bound(remaining_cliques_mask, current_size, best_size):
            """
            Stronger lower bound using greedy packing.
            remaining_cliques_mask: bitmask of which cliques are still uncovered.
            """
            if remaining_cliques_mask == 0:
                return 0
            
            # Simple bound: count uncovered cliques
            n_uncovered = remaining_cliques_mask.bit_count()
            if n_uncovered == 0:
                return 0
            
            # Find max coverage among vertices not in current set
            max_cov = 0
            for v in range(n):
                cov = (vertex_covers[v] & remaining_cliques_mask).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return float('inf')
            
            # Greedy packing lower bound
            lb = 0
            rem = remaining_cliques_mask
            while rem != 0:
                # Find vertex covering most remaining cliques
                best_v = -1
                best_cov = 0
                for v in range(n):
                    cov = (vertex_covers[v] & rem).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_cov == 0:
                    return float('inf')
                lb += 1
                rem &= ~vertex_covers[best_v]
                if lb + current_size >= best_size:
                    return lb  # Early exit
            
            return lb
        
        def greedy_solve(clique_masks, n, vertex_covers, order_seed=None):
            """
            Greedy hitting set with optional randomization.
            """
            if order_seed is not None:
                rng = random.Random(order_seed)
            else:
                rng = random.Random(42)
            
            remaining = (1 << len(clique_masks)) - 1  # all cliques uncovered
            selected = set()
            
            while remaining != 0:
                # Find vertex covering most remaining cliques
                best_v = -1
                best_cov = 0
                # Add some randomness to break ties
                for v in range(n):
                    cov = (vertex_covers[v] & remaining).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                    elif cov == best_cov and best_v != -1 and rng.random() < 0.3:
                        best_v = v
                
                if best_cov == 0:
                    break  # No vertex can cover remaining cliques (shouldn't happen)
                
                selected.add(best_v)
                remaining &= ~vertex_covers[best_v]
            
            return selected
        
        def local_search_2exchange(initial_set, clique_masks, n, vertex_covers, time_limit=None):
            """
            1-for-1 vertex replacement local search.
            Try to replace each vertex in the set with one that reduces size or maintains it.
            """
            current = set(initial_set)
            n_cliques = len(clique_masks)
            all_cliques_mask = (1 << n_cliques) - 1
            
            def is_valid(s):
                """Check if s hits all cliques."""
                covered = 0
                for v in s:
                    covered |= vertex_covers[v]
                return covered == all_cliques_mask
            
            # Verify initial set is valid
            if not is_valid(current):
                return initial_set
            
            improved = True
            while improved:
                improved = False
                if time_limit and time.time() > time_limit:
                    break
                
                # Try removing each vertex
                for v in list(current):
                    new_set = current - {v}
                    if is_valid(new_set):
                        current = new_set
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try 1-for-1 swaps: remove v, add w
                for v in list(current):
                    if time_limit and time.time() > time_limit:
                        break
                    
                    # Find all cliques that only v covers in current
                    only_v_covers = 0
                    for w in current:
                        if w != v:
                            only_v_covers |= vertex_covers[w]
                    
                    # Cliques that v covers but no other vertex in current covers
                    v_unique = vertex_covers[v] & ~only_v_covers
                    
                    # For each vertex w not in current, check if adding w covers v_unique
                    for w in range(n):
                        if w in current:
                            continue
                        
                        # Check if w covers all cliques that v uniquely covers
                        if (vertex_covers[w] & v_unique) == v_unique:
                            # Check if new set is valid
                            new_set = current - {v} | {w}
                            if is_valid(new_set):
                                current = new_set
                                improved = True
                                break
                    
                    if improved:
                        break
            
            return current
        
        # Strategy: Multiple greedy restarts + local search
        best_solution = None
        best_size = float('inf')
        
        # Try several greedy variants with different random seeds
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            seed = r if r == 0 else 1000 + r
            sol = greedy_solve(clique_masks, n, vertex_covers, seed)
            
            # Local search to improve
            sol = local_search_2exchange(sol, clique_masks, n, vertex_covers, deadline - 1.0)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
            
            if best_size <= 1:
                break
        
        # Also try exact B&B for small instances as backup
        if best_size is None or best_size > 15:
            # Use B&B with time limit
            best_mask = 0
            best_bb_size = float('inf')
            
            def solve_bb(remaining_mask, current_mask, current_size):
                nonlocal best_mask, best_bb_size
                
                if time.time() > deadline:
                    return
                
                if remaining_mask == 0:
                    if current_size < best_bb_size:
                        best_bb_size = current_size
                        best_mask = current_mask
                    return
                
                if current_size >= best_bb_size:
                    return
                
                # Lower bound
                lb = lower_bound(remaining_mask, current_size, best_bb_size)
                if current_size + lb >= best_bb_size:
                    return
                
                # Pick clique with fewest available vertices
                min_clique_idx = -1
                min_avail = n + 1
                for i in range(n_cliques):
                    if (remaining_mask >> i) & 1:
                        avail = 0
                        for v in range(n):
                            if (clique_masks[i] >> v) & 1:
                                avail += 1
                        if avail < min_avail:
                            min_avail = avail
                            min_clique_idx = i
                
                if min_clique_idx == -1:
                    return
                
                target = clique_masks[min_clique_idx]
                
                # Get vertices in target
                vertices = []
                for v in range(n):
                    if (target >> v) & 1:
                        vertices.append(v)
                
                # Order by coverage
                vertices.sort(key=lambda v: (vertex_covers[v] & remaining_mask).bit_count(), reverse=True)
                
                for v in vertices:
                    bit = 1 << v
                    new_mask = current_mask | bit
                    new_remaining = remaining_mask & ~vertex_covers[v]
                    solve_bb(new_remaining, new_mask, current_size + 1)
                    if best_bb_size <= 1:
                        return
            
            all_cliques = (1 << n_cliques) - 1
            solve_bb(all_cliques, 0, 0)
            
            if best_bb_size < best_size:
                best_size = best_bb_size
                result = set()
                for v in range(n):
                    if (best_mask >> v) & 1:
                        result.add(v)
                best_solution = result
        
        if best_solution is None:
            # Fallback: all vertices
            best_solution = set(range(n))
        
        return best_solution

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices for remaining instances
            while len(out) < 3:
                # Determine n for the next instance
                idx = len(out)
                n_next = (30, 35, 40)[idx]
                out.append(list(range(n_next)))
            break
        
        adj = random_graph(n, p, seed)
        
        # Get maximal cliques of size >= 2
        cliques = []
        for c in bron_kerbosch(adj):
            if len(c) >= 2:
                cliques.append(frozenset(c))
        
        # Solve hitting set
        solution = solve_hit_set(cliques, n)
        out.append(sorted(solution))
        
        # Check time
        if time.time() > deadline:
            while len(out) < 3:
                idx = len(out)
                n_next = (30, 35, 40)[idx]
                out.append(list(range(n_next)))
            break

    return out

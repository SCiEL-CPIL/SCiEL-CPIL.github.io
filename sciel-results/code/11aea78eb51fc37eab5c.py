# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_hitting_set_bnb(cliques, n, time_limit):
        """
        Solve minimum hitting set using Branch and Bound.
        cliques: list of frozensets (vertices to hit)
        n: number of vertices
        """
        if not cliques:
            return set()
        
        # Pre-filter: remove any clique that is a superset of another clique
        # If C1 is a superset of C2, hitting C2 is sufficient for C1.
        # This is a standard reduction for Hitting Set.
        cliques_list = list(cliques)
        # Sort by size ascending to make subset checks easier
        cliques_list.sort(key=len)
        
        reduced_cliques = []
        for c in cliques_list:
            is_subset = False
            for rc in reduced_cliques:
                if c.issuperset(rc):
                    is_subset = True
                    break
            if not is_subset:
                # Check if c is a subset of any existing reduced clique?
                # Since we sorted by size, existing ones are smaller or equal.
                # If c is a superset of an existing one, we skip.
                # We also need to remove existing ones if c is a subset of them?
                # No, we iterate small to large. If c is small, it might be subset of a large one added later?
                # Actually, if we add c to reduced, and later find c2 where c is subset of c2, c2 is skipped.
                # If we find c2 where c2 is subset of c, c2 would have been processed earlier (smaller) and c would be skipped.
                # So simple superset check against already added (smaller) is sufficient.
                reduced_cliques.append(c)
        
        if not reduced_cliques:
            return set()

        # Map cliques to indices for bitsets or lists
        # For n <= 40, we can use Python integers as bitsets for vertices?
        # Or just use sets. Sets are fine for n=40.
        
        # Precompute for each vertex, which cliques it hits
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(reduced_cliques):
            for v in c:
                vertex_cliques[v].append(i)
                
        # Greedy Initialization for Upper Bound
        # Select vertices that hit the most remaining cliques
        remaining_cliques = set(range(len(reduced_cliques)))
        greedy_sol = set()
        while remaining_cliques:
            # Find vertex with max coverage
            best_v = -1
            max_cov = -1
            for v in range(n):
                if v in greedy_sol:
                    continue
                cov = 0
                for ci in vertex_cliques[v]:
                    if ci in remaining_cliques:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_v = v
            
            if best_v == -1:
                break
                
            greedy_sol.add(best_v)
            # Remove all cliques hit by best_v
            for ci in vertex_cliques[best_v]:
                remaining_cliques.discard(ci)
                
        best_size = len(greedy_sol)
        best_solution = greedy_sol
        
        # If greedy is already 0 or 1, it's likely optimal (min hitting set >= 1 if cliques exist)
        # But we need to prove optimality or find better.
        
        # Branch and Bound
        # State: current selection, remaining cliques
        # We use a recursive approach with backtracking.
        
        # To speed up, we can order vertices by frequency in remaining cliques.
        
        global_time_limit = time_limit
        start_bnb_time = time.time()
        
        # We need a lower bound function.
        # Lower Bound: Greedy Packing (Set Packing on cliques)
        # Find a set of cliques that are pairwise disjoint? No, cliques in a graph share vertices.
        # Wait, "Hitting Set" lower bound:
        # If we can find k cliques that are pairwise disjoint (no common vertices), we need at least k vertices.
        # In a graph, maximal cliques often share vertices.
        # A better LB: 
        # Select a clique C. We must pick at least 1 vertex from C.
        # Suppose we pick v in C. This hits all cliques containing v.
        # The remaining problem is Hitting Set on (Cliques \ {cliques containing v}).
        # This is complex to bound directly without solving.
        
        # Alternative LB: Fractional or simple counting.
        # Let's try a simple recursive LB:
        # LB(remaining_cliques):
        # If empty, 0.
        # Pick clique C with smallest size.
        # LB = 1 + min_{v in C} LB(remaining_cliques - cliques_hit_by(v))
        # This is essentially the exact solution itself if we recurse all the way.
        # To make it a *bound* for pruning, we can use a heuristic version or just rely on the search order.
        
        # Given n is small (30-40) and cliques are sparse (p=0.2-0.3), the number of maximal cliques is manageable.
        # Let's implement a standard B&B with a simple LB:
        # LB = max(1, ceil(sum(1/|C_i| for C_i in remaining))) ? No, that's weak.
        
        # Let's use the "Greedy Independent Set" on the hypergraph of cliques?
        # If we find a set of cliques that are pairwise disjoint, the LB is the size of that set.
        # Since cliques in a graph intersect, this is hard.
        
        # Let's try a different approach: 
        # Since n is small, we can use a bitset representation for vertices (40 bits fits in int).
        # And for cliques, we store the bitset of vertices.
        
        cliques_bitsets = []
        for c in reduced_cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            cliques_bitsets.append(mask)
            
        # Precompute for each vertex, the bitmask of cliques it hits?
        # No, we need to know which cliques remain.
        
        # Let's use a simpler B&B:
        # Order vertices by degree (frequency in cliques).
        # Try to include the most frequent vertex first.
        
        vertex_order = sorted(range(n), key=lambda v: len(vertex_cliques[v]), reverse=True)
        
        # We will maintain a list of remaining cliques indices.
        # To make it fast, we'll use a set of indices for remaining cliques.
        
        # Optimization: If a clique is size 1, we MUST include that vertex.
        # Propagate this.
        
        def propagate(cliques_rem_set, forced_vertices):
            # cliques_rem_set: set of indices
            # forced_vertices: set of vertices that must be in the solution
            # Returns new_cliques_rem_set, new_forced_vertices, or None if conflict
            
            changed = True
            while changed:
                changed = False
                # Check for size 1 cliques
                for ci in list(cliques_rem_set):
                    c = reduced_cliques[ci]
                    if len(c) == 1:
                        v = next(iter(c))
                        if v not in forced_vertices:
                            forced_vertices.add(v)
                            changed = True
                        # Remove all cliques containing v
                        for other_ci in vertex_cliques[v]:
                            if other_ci in cliques_rem_set:
                                cliques_rem_set.discard(other_ci)
                                changed = True
                                
            return cliques_rem_set, forced_vertices

        # B&B Function
        # args: current_selected (set), remaining_cliques (set of indices), depth
        # We want to minimize size.
        
        # We use a global best_size.
        
        # To avoid deep recursion issues, we can use a stack or just recursion with n=40 is fine.
        
        # Let's add a time check in the recursion.
        
        def bnb(selected_size, selected_vertices, remaining_cliques, depth):
            nonlocal best_size, best_solution, global_time_limit, start_bnb_time
            
            if time.time() > global_time_limit:
                return
                
            if not remaining_cliques:
                if selected_size < best_size:
                    best_size = selected_size
                    best_solution = set(selected_vertices)
                return
                
            # Lower Bound:
            # If selected_size + 1 >= best_size, and we have remaining cliques, we might prune?
            # Actually, if selected_size >= best_size, prune.
            if selected_size >= best_size:
                return
                
            # Propagation
            rem, forced = propagate(set(remaining_cliques), set(selected_vertices))
            
            if not rem:
                if selected_size + len(forced - selected_vertices) < best_size:
                    # Wait, forced vertices are already in selected? 
                    # In propagate, I added to forced. If they were not in selected, they increase size.
                    # Let's handle forced vertices explicitly.
                    # If a vertex is forced, it must be in the solution.
                    # If it's not in selected, we add it.
                    new_selected = set(selected_vertices)
                    for v in forced:
                        if v not in new_selected:
                            new_selected.add(v)
                            selected_size += 1
                    if selected_size < best_size:
                        best_size = selected_size
                        best_solution = new_selected
                return

            # If we have forced vertices that are not in selected, we must add them.
            # But propagate already removed cliques hit by them.
            # So we can just recurse with the new state.
            
            # Choose a vertex to branch on.
            # Pick the clique with smallest size in rem.
            min_ci = min(rem, key=lambda i: len(reduced_cliques[i]))
            min_c = reduced_cliques[min_ci]
            
            # Branch on each vertex in min_c
            # Order vertices by frequency in remaining cliques
            verts_in_c = list(min_c)
            verts_in_c.sort(key=lambda v: len([ci for ci in vertex_cliques[v] if ci in rem]), reverse=True)
            
            for v in verts_in_c:
                if time.time() > global_time_limit:
                    return
                if v in selected_vertices:
                    continue
                
                # Branch: include v
                new_selected = set(selected_vertices)
                new_selected.add(v)
                
                # Remove all cliques hit by v
                new_rem = set(remaining_cliques)
                for ci in vertex_cliques[v]:
                    new_rem.discard(ci)
                    
                bnb(selected_size + 1, new_selected, new_rem, depth + 1)
                
                # Note: We don't have a "exclude v" branch because we must hit min_c.
                # Since min_c must be hit, we must pick one of its vertices.
                # So we branch on all vertices in min_c.
                
        # Start B&B
        # Initial state: selected empty, all reduced cliques remaining
        bnb(0, set(), set(range(len(reduced_cliques))), 0)
        
        return best_solution

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            return []

        # Solve using B&B
        # Allocate most of the time budget to B&B
        sol = solve_hitting_set_bnb(cliques, n, time_budget)
        
        return sorted(sol)

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 20:
            # Fallback: return all vertices (valid but large)
            results.append(list(range(n)))
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        # Reserve 10s for overhead and safety
        per_instance = (deadline - 10 - elapsed) / max(1, remaining_instances)
        per_instance = max(1.0, min(per_instance, deadline - 10 - elapsed))

        sol = solve_instance(n, p, seed, start_time + elapsed + per_instance)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END

"""Stronger algorithm for hitting all maximal cliques (size >= 2) in three random graphs."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
            else:
                return
            # Only recurse on vertices in P that are NOT adjacent to pivot
            candidates = [v for v in P if v not in adj[pivot]] if pivot is not None else list(P)
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def max_clique_in_complement(adj_complement, n):
        """Find maximum clique in a graph given as adjacency sets (complement graph)."""
        best = []
        
        def bk_max(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            pivot = max(pivot_candidates, key=lambda v: len(adj_complement[v] & P))
            candidates = [v for v in P if v not in adj_complement[pivot]]
            for v in candidates:
                bk_max(R | {v}, P & adj_complement[v], X & adj_complement[v])
                P.remove(v)
                X.add(v)
        
        bk_max(set(), set(range(n)), set())
        return best

    def remove_redundant(selected, cliques):
        """Remove vertices that are not needed (their removal still leaves all cliques hit)."""
        if not selected:
            return selected
        result = set(selected)
        
        for v in list(result):
            if time.time() > deadline:
                break
            test_set = result - {v}
            valid = True
            for c in cliques:
                if not (c & test_set):
                    valid = False
                    break
            if valid:
                result.discard(v)
        
        return result

    def greedy_hitting(cliques, adj, n, rng=None, max_iters=10000):
        """Greedy set cover with optional random tie-breaking and incremental updates."""
        if not cliques:
            return set()
        
        remaining = list(cliques)
        selected = set()
        
        while remaining and len(selected) < max_iters:
            if time.time() > deadline:
                break
            
            # Build coverage count for each vertex
            counts = {}
            for c in remaining:
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            
            if not counts:
                break
            
            max_count = max(counts.values())
            best_vertices = [v for v, cnt in counts.items() if cnt == max_count]
            
            if rng:
                v = rng.choice(best_vertices)
            else:
                v = best_vertices[0]
            
            selected.add(v)
            remaining = [c for c in remaining if v not in c]
        
        return selected

    def local_search(selected, cliques, adj, n, max_rounds=100):
        """Enhanced local search with 1-removal, 1-for-1 swap, and 2-for-1 moves."""
        if not selected:
            return selected
        
        selected_set = set(selected)
        best_size = len(selected_set)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 1-removal: can we drop a vertex?
            for v in list(selected_set):
                test = selected_set - {v}
                if all(c & test for c in cliques):
                    selected_set.discard(v)
                    best_size = len(selected_set)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swap: remove v, add w
            found_swap = False
            for v in list(selected_set):
                if time.time() > deadline:
                    break
                
                # Find cliques that only v covers
                critical = [c for c in cliques if v in c and not (c - {v}) & selected_set]
                if not critical:
                    # v is redundant
                    selected_set.discard(v)
                    best_size = len(selected_set)
                    improved = True
                    break
                
                # Candidate w must be in all critical cliques
                if len(critical) == 1:
                    candidates = [w for w in critical[0] if w not in selected_set]
                else:
                    inter = set(critical[0])
                    for c in critical[1:]:
                        inter &= c
                    candidates = [w for w in inter if w not in selected_set]
                
                if not candidates:
                    continue
                
                # Pick w that maximizes additional coverage
                best_w = None
                best_gain = -1
                for w in candidates:
                    # Count cliques that w covers which are not covered by (selected - v)
                    gain = 0
                    for c in cliques:
                        if w in c and not (c - {w}) & (selected_set - {v}):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                
                if best_w is not None and best_gain > 0:
                    selected_set.discard(v)
                    selected_set.add(best_w)
                    if len(selected_set) < best_size:
                        best_size = len(selected_set)
                    improved = True
                    found_swap = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1: remove two vertices {u,v}, add one w
            if not found_swap:
                found_two_for_one = False
                vertices_list = list(selected_set)
                for i in range(len(vertices_list)):
                    for j in range(i + 1, len(vertices_list)):
                        u, v = vertices_list[i], vertices_list[j]
                        # Find cliques that only u or v (or both) cover
                        critical = [c for c in cliques if (u in c or v in c) and not (c - {u, v}) & selected_set]
                        if not critical:
                            selected_set.discard(u)
                            selected_set.discard(v)
                            best_size = len(selected_set)
                            improved = True
                            found_two_for_one = True
                            break
                        
                        # Candidate w must be in all critical cliques
                        if len(critical) == 1:
                            candidates = [w for w in critical[0] if w not in selected_set]
                        else:
                            inter = set(critical[0])
                            for c in critical[1:]:
                                inter &= c
                            candidates = [w for w in inter if w not in selected_set]
                        
                        if not candidates:
                            continue
                        
                        # Pick w that maximizes additional coverage
                        best_w = None
                        best_gain = -1
                        for w in candidates:
                            gain = 0
                            for c in cliques:
                                if w in c and not (c - {w}) & (selected_set - {u, v}):
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_w = w
                        
                        if best_w is not None and best_gain > 0:
                            selected_set.discard(u)
                            selected_set.discard(v)
                            selected_set.add(best_w)
                            if len(selected_set) < best_size:
                                best_size = len(selected_set)
                            improved = True
                            found_two_for_one = True
                            break
                    if found_two_for_one:
                        break
            
            if not improved:
                break
        
        return selected_set

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []
        
        # Precompute: for each clique, its member set as a frozenset for fast intersection
        # Also build vertex -> cliques incidence for efficiency
        vertex_to_cliques = {}
        for c in cliques:
            for v in c:
                vertex_to_cliques.setdefault(v, []).append(c)
        
        # Build clique-intersection graph: nodes are cliques, edges connect cliques sharing a vertex
        num_cliques = len(cliques)
        # For each vertex, list of clique indices containing it
        vertex_to_clique_indices = {}
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_to_clique_indices.setdefault(v, []).append(ci)
        
        # Build complement graph for max clique finding
        # Two cliques are "disjoint" (non-adjacent in intersection graph) if they share no vertex
        # We want max set of pairwise disjoint cliques = max clique in complement of intersection graph
        # adj_complement[ci] = set of cj where cliques[ci] and cliques[cj] are DISJOINT
        adj_complement = [set() for _ in range(num_cliques)]
        for v in vertex_to_clique_indices:
            indices = vertex_to_clique_indices[v]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    # cliques[indices[i]] and cliques[indices[j]] share v, so they are NOT disjoint
                    # In complement, they should NOT have an edge
                    pass
        # Actually, build it directly: two cliques are adjacent in complement iff they are disjoint
        for ci in range(num_cliques):
            for cj in range(ci + 1, num_cliques):
                if not (cliques[ci] & cliques[cj]):  # disjoint
                    adj_complement[ci].add(cj)
                    adj_complement[cj].add(ci)
        
        # Find maximum set of pairwise disjoint cliques (lower bound on hitting set size)
        if num_cliques > 0:
            max_disjoint = max_clique_in_complement(adj_complement, num_cliques)
            lower_bound = len(max_disjoint)
        else:
            lower_bound = 0
        
        best_solution = None
        best_size = float('inf')
        
        rng = random.Random(seed + 1000)
        instance_start = time.time()
        instance_deadline = deadline - 10.0  # leave 10s buffer for other instances
        
        # Adaptive restarts: use more if we're fast
        num_restarts = 80
        
        for restart in range(num_restarts):
            if time.time() > instance_deadline:
                break
            
            # Greedy with random tie-breaking
            selected = greedy_hitting(cliques, adj, n, rng=rng)
            
            # Remove redundant vertices
            selected = remove_redundant(selected, cliques)
            
            # Local search with adaptive rounds
            rounds = 30 if len(selected) > 15 else 60
            selected = local_search(selected, cliques, adj, n, max_rounds=rounds)
            
            # Final redundancy removal
            selected = remove_redundant(selected, cliques)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
            
            if best_size == 0:
                break
            
            # If we've found a small solution, do more intensive local search
            if best_size <= 12 and time.time() < instance_deadline - 5:
                selected = local_search(selected, cliques, adj, n, max_rounds=150)
                selected = remove_redundant(selected, cliques)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
        
        # Branch-and-bound using lower bound from max disjoint cliques
        if best_solution is not None and best_size > lower_bound:
            # Try to improve using guided search
            # Use the structure of max disjoint cliques to guide choices
            if time.time() < instance_deadline - 3:
                # For each vertex in best_solution, check if it's critical
                # Try targeted swaps guided by disjoint clique structure
                critical_cliques_per_vertex = {}
                for v in best_solution:
                    crit = [c for c in cliques if v in c and not (c - {v}) & best_solution]
                    if crit:
                        critical_cliques_per_vertex[v] = crit
                
                # Try to replace pairs of vertices that cover disjoint critical cliques
                vertices_list = list(best_solution)
                for i in range(len(vertices_list)):
                    if time.time() > instance_deadline:
                        break
                    for j in range(i + 1, len(vertices_list)):
                        u, v = vertices_list[i], vertices_list[j]
                        # Find cliques only covered by u or v
                        critical = [c for c in cliques if (u in c or v in c) and not (c - {u, v}) & best_solution]
                        if not critical:
                            # Both redundant
                            new_sol = best_solution - {u, v}
                            if all(c & new_sol for c in cliques):
                                best_solution = new_sol
                                best_size = len(best_solution)
                                break
                        else:
                            # Find a single vertex w that covers all critical cliques
                            inter = set(critical[0])
                            for c in critical[1:]:
                                inter &= c
                            for w in inter:
                                if w not in best_solution:
                                    new_sol = (best_solution - {u, v}) | {w}
                                    if all(c & new_sol for c in cliques):
                                        if len(new_sol) < best_size:
                                            best_solution = new_sol
                                            best_size = len(best_solution)
                                        break
                    if best_size <= lower_bound:
                        break
                    if best_size <= lower_bound + 1:
                        break
        
        if best_solution is None:
            best_solution = greedy_hitting(cliques, adj, n)
            best_solution = remove_redundant(best_solution, cliques)
        
        return sorted(best_solution)

    # Solve each instance with time budget allocation
    results = []
    instance_configs = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instance_configs):
        # Allocate time: give more to harder instances
        if idx == 0:
            time_budget = 40.0
        elif idx == 1:
            time_budget = 45.0
        else:
            time_budget = 50.0
        
        instance_deadline = start_time + time_budget * (idx + 1) + 10.0
        if idx == 2:
            instance_deadline = deadline - 5.0
        
        if time.time() > instance_deadline:
            # Emergency: return minimal valid set (all vertices in any edge)
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
            continue
        
        sol = solve_instance(n, p, seed)
        results.append(sol)
    
    return results

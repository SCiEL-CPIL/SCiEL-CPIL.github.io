import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, graph, n):
        """
        Enumerates maximal cliques.
        R: current clique (frozen set)
        P: candidates that can extend R (set)
        X: candidates that were in P but are now excluded (set)
        graph: list of sets, adjacency list
        """
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        # Pivot selection: choose vertex in P union X with max |P \cap N(p)|
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
            
        max_inter = -1
        pivot = pivot_candidates[0]
        for p in pivot_candidates:
            inter = len(P & graph[p])
            if inter > max_inter:
                max_inter = inter
                pivot = p
        
        # Iterate over P \ N(pivot)
        for v in list(P - graph[pivot]):
            new_R = R | {v}
            new_P = P & graph[v]
            new_X = X & graph[v]
            yield from bron_kerbosch(new_R, new_P, new_X, graph, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find maximal cliques of size >= 2
        cliques = set()
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        C = len(valid_cliques)
        
        # Pre-compute incidence
        # vertex_cliques[v] = set of clique indices containing v
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        # Use multiple random restarts for the greedy + local search
        # We will use a simple greedy with random tie-breaking, then 1-move local search (add/remove/swap)
        
        num_restarts = 300
        rng_solver = random.Random(seed + 123)
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Greedy Phase
            hit_cliques = set()
            current_set = set()
            
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    # Should not happen if there are unhit cliques
                    break
                    
                # Random tie-breaking
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            # Local Search
            # 1. Removal: if a vertex is redundant, remove it.
            # 2. Swap: try to replace one vertex with another to potentially enable future removals.
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                    
                # Phase 1: Removal
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        # Check if clique ci is hit by any other vertex in current_set
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        break # Restart the loop to re-check removals in random order
                
                if improved:
                    continue
                    
                # Phase 2: Swap
                # Try swapping a vertex u in S with a vertex v not in S
                # Condition: S - {u} + {v} is a valid cover.
                # This is expensive to check naively. 
                # Optimization: 
                # For a swap to be valid, every clique hit ONLY by u in S must be hit by v.
                # Let U_only = {c in valid_cliques : u in c and no other v' in S \ {u} is in c}
                # We need v to be in all cliques in U_only.
                
                # If we find such a swap, we perform it and then try to remove redundancies again.
                # To keep it fast, we limit the number of swap attempts per restart.
                
                if len(current_set) > 0:
                    # Sample a few swaps
                    # We pick u from S randomly
                    # We pick v from V\S randomly
                    # Check validity
                    # If valid, swap and break to go back to removal phase.
                    
                    # Efficient check:
                    # Precompute for current_set: 
                    # exclusive_c[u] = set of cliques covered ONLY by u in current_set
                    # This depends on current_set, so we recompute or maintain.
                    # Recomputing for n=40, C~100 is cheap.
                    
                    exclusive = {u: set() for u in current_set}
                    for ci, c in enumerate(valid_cliques):
                        # Find which vertices in current_set are in c
                        coverers = [u for u in current_set if u in c]
                        if len(coverers) == 1:
                            u = coverers[0]
                            exclusive[u].add(ci)
                    
                    # Now, for a swap (u, v) to be valid, v must be in all cliques in exclusive[u].
                    # We can iterate u in current_set, and for each u, check candidates v.
                    
                    # To save time, we only try a limited number of swaps.
                    swaps_tried = 0
                    max_swaps = 20
                    
                    while swaps_tried < max_swaps:
                        swaps_tried += 1
                        u = rng_solver.choice(list(current_set))
                        if not exclusive[u]:
                            continue # u is not exclusive for any clique, swapping it might not help cover, but might allow removal?
                                     # Actually if exclusive[u] is empty, u is redundant? No, redundant means exclusive[u] is empty AND u is not needed for any clique? 
                                     # Wait, if exclusive[u] is empty, it means every clique covered by u is covered by at least one other vertex in S.
                                     # Then u IS redundant. Why didn't Phase 1 remove it?
                                     # Phase 1 checks: "if no other vertex in S covers ci". 
                                     # If exclusive[u] is empty, then for every ci in vertex_cliques[u], there is another vertex.
                                     # So u should have been removed.
                                     # So if we are in Phase 2, exclusive[u] must be non-empty for all u? 
                                     # Not necessarily. If u covers cliques, but all are covered by others, u is redundant.
                                     # So Phase 1 should have caught it.
                                     # So if we are here, likely all u have non-empty exclusive[u]? 
                                     # Or maybe Phase 1 failed to find a removable vertex due to order? No, it shuffles.
                                     # So if Phase 1 finishes, no vertex is redundant.
                                     # Thus, for every u in S, there exists at least one clique covered ONLY by u.
                                     # So exclusive[u] is non-empty.
                        
                        # We need v such that exclusive[u] subset of vertex_cliques[v]
                        # i.e., v is in all cliques in exclusive[u].
                        
                        # Candidates for v: must be in intersection of all cliques in exclusive[u].
                        # Let Inter = intersection of cliques in exclusive[u].
                        # v must be in Inter and v not in S.
                        
                        if not exclusive[u]:
                            continue
                            
                        inter_cliques = set(valid_cliques[ci] for ci in exclusive[u])
                        # v must be in inter_cliques
                        # and v not in current_set
                        
                        # If inter_cliques is empty, no such v exists?
                        # Wait, intersection of sets of vertices.
                        # If exclusive[u] contains cliques c1, c2.
                        # v must be in c1 AND c2.
                        # So v must be in c1 ∩ c2.
                        
                        candidates_v = inter_cliques
                        # Remove vertices already in S
                        candidates_v -= current_set
                        
                        if not candidates_v:
                            continue
                            
                        # Pick a random v from candidates_v
                        v = rng_solver.choice(list(candidates_v))
                        
                        # Perform swap
                        current_set.remove(u)
                        current_set.add(v)
                        improved = True
                        break # Go back to removal phase
                        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices involved in edges
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results

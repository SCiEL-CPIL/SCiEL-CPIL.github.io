import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        
        if not edges:
            return []

        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        end_time = start_time + time_budget

        # Helper: Check if two cliques can be merged (union is a clique)
        def can_merge(clique1, clique2):
            for u in clique1:
                for v in clique2:
                    if v not in adj[u]:
                        return False
            return True

        # Helper: Perform a merge pass on a list of cliques
        def merge_pass(cliques):
            changed = True
            while changed and time.time() < end_time - 2:
                changed = False
                # Sort by size descending to prioritize merging large ones? 
                # Or just iterate all pairs.
                # To speed up, we can break early if no merges found in a full pass.
                i = 0
                while i < len(cliques) and time.time() < end_time - 2:
                    j = i + 1
                    while j < len(cliques) and time.time() < end_time - 2:
                        if can_merge(cliques[i], cliques[j]):
                            # Merge
                            merged = list(cliques[i]) + list(cliques[j])
                            # Replace cliques[i] with merged, remove cliques[j]
                            cliques[i] = merged
                            cliques.pop(j)
                            changed = True
                            # Do not increment j, check new j with i
                        else:
                            j += 1
                    i += 1
            return cliques

        best_cliques = None
        best_count = float('inf')

        attempts = 0
        while time.time() < end_time - 5:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 777 + 111)

            # 1. Greedy Construction
            remaining_edges = set(edges)
            cliques = []
            
            # Order vertices for greedy expansion
            # Randomize order to get different solutions
            vertices = list(range(n))
            attempt_rng.shuffle(vertices)
            
            # We pick edges in a specific order? 
            # Standard greedy: pick an edge, expand.
            # To vary, we can pick the edge with the highest "potential" (common neighbors) 
            # or just random edge.
            
            # Let's use the "best seed edge" heuristic from the original, but randomized
            # Or simpler: just pick random edge from remaining, expand greedily.
            # Randomization comes from the order we pick edges and vertices.
            
            while remaining_edges and time.time() < end_time - 5:
                # Pick a random edge from remaining
                # To be faster, we can pick the one with max common neighbors?
                # That was the original approach. Let's stick to a randomized greedy.
                
                # Heuristic: Pick edge (u,v) that maximizes common neighbors in remaining
                # But to allow structural variation, we sometimes pick random.
                if attempt_rng.random() < 0.8:
                    # Greedy choice: max common neighbors
                    best_edge = None
                    max_common = -1
                    # Sample a subset of edges if too many? 
                    # For n=64, edges ~ 1000. Iterating all is fine.
                    for e in remaining_edges:
                        u, v = e
                        common = adj[u].intersection(adj[v])
                        count = 0
                        for w in common:
                            if w == u or w == v: continue
                            if (min(w, u), max(w, u)) in remaining_edges and (min(w, v), max(w, v)) in remaining_edges:
                                count += 1
                        if count > max_common:
                            max_common = count
                            best_edge = e
                    if best_edge is None:
                        best_edge = next(iter(remaining_edges))
                else:
                    best_edge = attempt_rng.choice(list(remaining_edges))
                
                u, v = best_edge
                current_clique = [u, v]
                
                # Expand
                # Candidates: neighbors of u and v, with edges in remaining
                cand_set = adj[u].intersection(adj[v])
                cand_set.discard(u)
                cand_set.discard(v)
                
                # Filter: must have edges to current_clique in remaining
                cand_set = {w for w in cand_set if all((min(w, c), max(w, c)) in remaining_edges for c in current_clique)}
                
                # Add candidates in random order (shuffled by attempt_rng)
                cand_list = list(cand_set)
                attempt_rng.shuffle(cand_list)
                
                for w in cand_list:
                    # Check if w can still be added
                    # It must be adjacent to all in current_clique
                    # And edges must be in remaining
                    if all((min(w, c), max(w, c)) in remaining_edges for c in current_clique):
                        current_clique.append(w)
                        # Remove edges incident to w that are in remaining? 
                        # No, we remove them when we finalize the clique.
                        # But we must ensure we don't add a vertex that uses an edge already "reserved"?
                        # In this greedy, we just check if edges exist in remaining.
                        # Since we are building one clique at a time, and we remove all its edges at the end,
                        # this is safe.
                
                # Finalize clique
                clique_edges = set()
                for i in range(len(current_clique)):
                    for j in range(i+1, len(current_clique)):
                        e = (min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j]))
                        clique_edges.add(e)
                
                cliques.append(current_clique)
                for e in clique_edges:
                    remaining_edges.discard(e)

            # 2. Merge Pass
            cliques = merge_pass(cliques)
            
            # 3. Update Best
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

        # Fallback if no solution found (should not happen)
        if best_cliques is None:
            best_cliques = [list(e) for e in edges]

        return best_cliques

    # Solve both instances
    t1 = time.time()
    budget1 = min(30, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(110, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]

import time
import random
from itertools import combinations

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(R, P, X, graph, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        U = P.union(X)
        if U:
            pivot = max(U, key=lambda u: len(graph[u].intersection(P)))
        else:
            pivot = None
        
        if pivot is not None:
            neighbors_of_pivot = graph[pivot]
            candidates = P.difference(neighbors_of_pivot)
        else:
            candidates = P

        for v in list(candidates):
            bron_kerbosch(R | {v}, P & graph[v], X & graph[v], graph, cliques)
            P.remove(v)
            X.add(v)

    def find_maximal_cliques(n, edges):
        # Build adjacency sets
        graph = {i: set() for i in range(n)}
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Remove isolated vertices? They can't be in cliques of size >= 2.
        # But Bron-Kerbosch handles them naturally (they won't be in P if we start with all vertices, 
        # but they will be removed quickly or just not form cliques).
        # Actually, better to start P with vertices that have degree > 0 to save time.
        active_vertices = {v for v in range(n) if len(graph[v]) > 0}
        
        cliques = []
        bron_kerbosch(set(), active_vertices, set(), graph, cliques)
        return cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()
        
        # Map vertex to list of cliques it belongs to
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        uncovered = set(range(len(cliques)))
        S = set()
        
        while uncovered:
            best_v = -1
            max_covered = -1
            # Use a scoring function that considers both coverage and clique sizes
            for v in range(n):
                if v in S:
                    continue
                covered = sum(1 for i in vertex_to_cliques[v] if i in uncovered)
                if covered > max_covered:
                    max_covered = covered
                    best_v = v
            if best_v == -1:
                break
            S.add(best_v)
            for i in vertex_to_cliques[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
        
        return S

    def local_search(S, cliques, n, time_limit=5.0):
        start_time = time.time()
        # Precompute for each clique, the set of vertices in S that hit it
        # To check validity of removal of v:
        # v is redundant if for all c in cliques, c has some other vertex in S.
        # Equivalent: For all c containing v, |c intersect S| > 1.
        
        # Let's maintain counts: count[c] = number of vertices in S that are in clique c
        counts = [0] * len(cliques)
        vertex_in_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_in_cliques[v].append(i)
        
        for v in S:
            for i in vertex_in_cliques[v]:
                counts[i] += 1
                
        current_S = set(S)
        
        # Helper to try removing v
        def try_remove(v):
            nonlocal counts
            # Check if removing v makes any clique uncovered
            can_remove = True
            for i in vertex_in_cliques[v]:
                if counts[i] == 1:
                    can_remove = False
                    break
            if can_remove:
                for i in vertex_in_cliques[v]:
                    counts[i] -= 1
                current_S.remove(v)
                return True
            return False

        # Helper to try adding v
        def try_add(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] += 1
            current_S.add(v)
            return True

        # Helper to remove v (undo)
        def undo_remove(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] += 1
            current_S.add(v)

        # Helper to remove v (undo add)
        def undo_add(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] -= 1
            current_S.remove(v)

        improved = True
        pass_count = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            pass_count += 1
            
            # Phase 1: Removal - try to remove vertices
            for v in list(current_S):
                if try_remove(v):
                    improved = True
                    break 
            
            if improved:
                continue

            # Phase 2: Swap - try to swap v in S with u not in S
            # This can enable further removals
            if not current_S:
                break
            
            v = random.choice(list(current_S))
            
            # Try multiple random u's
            candidates_u = [u for u in range(n) if u not in current_S]
            if not candidates_u:
                break
            
            random.shuffle(candidates_u)
            # Increase swap attempts based on pass count
            num_attempts = min(len(candidates_u), max(5, 20 - pass_count))
            
            for u in candidates_u[:num_attempts]:
                # Check if swap is valid
                valid = True
                for i in vertex_in_cliques[v]:
                    if counts[i] == 1:
                        if u not in cliques[i]:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    for i in vertex_in_cliques[v]:
                        counts[i] -= 1
                    current_S.remove(v)
                    
                    for i in vertex_in_cliques[u]:
                        counts[i] += 1
                    current_S.add(u)
                    
                    improved = True
                    break
        
        return current_S

    def two_opt_search(S, cliques, n, time_limit=5.0):
        """Enhanced local search with 2-opt moves: remove two vertices, add one or two"""
        start_time = time.time()
        
        counts = [0] * len(cliques)
        vertex_in_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_in_cliques[v].append(i)
        
        for v in S:
            for i in vertex_in_cliques[v]:
                counts[i] += 1
                
        current_S = set(S)
        
        # First do standard local search
        current_S = local_search(current_S, cliques, n, time_limit=min(2.0, time_limit * 0.4))
        
        # Recompute counts after local search
        counts = [0] * len(cliques)
        for v in current_S:
            for i in vertex_in_cliques[v]:
                counts[i] += 1
        
        # Now try 2-opt: remove two vertices from S, add vertices to compensate
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            if len(current_S) < 2:
                break
                
            # Try removing two vertices v1, v2
            v1_list = list(current_S)
            random.shuffle(v1_list)
            v1 = v1_list[0]
            
            remaining = [v for v in current_S if v != v1]
            if not remaining:
                break
            random.shuffle(remaining)
            v2 = remaining[0]
            
            # Check what cliques are only covered by v1 or v2
            critical_cliques = set()
            for i in vertex_in_cliques[v1]:
                if counts[i] == 1:
                    critical_cliques.add(i)
            for i in vertex_in_cliques[v2]:
                if counts[i] == 1:
                    critical_cliques.add(i)
            
            if not critical_cliques:
                # Both can be removed
                for i in vertex_in_cliques[v1]:
                    counts[i] -= 1
                current_S.remove(v1)
                for i in vertex_in_cliques[v2]:
                    counts[i] -= 1
                current_S.remove(v2)
                improved = True
                continue
            
            # Find vertices that can cover the critical cliques
            # A vertex u can cover a clique i if u is in clique i
            candidate_adds = set()
            for i in critical_cliques:
                for v in cliques[i]:
                    if v not in current_S:
                        candidate_adds.add(v)
            
            if not candidate_adds:
                continue
            
            # Try adding one or two vertices from candidates
            candidates = list(candidate_adds)
            random.shuffle(candidates)
            
            # Try adding one vertex
            for u in candidates[:10]:
                # Check if u covers all critical cliques
                covers_all = True
                for i in critical_cliques:
                    if u not in cliques[i]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Perform the move: remove v1, v2, add u
                    for i in vertex_in_cliques[v1]:
                        counts[i] -= 1
                    current_S.remove(v1)
                    for i in vertex_in_cliques[v2]:
                        counts[i] -= 1
                    current_S.remove(v2)
                    for i in vertex_in_cliques[u]:
                        counts[i] += 1
                    current_S.add(u)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adding two vertices
            if len(candidates) >= 2:
                for idx in range(min(5, len(candidates) - 1)):
                    u1 = candidates[idx]
                    u2 = candidates[idx + 1]
                    
                    # Check if {u1, u2} covers all critical cliques
                    covers_all = True
                    for i in critical_cliques:
                        if u1 not in cliques[i] and u2 not in cliques[i]:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Perform the move: remove v1, v2, add u1, u2
                        # This doesn't change size, but might enable further removals
                        for i in vertex_in_cliques[v1]:
                            counts[i] -= 1
                        current_S.remove(v1)
                        for i in vertex_in_cliques[v2]:
                            counts[i] -= 1
                        current_S.remove(v2)
                        for i in vertex_in_cliques[u1]:
                            counts[i] += 1
                        current_S.add(u1)
                        for i in vertex_in_cliques[u2]:
                            counts[i] += 1
                        current_S.add(u2)
                        improved = True
                        break
        
        # Final polish with standard local search
        if time.time() - start_time < time_limit:
            current_S = local_search(current_S, cliques, n, time_limit=min(2.0, time_limit - (time.time() - start_time)))
        
        return current_S

    def greedy_vertex_cover(n, edges, deadline):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        uncovered_edges = set()
        for u, v in edges:
            uncovered_edges.add((u, v))
        
        cover = set()
        
        while uncovered_edges:
            if time.time() > deadline:
                break
            
            # Pick vertex with max degree in remaining graph
            degs = {v: 0 for v in range(n)}
            for u, v in uncovered_edges:
                degs[u] += 1
                degs[v] += 1
            
            best_v = max(range(n), key=lambda x: degs[x])
            
            cover.add(best_v)
            
            new_uncovered = set()
            for u, v in uncovered_edges:
                if u != best_v and v != best_v:
                    new_uncovered.add((u, v))
            uncovered_edges = new_uncovered
        
        return cover

    def verify_hitting_set(S, cliques):
        for c in cliques:
            if not (S & c):
                return False
        return True

    def repair_hitting_set(S, cliques, n, deadline):
        S = set(S)
        # Check validity
        if verify_hitting_set(S, cliques):
            return S
        
        # Find uncovered cliques
        uncovered = [c for c in cliques if not (S & c)]
        
        # Greedily add vertices to cover uncovered cliques
        while uncovered:
            if time.time() > deadline:
                break
            
            # Count how many uncovered cliques each vertex covers
            cover_counts = {v: 0 for v in range(n)}
            for c in uncovered:
                for v in c:
                    cover_counts[v] += 1
            
            best_v = max(range(n), key=lambda x: cover_counts[x])
            S.add(best_v)
            
            # Remove cliques now covered by best_v
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
        
        return S

    def solve_instance(n, p, seed, deadline):
        edges = random_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        if not cliques:
            return []
        
        # Strategy 1: Hitting set approach
        initial_S = greedy_hitting_set(cliques, n)
        
        # Time budget allocation
        elapsed = time.time()
        remaining_time = deadline - elapsed
        
        # Use two_opt_search which includes local search
        search_time = min(8.0, max(1.0, remaining_time * 0.6))
        
        final_S = two_opt_search(initial_S, cliques, n, time_limit=search_time)
        
        # Verify and repair if needed
        if not verify_hitting_set(final_S, cliques):
            final_S = repair_hitting_set(final_S, cliques, n, deadline)
        
        # Strategy 2: Vertex cover approach (complement of MIS)
        # We can try to improve using vertex cover heuristics
        if time.time() < deadline - 2.0:
            vc = greedy_vertex_cover(n, edges, deadline)
            if len(vc) < len(final_S):
                # Check if vc is a valid hitting set
                if verify_hitting_set(vc, cliques):
                    final_S = vc
                    # Try local search on vc
                    remaining_time = deadline - time.time()
                    if remaining_time > 1.0:
                        final_S = local_search(final_S, cliques, n, time_limit=min(3.0, remaining_time * 0.5))
        
        # Final verification
        if not verify_hitting_set(final_S, cliques):
            final_S = repair_hitting_set(final_S, cliques, n, deadline)
        
        return sorted(final_S)

    results = []
    deadline = time.time() + 140  # 10s margin for 150s limit
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        result = solve_instance(n, p, seed, deadline)
        results.append(result)
    
    return results

# Ensure the function is named construct
# The code above defines construct.

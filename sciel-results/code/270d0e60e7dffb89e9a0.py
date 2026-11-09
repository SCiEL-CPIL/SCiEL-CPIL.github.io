import time
import random
from itertools import combinations
import sys
import numpy as np

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

        # Convert to list for iteration
        candidates_list = list(candidates)
        
        for v in candidates_list:
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

    def greedy_hitting_set(cliques, n, rng=None):
        if not cliques:
            return set()
        if rng is None:
            rng = random.Random()
        
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
            # Random tie-breaking for diversity
            for v in range(n):
                if v in S:
                    continue
                covered = sum(1 for i in vertex_to_cliques[v] if i in uncovered)
                if covered > max_covered or (covered == max_covered and rng.random() < 0.3):
                    max_covered = covered
                    best_v = v
            if best_v == -1:
                break
            S.add(best_v)
            for i in vertex_to_cliques[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
        
        return S

    def local_search(S, cliques, n, time_limit=5.0, seed=42):
        start_time = time.time()
        rng = random.Random(seed)
        
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

        improved = True
        iteration = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            iteration += 1
            
            # Phase 1: Removal - try all vertices
            for v in list(current_S):
                if try_remove(v):
                    improved = True
            
            if improved:
                continue

            # Phase 2: Swap - more aggressive
            if time.time() - start_time > time_limit * 0.8:
                break
                
            if not current_S:
                break
            
            # Try multiple swaps in one pass
            for _ in range(3):  # Try up to 3 swaps per iteration
                v = rng.choice(list(current_S))
                
                # Find candidates u that could replace v
                candidates_u = [u for u in range(n) if u not in current_S]
                if not candidates_u:
                    break
                
                # Score candidates by how many uncovered-by-v cliques they cover
                best_u = -1
                best_score = -1
                shuffled = candidates_u[:]
                rng.shuffle(shuffled)
                
                # Limit evaluation to keep time bounded
                eval_limit = min(len(shuffled), 50)
                for u in shuffled[:eval_limit]:
                    # Check if swap is valid and score it
                    valid = True
                    score = 0
                    for i in vertex_in_cliques[v]:
                        if counts[i] == 1:
                            if u not in cliques[i]:
                                valid = False
                                break
                            score += 1  # This clique would still be covered
                    if valid:
                        if score > best_score:
                            best_score = score
                            best_u = u
                
                if best_u != -1:
                    # Perform swap
                    for i in vertex_in_cliques[v]:
                        counts[i] -= 1
                    current_S.remove(v)
                    
                    for i in vertex_in_cliques[best_u]:
                        counts[i] += 1
                    current_S.add(best_u)
                    
                    improved = True
        
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
        
        # Multiple randomized restarts for better solution
        best_S = None
        best_size = float('inf')
        
        # Time budget for multiple attempts
        remaining_time = deadline - time.time()
        num_restarts = min(5, max(2, int(remaining_time / 5.0)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
                
            rng_seed = seed + restart * 1000
            initial_S = greedy_hitting_set(cliques, n, rng=random.Random(rng_seed))
            
            # Time budget for local search
            remaining_time = deadline - time.time()
            search_time = min(10.0, max(1.0, remaining_time * 0.4 / (num_restarts - restart)))
            
            candidate_S = local_search(initial_S, cliques, n, time_limit=search_time, seed=rng_seed)
            
            # Verify and repair if needed
            if not verify_hitting_set(candidate_S, cliques):
                candidate_S = repair_hitting_set(candidate_S, cliques, n, deadline)
            
            if len(candidate_S) < best_size:
                best_size = len(candidate_S)
                best_S = set(candidate_S)
        
        final_S = best_S if best_S else greedy_hitting_set(cliques, n)
        
        # Final local search polish with remaining time
        remaining_time = deadline - time.time()
        if remaining_time > 1.0:
            final_S = local_search(final_S, cliques, n, time_limit=min(8.0, remaining_time * 0.8), seed=seed + 9999)
            if not verify_hitting_set(final_S, cliques):
                final_S = repair_hitting_set(final_S, cliques, n, deadline)
        
        return sorted(final_S)

    results = []
    deadline = time.time() + 115  # 5s margin for 120s limit
    
    # Allocate time proportionally based on graph size
    # Larger graphs need more time for clique enumeration and search
    time_budgets = [30.0, 40.0, 45.0]  # Total 115s
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        if time.time() > deadline:
            # Fallback: return empty or minimal valid set
            # For safety, return all vertices
            results.append(list(range(n)))
            continue
        
        instance_deadline = min(deadline, time.time() + time_budgets[idx])
        result = solve_instance(n, p, seed, instance_deadline)
        results.append(result)
    
    return results

# Ensure the function is named construct
# The code above defines construct.

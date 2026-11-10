"""Structural edge-cover minimization via clique packing in split graphs."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def trivial_solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        return [list(e) for e in edges]

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        m = len(edges)
        edge_set = set(edges)

        if time.time() > DEADLINE - 5.0:
            return trivial_solve(n, k, p, seed)

        # Split graph structure:
        # - Clique vertices: 0 to k-1
        # - Independent set vertices: k to n-1
        # - Edges: all pairs within clique, plus cross edges with probability p
        
        # Key insight: We can form cliques by taking a subset of clique vertices S and 
        # a subset of independent vertices T such that:
        # 1. S is a clique (always true since all clique vertices are connected)
        # 2. T is an independent set (always true)
        # 3. Every vertex in T is connected to every vertex in S
        # 4. The clique S ∪ T contains at least 2 vertices
        # 5. Each edge is covered exactly once
        
        # Strategy: 
        # - For each independent vertex v, find the set of clique vertices it's connected to: N(v)
        # - We want to partition edges into cliques
        # - A clique can be: {u, v} where u in clique, v in independent (edge u-v)
        # - Or {u, v, w} where u,v in clique, w in independent (edges u-v, u-w, v-w)
        # - Or larger cliques
        
        # Better approach: Think of it as covering edges
        # Each clique covers:
        # - All edges within its clique-vertex part
        # - All edges between its clique-vertex part and its independent-vertex part
        
        # Let's use a greedy approach with backtracking:
        # 1. Start with all edges uncovered
        # 2. Repeatedly find a maximal clique that covers many uncovered edges
        # 3. Remove those edges from consideration
        # 4. Continue until all edges are covered
        
        # Build adjacency for independent vertices to clique vertices
        indep_to_clique = defaultdict(set)  # independent vertex -> set of clique neighbors
        for u, v in edges:
            if u < k and v >= k:
                indep_to_clique[v].add(u)
            elif v < k and u >= k:
                indep_to_clique[u].add(v)
        
        # Track which edges are covered
        covered = [False] * m
        
        # Find all edges and their indices
        clique_edges = []  # edges within clique
        cross_edges = []   # edges between clique and independent
        for idx, (u, v) in enumerate(edges):
            if u < k and v < k:
                clique_edges.append((idx, u, v))
            else:
                cross_edges.append((idx, u, v))
        
        cliques = []
        
        # Greedy: repeatedly pick a clique that covers the most uncovered edges
        # Try multiple random orderings and keep the best
        
        best_cliques = None
        best_count = float('inf')
        
        num_restarts = 15
        time_per_restart = (DEADLINE - time.time()) / num_restarts - 0.2
        if time_per_restart < 0.1:
            num_restarts = 1
            time_per_restart = max(0.1, DEADLINE - time.time() - 0.5)
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 2.0:
                break
            
            covered = [False] * m
            cliques = []
            
            # Random priority for independent vertices
            indep_vertices = list(range(k, n))
            random.Random(seed + restart * 7919).shuffle(indep_vertices)
            
            # For each independent vertex, we'll decide which clique vertices to include
            # Process independent vertices in random order
            for w in indep_vertices:
                if time.time() > DEADLINE - 1.0:
                    break
                
                # Find all uncovered cross edges incident to w
                neigh = indep_to_clique.get(w, set())
                if not neigh:
                    continue
                
                # Find which of these edges are still uncovered
                uncovered_neigh = set()
                for u in neigh:
                    for idx, a, b in cross_edges:
                        if (a == u and b == w) or (a == w and b == u):
                            if not covered[idx]:
                                uncovered_neigh.add(u)
                                break
                
                if not uncovered_neigh:
                    continue
                
                # Try to form a clique {w} ∪ S where S ⊆ uncovered_neigh
                # S must be a clique (always true since all clique vertices are connected)
                # We want to maximize the number of uncovered edges covered
                
                # The edges covered by clique {w} ∪ S are:
                # - All cross edges (u, w) for u in S
                # - All clique edges (u, v) for u, v in S
                
                # Greedily add clique vertices to S
                S = set()
                remaining = list(uncovered_neigh)
                random.Random(seed + restart * 104729 + w).shuffle(remaining)
                
                for u in remaining:
                    if time.time() > DEADLINE - 1.0:
                        break
                    # Check if adding u to S covers any new uncovered edges
                    new_cross = 1  # edge (u, w)
                    new_clique = 0
                    for v in S:
                        # Check if edge (u, v) is uncovered
                        for idx, a, b in clique_edges:
                            if (a == u and b == v) or (a == v and b == u):
                                if not covered[idx]:
                                    new_clique += 1
                                break
                    
                    if new_cross + new_clique > 0:
                        S.add(u)
                
                if len(S) >= 1:
                    # Form clique {w} ∪ S
                    clique_verts = list(S) + [w]
                    cliques.append(clique_verts)
                    
                    # Mark all edges in this clique as covered
                    for i in range(len(clique_verts)):
                        for j in range(i + 1, len(clique_verts)):
                            u, v = clique_verts[i], clique_verts[j]
                            for idx, a, b in edges:
                                if (a == u and b == v) or (a == v and b == u):
                                    covered[idx] = True
                                    break
        
        # After processing all independent vertices, handle remaining clique edges
        # Group remaining uncovered clique edges into cliques
        remaining_clique_edges = []
        for idx, u, v in clique_edges:
            if not covered[idx]:
                remaining_clique_edges.append((u, v))
        
        if remaining_clique_edges:
            # Find maximal cliques in the remaining edges (which is a subgraph of the clique)
            # Since it's a subgraph of a complete graph, we can greedily form cliques
            used = set()
            while remaining_clique_edges:
                # Pick an edge and try to extend it to a maximal clique
                u, v = remaining_clique_edges[0]
                clique = {u, v}
                # Try to add more vertices
                for idx, a, b in remaining_clique_edges[1:]:
                    if a not in clique and b not in clique:
                        # Check if a or b is connected to all in clique
                        if all((min(a, x), max(a, x)) in {(min(p, q), max(p, q)) for p, q in remaining_clique_edges} or 
                               (min(b, x), max(b, x)) in {(min(p, q), max(p, q)) for p, q in remaining_clique_edges} 
                               for x in clique):
                            # This is getting complex, let's just use pairs for now
                            pass
                
                # For simplicity, just use pairs for remaining clique edges
                cliques.append([u, v])
                covered = [False] * m  # This is wrong, need to track properly
        
        # This approach is getting complicated. Let me use a simpler but effective method:
        # Just use the triangle pairing approach but with better heuristics
        
        return None

    # Fallback to the original approach but improved
    def solve_original(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        m = len(edges)
        edge_set = set(edges)

        if time.time() > DEADLINE - 5.0:
            return trivial_solve(n, k, p, seed)

        # Build vertex-to-edges mapping
        by_vertex = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            by_vertex[u].append(idx)
            by_vertex[v].append(idx)
        
        # Build triangle adjacency
        tri_adj = [[] for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for s in (u1, v1):
                for j in by_vertex[s]:
                    if j <= i:
                        continue
                    u2, v2 = edges[j]
                    shared = set((u1, v1)) & set((u2, v2))
                    if len(shared) != 1:
                        continue
                    a = u1 if u1 != s else v1
                    b = u2 if u2 != s else v2
                    if a == b:
                        continue
                    if (min(a, b), max(a, b)) in edge_set:
                        tri_adj[i].append(j)
        
        # Find maximum matching using randomized greedy with multiple restarts
        best_pairs = []
        best_count = 0
        
        num_restarts = 25
        time_per_restart = (DEADLINE - time.time()) / num_restarts - 0.1
        if time_per_restart < 0.1:
            num_restarts = 1
            time_per_restart = max(0.1, DEADLINE - time.time() - 0.5)
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 2.0:
                break
            
            matched = [False] * m
            pairs = []
            order = list(range(m))
            random.Random(seed + restart * 1000).shuffle(order)
            
            for i in order:
                if matched[i]:
                    continue
                # Pick the j that gives the best "value" (e.g., highest degree)
                best_j = -1
                best_score = -1
                for j in tri_adj[i]:
                    if matched[j]:
                        continue
                    # Score based on remaining options
                    score = len([x for x in tri_adj[j] if not matched[x]])
                    if score > best_score:
                        best_score = score
                        best_j = j
                
                if best_j != -1:
                    matched[i] = True
                    matched[best_j] = True
                    u1, v1 = edges[i]
                    u2, v2 = edges[best_j]
                    shared = set((u1, v1)) & set((u2, v2))
                    s = next(iter(shared))
                    a = u1 if u1 != s else v1
                    b = u2 if u2 != s else v2
                    pairs.append((i, best_j, (s, a, b)))
            
            count = len(pairs)
            if count > best_count:
                best_count = count
                best_pairs = pairs[:]
        
        # Build partition
        matched = [False] * m
        for (i, j, tri) in best_pairs:
            matched[i] = True
            matched[j] = True
        
        cliques = []
        for (i, j, tri) in best_pairs:
            cliques.append(list(tri))
        for idx in range(m):
            if not matched[idx]:
                cliques.append(list(edges[idx]))
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        results.append(solve_original(*spec))

    out = []
    for lst in results:
        cl = []
        for c in lst:
            cl.append([int(x) for x in c])
        out.append(cl)
    return out

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END

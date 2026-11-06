"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from typing import List, Set, Tuple

def construct():
    """
    Construct a K4-free graph on 18 vertices that minimizes the size of the
    largest triangle-free induced subgraph.
    
    Strategy:
    1. Start with a base structure that is K4-free.
    2. Use local search (edge flips) to minimize the maximum triangle-free set size.
    3. Use a deadline guard to return the best result found within the time limit.
    """
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Vertices 0..17
    n = 18
    
    # Helper to check if a set of vertices induces a triangle
    def has_triangle(vertices: Set[int], edges: Set[Tuple[int, int]]) -> bool:
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                for k in range(j + 1, len(verts)):
                    u, v, w = verts[i], verts[j], verts[k]
                    if (min(u, v), max(u, v)) in edges and \
                       (min(v, w), max(v, w)) in edges and \
                       (min(u, w), max(u, w)) in edges:
                        return True
        return False
    
    # Helper to check if a set of vertices induces a K4
    def has_k4(vertices: Set[int], edges: Set[Tuple[int, int]]) -> bool:
        verts = list(vertices)
        if len(verts) < 4:
            return False
        for quad in itertools.combinations(verts, 4):
            u, v, w, x = quad
            pairs = [(min(u,v),max(u,v)), (min(u,w),max(u,w)), (min(u,x),max(u,x)),
                     (min(v,w),max(v,w)), (min(v,x),max(v,x)), (min(w,x),max(w,x))]
            if all(p in edges for p in pairs):
                return True
        return False
    
    # Helper to compute the size of the largest triangle-free induced subgraph
    # Uses a greedy + local search approach for speed
    def max_triangle_free_size(edges: Set[Tuple[int, int]], n: int) -> int:
        # We want to find the largest subset S such that G[S] is triangle-free.
        # This is equivalent to: find the largest S such that no triangle is fully contained in S.
        # Alternatively, we can use the complement: find the smallest set T such that 
        # every triangle has at least one vertex in T (vertex cover of triangles).
        # Then the answer is n - min_vertex_cover_of_triangles.
        
        # Let's find all triangles in the graph
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges:
                    continue
                for w in range(v + 1, n):
                    if (v, w) in edges and (u, w) in edges:
                        triangles.append((u, v, w))
        
        if not triangles:
            return n
        
        # Find minimum vertex cover of the hypergraph of triangles
        # Use a greedy heuristic + local search
        # Greedy: repeatedly pick the vertex in the most uncovered triangles
        covered = [False] * len(triangles)
        chosen = set()
        
        # Precompute for each vertex, which triangles it covers
        vert_to_tris = {v: [] for v in range(n)}
        for i, tri in enumerate(triangles):
            for v in tri:
                vert_to_tris[v].append(i)
        
        # Greedy set cover
        uncovered = len(triangles)
        while uncovered > 0:
            best_v = -1
            best_gain = 0
            for v in range(n):
                if v in chosen:
                    continue
                gain = sum(1 for i in vert_to_tris[v] if not covered[i])
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1 or best_gain == 0:
                break
            chosen.add(best_v)
            for i in vert_to_tris[best_v]:
                if not covered[i]:
                    covered[i] = True
                    uncovered -= 1
        
        # Local search: try to swap vertices in the cover
        improved = True
        while improved:
            improved = False
            for v_in in list(chosen):
                # Try removing v_in
                new_covered = covered[:]
                for i in vert_to_tris[v_in]:
                    # Check if triangle i is still covered by another chosen vertex
                    still_covered = False
                    for v_other in chosen:
                        if v_other == v_in:
                            continue
                        if i in vert_to_tris[v_other]:
                            still_covered = True
                            break
                    if not still_covered:
                        new_covered[i] = False
                
                if any(not c for c in new_covered):
                    # Can't remove v_in directly
                    continue
                
                # Try adding a different vertex to see if we can reduce size
                # Actually, we just removed v_in, so size decreased by 1. 
                # But we need to check if all triangles are still covered.
                # If yes, we improved.
                if all(new_covered):
                    chosen.discard(v_in)
                    covered = new_covered
                    improved = True
                    break
        
        # Now try to add vertices to see if we can reduce further? No, we want min cover.
        # The above greedy + local removal should give a decent cover.
        
        return n - len(chosen)
    
    # Initialize with a random K4-free graph
    # Start with a complete tripartite graph as a good initial guess
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a, b), max(a, b)))
    
    best_edges = edges.copy()
    best_score = max_triangle_free_size(edges, n)
    
    # Local search: try edge flips
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Shuffle for randomization
    random.seed(42)
    random.shuffle(all_pairs)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Try each edge flip
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            
            edge_key = (u, v)
            
            # Try adding the edge
            if edge_key not in edges:
                new_edges = edges | {edge_key}
                # Check K4-free
                # Only need to check K4s containing u and v
                is_k4_free = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (min(u, w), max(u, w)) in new_edges and (min(v, w), max(v, w)) in new_edges:
                        for x in range(w + 1, n):
                            if x == u or x == v or x == w:
                                continue
                            if (min(u, x), max(u, x)) in new_edges and \
                               (min(v, x), max(v, x)) in new_edges and \
                               (min(w, x), max(w, x)) in new_edges:
                                is_k4_free = False
                                break
                    if not is_k4_free:
                        break
                if not is_k4_free:
                    continue
                
                score = max_triangle_free_size(new_edges, n)
                if score < best_score:
                    best_score = score
                    best_edges = new_edges
                    edges = new_edges.copy()
                    # Restart local search
                    break
            else:
                # Try removing the edge
                new_edges = edges - {edge_key}
                score = max_triangle_free_size(new_edges, n)
                if score < best_score:
                    best_score = score
                    best_edges = new_edges
                    edges = new_edges.copy()
                    break
        
        # If we broke out early due to improvement, continue
        if time.time() > deadline:
            break
        
        # Occasionally randomize
        if iterations % 10 == 0:
            # Try a random restart with a different initial structure
            # Use a random K4-free graph
            # Simple approach: random graph with low edge density
            if random.random() < 0.1:
                edges = set()
                for u, v in all_pairs:
                    if random.random() < 0.3:
                        edges.add((u, v))
                # Check K4-free
                if not any(has_k4(set(range(n)), edges) for _ in [0]):
                    score = max_triangle_free_size(edges, n)
                    if score < best_score:
                        best_score = score
                        best_edges = edges.copy()
    
    # Return the best edges found
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END

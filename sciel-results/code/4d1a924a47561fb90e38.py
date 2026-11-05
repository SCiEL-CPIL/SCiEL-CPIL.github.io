import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    # Build adjacency matrix for polarity graph (C4-free, 90 edges)
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                A[i, j] = 1
                A[j, i] = 1

    m = A.sum() // 2  # should be 90
    assert m == 90, f"Expected 90 edges, got {m}"

    # Find the edge that adds fewest C4s to the polarity graph
    # C4s added by edge (u,v) = number of common neighbors of u and v
    # (since polarity graph is C4-free, no existing C4s are destroyed)
    best_edge = None
    best_common = n + 1
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 0:
                common = np.sum(A[u] & A[v])
                if common < best_common:
                    best_common = common
                    best_edge = (u, v)

    A[best_edge[0], best_edge[1]] = 1
    A[best_edge[1], best_edge[0]] = 1
    m += 1
    assert m == 91

    # Compute total C4 count: sum over all edges (u,v) of C(common_neighbors(u,v), 2)
    # But this double-counts each C4 (each C4 has 4 edges, each edge sees the C4)
    # Actually: number of C4s = (1/4) * sum over edges (u,v) of number of length-2 paths from u to v not using edge (u,v)
    # More precisely: for each pair of non-adjacent vertices (u,v), count common neighbors
    # Each C4 is counted once per pair of opposite vertices
    # Total C4s = (1/2) * sum over all pairs (u,v) of C(common(u,v), 2) where u,v non-adjacent
    # Wait, let me think again.
    # A C4 is u-a-v-b-u. The pairs of opposite vertices are (u,v) and (a,b).
    # For pair (u,v): common neighbors include a and b, so C(common(u,v),2) counts this C4.
    # For pair (a,b): common neighbors include u and v, so C(common(a,b),2) counts this C4.
    # So total C4s = (1/2) * sum over all non-adjacent pairs (u,v) of C(common(u,v), 2)
    # But we need to be careful: if u and v are adjacent, common neighbors form triangles, not C4s directly.
    # Actually for adjacent u,v, a common neighbor w gives triangle u-v-w, not a C4.
    # So we only sum over non-adjacent pairs.

    def count_total_c4(A):
        """Count total number of 4-cycles in graph with adjacency matrix A."""
        total = 0
        # For each pair of non-adjacent vertices, count C(common_neighbors, 2)
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 0:
                    common = np.sum(A[u] & A[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2  # Each C4 counted twice (once for each pair of opposite vertices)

    total_c4 = count_total_c4(A)

    # Local search: try edge swaps to reduce C4 count
    # For a swap: remove edge (ru,rv), add edge (au,av)
    # Delta in C4 count can be computed efficiently
    
    def delta_c4_for_swap(A, ru, rv, au, av):
        """Compute change in total C4 when removing (ru,rv) and adding (au,av)."""
        # We need to compute: new_total - old_total
        # This is complex, so let's use a direct approach:
        # Temporarily make the change, compute delta, revert.
        
        # Save state
        orig_ru_rv = A[ru, rv]
        orig_av_au = A[au, av]
        
        # Make change
        A[ru, rv] = 0
        A[rv, ru] = 0
        A[au, av] = 1
        A[av, au] = 1
        
        new_total = count_total_c4(A)
        
        # Revert
        A[ru, rv] = orig_ru_rv
        A[rv, ru] = orig_ru_rv
        A[au, av] = orig_av_au
        A[av, au] = orig_av_au
        
        return new_total - total_c4

    # Faster delta computation using incremental updates
    def fast_delta_c4(A, ru, rv, au, av):
        """
        Compute delta in C4 count when removing edge (ru,rv) and adding edge (au,av).
        
        A C4 is affected if it contains edge (ru,rv) or edge (au,av) (or both).
        
        C4s destroyed: those containing edge (ru,rv) in current graph.
        C4s created: those containing edge (au,av) in the graph after removal of (ru,rv).
        
        For edge (x,y) in graph G, C4s containing it = number of length-2 paths from x to y 
        using vertices not x or y, where the two internal vertices are adjacent.
        = sum over pairs (a,b) with x~a, a~b, b~y, a!=y, b!=x, a!=b
        = for each common neighbor structure... 
        
        Actually: C4s through edge (x,y) = number of pairs (a,b) such that:
        x~a, a~b, b~y, and {x,y,a,b} are all distinct.
        This equals: for each a in N(x) \ {y}, for each b in N(y) \ {x}, check if a~b and a!=b.
        = sum over a in N(x)\{y} of |N(a) ∩ N(y) \ {x}|
        
        But we also need a != b, which is handled by the set operations.
        """
        # C4s destroyed by removing (ru, rv)
        destroyed = 0
        for a in range(n):
            if a == ru or a == rv:
                continue
            if A[ru, a] == 1:
                for b in range(n):
                    if b == ru or b == rv or b == a:
                        continue
                    if A[rv, b] == 1 and A[a, b] == 1:
                        destroyed += 1
        
        # C4s created by adding (au, av) in graph without (ru, rv)
        # Temporarily remove (ru, rv)
        A[ru, rv] = 0
        A[rv, ru] = 0
        
        created = 0
        for a in range(n):
            if a == au or a == av:
                continue
            if A[au, a] == 1:
                for b in range(n):
                    if b == au or b == av or b == a:
                        continue
                    if A[av, b] == 1 and A[a, b] == 1:
                        created += 1
        
        # Restore (ru, rv)
        A[ru, rv] = 1
        A[rv, ru] = 1
        
        return created - destroyed

    # Get list of current edges and non-edges
    def get_edges(A):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 1:
                    edges.append((u, v))
        return edges

    def get_nonedges(A):
        nonedges = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 0:
                    nonedges.append((u, v))
        return nonedges

    # Precompute edge lists
    edges = get_edges(A)
    nonedges = get_nonedges(A)
    edge_set = set(edges)
    
    best_A = A.copy()
    best_total = total_c4
    best_edges = list(edges)

    # Local search with time budget
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Get current edges and nonedges
        edges = get_edges(A)
        nonedges = get_nonedges(A)
        edge_set = set(edges)
        
        # Score each edge by how many C4s it's in (higher = worse to keep)
        # Score each non-edge by how many C4s it would create (lower = better to add)
        
        # For efficiency, compute C4 contribution of each edge
        # C4s through edge (u,v) = number of paths u-a-b-v with a~b, all distinct
        edge_scores = []
        for (u, v) in edges:
            count = 0
            for a in range(n):
                if a == u or a == v:
                    continue
                if A[u, a] == 1:
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if A[v, b] == 1 and A[a, b] == 1:
                            count += 1
            edge_scores.append((count, (u, v)))
        
        nonedge_scores = []
        for (u, v) in nonedges:
            count = 0
            for a in range(n):
                if a == u or a == v:
                    continue
                if A[u, a] == 1:
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if A[v, b] == 1 and A[a, b] == 1:
                            count += 1
            nonedge_scores.append((count, (u, v)))
        
        # Sort: remove edges with highest C4 count, add non-edges with lowest
        edge_scores.sort(key=lambda x: -x[0])  # descending by C4 count
        nonedge_scores.sort(key=lambda x: x[0])  # ascending by C4 count
        
        # Try top swaps
        for (ec, (ru, rv)) in edge_scores[:30]:
            if improved:
                break
            for (ac, (au, av)) in nonedge_scores[:30]:
                delta = fast_delta_c4(A, ru, rv, au, av)
                if delta < 0:
                    # Commit swap
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                    total_c4 += delta
                    improved = True
                    break
        
        if not improved:
            # Try random swaps to escape local minima
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                re = edges[np.random.randint(len(edges))]
                ae = nonedges[np.random.randint(len(nonedges))]
                ru, rv = re
                au, av = ae
                delta = fast_delta_c4(A, ru, rv, au, av)
                # Simulated annealing: accept with probability exp(-delta/T)
                T = max(1.0, 10.0 - (no_improve_count / 100.0))
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                    total_c4 += delta
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            if total_c4 < best_total:
                best_total = total_c4
                best_A = A.copy()
                best_edges = get_edges(A)
        else:
            no_improve_count += 1
            if no_improve_count > 500:
                # Restart with random perturbation
                for _ in range(5):
                    re = edges[np.random.randint(len(edges))]
                    ae = nonedges[np.random.randint(len(nonedges))]
                    ru, rv = re
                    au, av = ae
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                total_c4 = count_total_c4(A)
                no_improve_count = 0

    # Return best solution
    final_edges = get_edges(best_A)
    assert len(final_edges) == 91
    return [[e[0], e[1]] for e in final_edges]

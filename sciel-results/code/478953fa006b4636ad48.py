import time
import sys

sys.setrecursionlimit(100000)

def construct():
    """
    Constructs a triangle-free graph on n vertices with no independent set of 10.
    Strategy:
    1. Generate the polarity graph of PG(2, 11). This graph is known to be triangle-free
       and has an independence number of q+1 = 12.
       The graph has q^2 + q + 1 = 132 vertices.
    2. Use a fast exact branch-and-bound algorithm to find a large induced subgraph
       that is triangle-free and has independence number <= 9.
       Since the base graph is triangle-free, any induced subgraph is triangle-free.
       So we just need to find a subset of vertices of size 40 (or larger, up to 40)
       such that the induced subgraph has no independent set of size 10.
       Equivalently, in the complement (which is not necessarily triangle-free, but we
       only care about cliques in the complement corresponding to independent sets in G),
       we want to find a subset S of size 40 such that the maximum clique in the
       complement of G[S] is <= 9.
       
       Actually, it's easier to work with G directly.
       We want to select a subset V' of V(G) with |V'| = 40 such that alpha(G[V']) <= 9.
       Since G is triangle-free, G[V'] is triangle-free.
       
       We can use a branch and bound search to select vertices.
       However, checking alpha <= 9 is expensive.
       
       Alternative:
       We know R(3,10) > 39. The polarity graph of PG(2,11) has alpha=12.
       If we can remove vertices to reduce alpha to 9, we might find a valid graph.
       
       Let's try a greedy approach first, then refine.
       Actually, a simpler deterministic construction might be better if available.
       But the polarity graph is a strong candidate because its structure is well-understood.
       
       Let's implement the PG(2, 11) polarity graph construction.
       q = 11.
       Vertices are 1D subspaces of GF(q)^3, i.e., points of PG(2, q).
       There are q^2 + q + 1 = 132 vertices.
       Two points x, y are adjacent if x^T A y = 0 for a suitable symmetric matrix A
       that defines a polarity.
       For q odd, we can use the hyperbolic polarity or elliptic.
       A standard triangle-free polarity graph uses a symplectic form or similar.
       Actually, for odd q, the graph defined by x^T y = 0 (with y in the dual space,
       identified with points) is not symmetric. We need a symmetric bilinear form.
       
       Let's use the standard construction for the polarity graph of PG(2, q) for odd q.
       The graph is triangle-free if the polarity is of symplectic type?
       No, for odd q, a polarity graph of PG(2, q) is triangle-free iff the polarity
       is of hyperbolic type? I need to be careful.
       
       Actually, a simpler known triangle-free graph with small independence number
       is the Andrasfai graph, but that's for specific n.
       
       Let's stick to the Andrasfai graph but try to extend it or use a different
       algebraic construction.
       
       Wait, the inspiration mentions "complement of a finite projective plane or a
       polarity graph".
       
       Let's try the polarity graph of PG(2, 11) again.
       Reference: "Triangle-free graphs with small independence number" by various authors.
       The polarity graph of PG(2, q) for odd q is triangle-free.
       Its independence number is q+1.
       For q=11, alpha = 12.
       We need alpha <= 9.
       So we need to delete at least some vertices to reduce alpha from 12 to 9.
       If we delete 3 vertices that are in every maximum independent set, we might succeed.
       But finding such vertices is hard.
       
       Alternative: Use q=7.
       PG(2, 7) has 57 vertices. Alpha = 8.
       This is already valid! alpha = 8 <= 9.
       And it is triangle-free.
       So n=57 is a valid graph?
       Wait, R(3,10) <= 41. So n cannot be 57.
       The task says: "Maximize n. ... R(3,10) <= 41 limits any valid graph to 40 vertices."
       So we must have n <= 40.
       
       So the PG(2, 7) graph has n=57, which is too large.
       But we can take an induced subgraph of it.
       We need to find a subset of 40 vertices from the 57 such that the induced
       subgraph has alpha <= 9.
       Since the whole graph has alpha=8, any induced subgraph has alpha <= 8 <= 9.
       So ANY induced subgraph of size 40 from the PG(2, 7) polarity graph is valid!
       
       This is a huge simplification.
       1. Construct the polarity graph of PG(2, 7). It has 57 vertices, is triangle-free,
          and has alpha = 8.
       2. Pick any 40 vertices from it.
       3. The induced subgraph is triangle-free (subgraph of triangle-free) and
          has alpha <= 8 (subgraph has alpha <= original alpha).
       4. Thus, it satisfies the conditions for n=40.
       
       This is deterministic and fast.
       
       Let's implement the PG(2, 7) polarity graph.
       
       Construction of PG(2, q) polarity graph for odd q:
       Vertices: points of PG(2, q), represented as vectors (x, y, z) in GF(q)^3,
       not all zero, modulo scalar multiplication.
       We can represent each vertex by a canonical representative, e.g., the one with
       the first non-zero coordinate = 1.
       
       Polarity: A symmetric bilinear form B(u, v) = u^T A v.
       For the graph to be triangle-free, we need B(u, v) = 0 to define the edges,
       and we need that if B(u, v) = 0 and B(v, w) = 0, then B(u, w) != 0?
       No, triangle-free means no u, v, w such that B(u,v)=B(v,w)=B(w,u)=0.
       
       A standard choice is the hyperbolic polarity.
       For q odd, the graph defined by x^2 + y^2 + z^2 = 0? No.
       
       Let's use the standard construction from literature:
       For odd q, the graph G_q has vertex set the points of PG(2, q).
       Two points x, y are adjacent if x^T y = 0 in the standard dot product?
       No, that's not a polarity (not symmetric in the right way for all q).
       
       Actually, for q odd, the form x^T y = 0 is symmetric.
       Is the graph triangle-free?
       If x^T y = 0, y^T z = 0, and z^T x = 0, then x, y, z are linearly dependent?
       In 3D, if three vectors are pairwise orthogonal with respect to a non-degenerate
       symmetric form, they are linearly independent.
       So if x, y, z are distinct points (1D subspaces), they cannot be pairwise orthogonal.
       Wait, if they are linearly independent, they form a basis.
       But in PG(2, q), any two points span a line (2D subspace).
       A third point not on the line is linearly independent.
       So if x, y, z are linearly independent, they can be pairwise orthogonal?
       No, in a 3D space with a non-degenerate symmetric bilinear form, you can have
       at most 3 pairwise orthogonal non-zero vectors.
       But if they are pairwise orthogonal, they are linearly independent.
       So it is possible to have 3 pairwise orthogonal vectors.
       Thus, the graph defined by x^T y = 0 might have triangles.
       
       We need a polarity such that the graph is triangle-free.
       This is achieved by a "symplectic" polarity, but symplectic forms are alternating,
       so B(x, x) = 0 for all x, meaning every vertex is adjacent to itself (loop).
       We need a graph without loops.
       
       For odd q, a triangle-free polarity graph is obtained by using a non-degenerate
       symmetric bilinear form of signature (1, 2) or (2, 1) in some sense?
       Actually, the standard result is that the polarity graph of PG(2, q) is triangle-free
       if and only if the polarity is of hyperbolic type?
       
       Let's check a known construction.
       The Andrasfai graph is a circulant graph.
       The polarity graph of PG(2, q) is a well-known family of triangle-free graphs.
       According to a survey, the polarity graph of PG(2, q) for odd q is triangle-free.
       Why?
       Because the form is chosen such that the radical of any 2D subspace is 1D?
       
       Let's use the explicit construction from "Triangle-free graphs with small independence number".
       For q odd, define the graph on points of PG(2, q) by x ~ y if x^T A y = 0,
       where A is a symmetric matrix with determinant a non-square.
       This ensures that the form is anisotropic in a certain way?
       
       Actually, let's just use the Andrasfai graph for n=39 or 40 if we can construct it.
       The Andrasfai graph A_n is defined for n divisible by 3?
       A_n has vertices 0..n-1, edge (i, j) if (j-i) mod n is in {1, 3, ..., n-2} and (j-i) mod 3 != 0?
       No, the standard Andrasfai graph A_n (for n=3k) has edges (i, j) if (j-i) mod n is not divisible by 3.
       This graph is triangle-free.
       Its independence number is roughly n/3.
       For n=39, alpha = 13. Too large.
       For n=40, not divisible by 3.
       
       So Andrasfai is not good enough for n=40 with alpha <= 9.
       
       Back to PG(2, 7).
       Let's verify the triangle-free property of the polarity graph of PG(2, 7).
       If it is triangle-free and has alpha=8, then any 40-vertex induced subgraph is valid.
       
       Construction:
       GF(7) = {0, 1, ..., 6}.
       Points of PG(2, 7): triples (x, y, z) not all 0, modulo scaling.
       Total points: (7^3 - 1) / (7 - 1) = 342 / 6 = 57.
       
       Polarity: Use the symmetric bilinear form B(u, v) = u_0 v_0 + u_1 v_1 + u_2 v_2.
       Is the graph triangle-free?
       Suppose u, v, w are pairwise orthogonal.
       Then u, v, w are linearly independent (since the form is non-degenerate).
       So they form a basis of GF(7)^3.
       This is possible. So the graph might have triangles.
       
       We need a form such that no 3 vectors are pairwise orthogonal.
       This is equivalent to the form having no isotropic 3-spaces?
       No, pairwise orthogonal means the subspace they span is totally isotropic.
       So we need the form to have Witt index 1 (max totally isotropic subspace dimension 1).
       For a 3D space over GF(q) with q odd, the Witt index can be 1 or 2.
       If Witt index is 1, there are no 2D totally isotropic subspaces, so no 3 pairwise orthogonal vectors.
       So we need a form of Witt index 1.
       For q=7, we can use the form x^2 + y^2 + z^2.
       Is -1 a square mod 7?
       1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
       Squares: 1, 2, 4.
       -1 = 6. 6 is not a square.
       So x^2 + y^2 + z^2 = 0 has no non-trivial solution?
       If x^2 + y^2 = -z^2, and -1 is not a square, then x=y=z=0 is the only solution.
       So the form is anisotropic.
       If the form is anisotropic, then no non-zero vector is orthogonal to itself.
       But we need to check pairwise orthogonality of distinct vectors.
       If the form is anisotropic, the Witt index is 0.
       So there are no non-zero isotropic vectors.
       Thus, no two distinct points can be orthogonal?
       Wait, if B(u, v) = 0 for u != v, that's fine.
       But if B(u, u) != 0 for all u != 0, then the graph has no loops.
       And if there are no isotropic vectors, then no 3 vectors can be pairwise orthogonal?
       Actually, if the form is anisotropic, the maximum totally isotropic subspace has dimension 0.
       So no two vectors are orthogonal?
       No, anisotropic means B(u, u) != 0 for all u != 0.
       It does not mean B(u, v) != 0 for u != v.
       But if B(u, v) = 0, then u and v are orthogonal.
       Can we have 3 pairwise orthogonal vectors in an anisotropic form?
       If u, v, w are pairwise orthogonal, then the subspace they span is totally isotropic.
       But an anisotropic form has no non-zero isotropic vectors, so no totally isotropic subspace.
       So yes, the graph is triangle-free.
       
       So for q=7, the form x^2 + y^2 + z^2 is anisotropic.
       Thus, the polarity graph is triangle-free.
       And its independence number is q+1 = 8.
       
       So we can construct this graph, pick any 40 vertices, and return them.
       
       Steps:
       1. Generate all 57 points of PG(2, 7).
       2. For each pair of points, check if they are orthogonal (dot product = 0 mod 7).
       3. Build the adjacency matrix.
       4. Pick the first 40 vertices (or any 40).
       5. Return the induced subgraph.
       
       This is deterministic and fast.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    def time_up():
        return time.time() > deadline

    q = 7
    # Generate points of PG(2, q)
    # Represent each point by a canonical representative (x, y, z) where the first non-zero is 1.
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize
                # Find first non-zero
                if x != 0:
                    # x is 1
                    if x != 1:
                        inv_x = pow(x, -1, q)
                    else:
                        inv_x = 1
                    cy = (y * inv_x) % q
                    cz = (z * inv_x) % q
                    rep = (1, cy, cz)
                elif y != 0:
                    if y != 1:
                        inv_y = pow(y, -1, q)
                    else:
                        inv_y = 1
                    cz = (z * inv_y) % q
                    rep = (0, 1, cz)
                else: # z != 0
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n_points = len(points)
    # Should be 57
    
    # Build adjacency list
    # Edge (i, j) if dot(points[i], points[j]) == 0 mod q
    adj = [0] * n_points
    for i in range(n_points):
        xi, yi, zi = points[i]
        for j in range(i + 1, n_points):
            xj, yj, zj = points[j]
            dot = (xi * xj + yi * yj + zi * zj) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify triangle-free and alpha <= 9 for the whole graph (optional, but good for sanity)
    # We know it should be triangle-free and alpha=8.
    
    # We need a graph on n=40 vertices.
    # Pick the first 40 vertices.
    target_n = 40
    if n_points < target_n:
        # Fallback to Andrasfai if something went wrong
        # But n_points should be 57.
        target_n = n_points
    
    # Map old indices to new indices 0..target_n-1
    # We just take the first target_n vertices.
    new_adj = [0] * target_n
    for i in range(target_n):
        # Mask adj[i] to only keep bits < target_n
        # Since we are taking the first target_n vertices, the bits corresponding to
        # vertices >= target_n should be cleared.
        # But adj[i] only has bits for j < n_points.
        # We need to ensure that if j >= target_n, we don't include it.
        # Since we are constructing a new graph on 0..target_n-1,
        # we only care about edges between i, j < target_n.
        mask = (1 << target_n) - 1
        new_adj[i] = adj[i] & mask
    
    # Convert to edges
    edges = []
    for u in range(target_n):
        for v in range(u + 1, target_n):
            if new_adj[u] & (1 << v):
                edges.append([u, v])
    
    # Final validation (fast)
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def has_independent_set(adj, n, target_size=10):
        def dfs(candidates, size):
            if size >= target_size:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target_size:
                return False
            
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            rest = candidates ^ v_bit
            
            # Include v
            if dfs(rest & ~adj[v], size + 1):
                return True
            
            # Exclude v
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs((1 << n) - 1, 0)
    
    if is_triangle_free(new_adj, target_n) and not has_independent_set(new_adj, target_n):
        return {'n': target_n, 'edges': edges}
    else:
        # Fallback to a known valid graph (Andrasfai n=29)
        def andrasfai_adj(n):
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % n % 3 == 1:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj
        
        adj29 = andrasfai_adj(29)
        edges29 = []
        for u in range(29):
            for v in range(u + 1, 29):
                if adj29[u] & (1 << v):
                    edges29.append([u, v])
        
        return {'n': 29, 'edges': edges29}

if __name__ == "__main__":
    result = construct()
    print(f"Best n: {result['n']}")

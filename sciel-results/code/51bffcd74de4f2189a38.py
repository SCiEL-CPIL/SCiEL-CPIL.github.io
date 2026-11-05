import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 31
    target_edges = 91
    
    # --- 1. Structured Initialization: Polarity Graph of PG(2,5) ---
    
    def generate_polarity_graph():
        """
        Generates the polarity graph of the projective plane of order q=5.
        Vertices: Points of PG(2, q). There are q^2 + q + 1 = 31 points.
        Edges: Two points are connected if they are orthogonal w.r.t the polarity.
        
        For PG(2, q) with q odd, the standard polarity is given by the symmetric matrix:
        A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]] (in homogeneous coordinates)
        Actually, a standard involution (polarity) for PG(2, q) is:
        (x, y, z) -> (y, x, z) is a reflection, not necessarily a polarity with the right C4 properties.
        
        Let's use the standard construction for the Erdos-Renyi graph or similar.
        The graph G_{q} has vertices (x, y) in F_q^2 plus a point at infinity.
        Actually, simpler: The polarity graph of PG(2, q) for q=5 has 31 vertices.
        The number of edges is roughly n*q/2 + n/2.
        For q=5, n=31.
        Degree of each vertex is q+1 = 6? No, in a polarity graph of PG(2,q),
        the degree of a point is q+1.
        Total edges = 31 * 6 / 2 = 93.
        
        We need 91 edges. So we start with 93 edges and remove 2.
        """
        q = 5
        # Points of PG(2, q) can be represented as pairs (x, y) in F_q^2 and points at infinity (1, m) for m in F_q, and (0, 1).
        # Total points: q^2 + q + 1 = 25 + 5 + 1 = 31.
        
        # Let's map points to indices 0..30
        # 0..24: affine points (x, y)
        # 25..29: points at infinity (1, m) for m=0..4
        # 30: point at infinity (0, 1)
        
        def get_point(idx):
            if idx < 25:
                x = idx % 5
                y = idx // 5
                return (x, y, 1)
            elif idx < 30:
                m = idx - 25
                return (1, m, 0)
            else:
                return (0, 1, 0)
        
        def dot(p1, p2):
            return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
        
        # Polarity matrix for PG(2,5).
        # A common polarity is the "orthogonal polarity" defined by the conic.
        # For q=5, the equation x^2 + y^2 + z^2 = 0 defines a conic.
        # Two points P, Q are orthogonal if P^T A Q = 0 where A is the matrix of the form.
        # A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]? No, that's not a polarity with the right properties for C4-free.
        # The standard polarity graph for C4 minimization uses the form x y + z^2 = 0 or similar.
        # Let's use the matrix A = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]?
        
        # Actually, the graph derived from the polarity of PG(2, q) is known to be C4-free.
        # Let's construct it using the standard "Erdos-Renyi" polarity:
        # Vertices are (x,y) in F_q^2 and points at infinity.
        # Two affine points (x1, y1) and (x2, y2) are adjacent if (x1-x2)^2 + (y1-y2)^2 = 0? No.
        
        # Let's stick to the definition:
        # A polarity is a symmetric bilinear form.
        # For q=5, let A = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]?
        
        # Simpler approach:
        # The graph with vertices F_q^2 U {infinity} where (x,y) ~ (x',y') if (x-x')^2 + (y-y')^2 = 0 is not quite right.
        
        # Let's use the explicit adjacency for the polarity graph of PG(2,5).
        # It is a (31, 6)-regular graph.
        # We can generate it by picking a non-degenerate symmetric matrix A.
        # Let A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]? This is the standard dot product.
        # In PG(2,5), the number of points orthogonal to a non-isotropic point is q=5.
        # If the point is isotropic (on the conic), it is orthogonal to itself and q-1 others?
        # For q=5, -1 is not a square (5 mod 4 = 1, so -1 is not a square in F_5? 1^2=1, 2^2=4, 3^2=4, 4^2=1. Yes, -1=4 is a square. So the conic is non-degenerate).
        
        # If we use the standard dot product, the graph is not regular.
        # To get a regular graph, we need a polarity where all points are non-isotropic or handle isotropic points carefully.
        # Actually, the "Erdos-Renyi graph" $ER_q$ is defined on $F_q^2$ plus a point at infinity.
        # Edges:
        # 1. (x,y) connected to (x',y') if (x-x')^2 + (y-y')^2 = 0.
        # 2. (x,y) connected to infinity if x^2 + y^2 = 0.
        
        # For q=5, x^2+y^2=0 has solutions. (1,2) -> 1+4=0. (1,3) -> 1+9=10=0.
        # So this graph exists.
        # Let's build this graph.
        
        edges = set()
        
        # Affine-Affine edges
        for i in range(25):
            x1, y1 = i % 5, i // 5
            for j in range(i+1, 25):
                x2, y2 = j % 5, j // 5
                dx = (x1 - x2) % 5
                dy = (y1 - y2) % 5
                if (dx*dx + dy*dy) % 5 == 0:
                    edges.add((i, j))
        
        # Affine-Infinity edges
        # Infinity is index 25.
        # (x,y) connected to inf if x^2 + y^2 = 0.
        for i in range(25):
            x, y = i % 5, i // 5
            if (x*x + y*y) % 5 == 0:
                edges.add((i, 25))
                
        # Infinity-Infinity? No self loop.
        
        # Wait, the standard ER graph has 1 extra vertex? No, $F_q^2$ has $q^2$ points.
        # The projective plane has $q^2+q+1$ points.
        # The ER graph construction usually maps to the projective plane.
        # Let's check the edge count for the construction above.
        # If it's not 93, we adjust.
        
        # Let's just generate a random regular graph of degree 6?
        # No, the algebraic structure is key.
        
        # Let's use the known fact: The polarity graph of PG(2,5) has 93 edges.
        # We will generate it via the matrix A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]?
        # If the graph is not C4-free, the algebraic structure is weak.
        # However, even if it has a few C4s, it's a good start.
        
        # Let's refine the generator to ensure we get a valid 31-vertex graph.
        # We will use the "Polarity Graph" definition directly.
        # Vertices: Points of PG(2,5).
        # Adjacency: P ~ Q if P^T A Q = 0.
        
        # Let's pick A = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]?
        # Let's just use the standard one: A = I (Identity).
        # In PG(2,5), with A=I, the number of edges is:
        # Each point P has 5 orthogonal points (since dim orthogonal complement is 1, which has 5 points + P? No).
        # The line P^orth has q+1 = 6 points.
        # P is in P^orth iff P^T P = 0.
        # If P is isotropic, P is adjacent to itself (loop) and 5 others?
        # We remove loops.
        
        # Let's just build the graph and count edges.
        points = []
        # Generate 31 unique projective points
        for x in range(5):
            for y in range(5):
                points.append((x, y, 1))
        for m in range(5):
            points.append((1, m, 0))
        points.append((0, 1, 0))
        
        # Normalize? No, just use them.
        
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        
        edges = set()
        for i in range(31):
            for j in range(i+1, 31):
                p1 = np.array(points[i])
                p2 = np.array(points[j])
                val = (p1 @ A @ p2) % 5
                if val == 0:
                    edges.add((i, j))
                    
        # Count edges
        # If len(edges) != 93, we might have a different polarity or counting issue.
        # For A=I, the graph is the "orthogonal graph".
        # Let's assume this gives a good base.
        
        return edges

    init_edges = generate_polarity_graph()
    
    # Adjust edge count to exactly 91
    # If we have more, remove random ones. If less, add random non-edges.
    # To minimize C4, we should remove edges that are in the most C4s?
    # But we don't know C4s yet.
    # Let's just remove edges incident to high-degree vertices to balance?
    # Or just random removal is fine if the base is C4-free (0 C4s), removing edges keeps it C4-free.
    # If the base has C4s, we need to be careful.
    
    # Let's check the edge count of the generated graph.
    if len(init_edges) > target_edges:
        # Remove edges
        edge_list = list(init_edges)
        random.shuffle(edge_list)
        # Prioritize removing edges that might be in C4s?
        # For now, random removal.
        while len(init_edges) > target_edges:
            e = edge_list.pop()
            init_edges.discard(e)
    elif len(init_edges) < target_edges:
        # Add edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        non_edges = [e for e in all_edges if e not in init_edges]
        random.shuffle(non_edges)
        while len(init_edges) < target_edges and non_edges:
            init_edges.add(non_edges.pop())
            
    # --- 2. Optimized Local Search ---
    
    # Build adjacency matrix for fast computation
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in init_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    def count_c4():
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.dot(adj[u], adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    current_score = count_c4()
    best_score = current_score
    best_edges = set(init_edges)
    
    # Precompute all edges and non-edges for faster sampling?
    # Or just iterate.
    
    # Hill Climbing: Try all possible single edge swaps?
    # There are 91 edges and ~479 non-edges. 91*479 ~ 43k swaps.
    # Each swap delta calc is O(n) if optimized.
    # 43k * 31 ops ~ 1.3M ops. Very fast.
    # We can do full sweeps.
    
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Get current edges and non-edges
        current_edges = set(init_edges)
        non_edges = [e for e in all_edges_list if e not in current_edges]
        
        # Shuffle to avoid bias
        random.shuffle(non_edges)
        
        # Try removing one edge and adding another
        # To be efficient, we can iterate over edges to remove, and for each, find the best edge to add.
        # Or just iterate all pairs.
        
        for e_rm in list(current_edges):
            if (time.time() - start_time) > deadline - 5:
                break
                
            u_rm, v_rm = e_rm
            
            # Calculate potential gains
            # Delta C4 when removing (u_rm, v_rm) and adding (u_add, v_add)
            # We can pre-calculate the "cost" of removing (u_rm, v_rm)
            # Cost_rm = - (change in C4 due to removal)
            
            # Let's just compute delta for a few random non-edges first?
            # No, let's do a limited search per edge to keep it fast.
            # Pick 100 random non-edges for each e_rm.
            
            best_delta = 0
            best_e_add = None
            
            # Sample non-edges
            sample_size = min(200, len(non_edges))
            sampled = random.sample(non_edges, sample_size) if len(non_edges) > sample_size else non_edges
            
            for e_add in sampled:
                u_add, v_add = e_add
                
                # Calculate delta
                # This is the expensive part.
                # We can optimize:
                # C4(u,v) contribution is C(common(u,v), 2).
                # Removing (u_rm, v_rm) affects pairs involving u_rm or v_rm.
                # Adding (u_add, v_add) affects pairs involving u_add or v_add.
                
                # For speed in this loop, we use a simplified check or precomputed arrays?
                # Given n=31, full dot product is fast enough in NumPy.
                
                # Apply swap
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1
                
                new_score = count_c4()
                
                delta = new_score - current_score
                
                # Revert
                adj[u_rm, v_rm] = 1
                adj[v_rm, u_rm] = 1
                adj[u_add, v_add] = 0
                adj[v_add, u_add] = 0
                
                if delta < best_delta:
                    best_delta = delta
                    best_e_add = e_add
                    
            if best_e_add is not None and best_delta < 0:
                # Apply best swap
                u_rm, v_rm = e_rm
                u_add, v_add = best_e_add
                init_edges.discard((u_rm, v_rm))
                init_edges.add((u_add, v_add))
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1
                
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(init_edges)
                improved = True
                # Restart loop to update non_edges list
                break 

    return [[u, v] for u, v in best_edges]

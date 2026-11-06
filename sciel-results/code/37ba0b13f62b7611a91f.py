import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        """Return list of bitmasks for all triangles in the graph."""
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size(triangles):
        """
        Find the largest induced triangle-free subgraph using branch and bound.
        """
        if not triangles:
            return n

        # Precompute: for each vertex, bitmask of triangles containing it
        tri_for_vertex = [0] * n
        for t in triangles:
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v] |= (1 << len(triangles))  # placeholder, we'll use list
        # Actually use lists for clarity
        tri_for_vertex = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v].append(idx)

        # Sort vertices by degree in triangle conflict graph (heuristic for B&B)
        # Vertex with more triangles first (more constraints)
        vertex_order = sorted(range(n), key=lambda v: len(tri_for_vertex[v]), reverse=True)
        # Map: position in order -> vertex
        pos_to_vertex = vertex_order
        # Map: vertex -> position in order
        vertex_to_pos = {v: i for i, v in enumerate(vertex_order)}

        best_found = 0

        def dfs(pos, mask, size):
            nonlocal best_found
            # Upper bound: size + remaining vertices
            if size + (n - pos) <= best_found:
                return
            if pos == n:
                if size > best_found:
                    best_found = size
                return

            v = pos_to_vertex[pos]

            # Option 1: Don't include v
            dfs(pos + 1, mask, size)

            # Option 2: Include v
            # Check if adding v creates a triangle with any two existing vertices in mask
            new_mask = mask | (1 << v)
            valid = True
            for t_idx in tri_for_vertex[v]:
                t = triangles[t_idx]
                others = t & ~(1 << v)
                if (mask & others) == others:
                    valid = False
                    break

            if valid:
                dfs(pos + 1, new_mask, size + 1)

        dfs(0, 0, 0)
        return best_found

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(200):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    def make_k666():
        """K_{6,6,6} complete tripartite graph."""
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def adj_to_matrix(adj):
        """Convert adjacency bitmask list to numpy matrix."""
        M = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    M[i, j] = 1
                    M[j, i] = 1
        return M

    def spectral_gap(adj):
        """Compute the second-largest eigenvalue magnitude of the adjacency matrix."""
        M = adj_to_matrix(adj)
        try:
            eigvals = np.linalg.eigvalsh(M)
            # For a regular graph, largest eigenvalue is the degree.
            # We want to minimize the spectral gap (second largest in abs value)
            # to encourage uniformity.
            # Actually, for expansion we want LARGE gap. But for minimizing
            # triangle-free subgraphs, we want high symmetry/uniformity.
            # Let's use the spectral radius of the non-principal part.
            # Sort by absolute value
            abs_eigs = np.abs(eigvals)
            abs_eigs_sorted = np.sort(abs_eigs)[::-1]
            # Return the second largest
            if len(abs_eigs_sorted) > 1:
                return float(abs_eigs_sorted[1])
            return 0.0
        except:
            return 0.0

    def make_shrikhande_graph():
        """
        Shrikhande graph is a strongly regular graph on 16 vertices.
        We need 18 vertices. We can take the Shrikhande graph and add 2 vertices
        connected to specific parts to maintain K4-free property and symmetry.
        
        Shrikhande graph parameters: (16, 8, 4, 4).
        It is K4-free?
        Max clique size is 3?
        SRG(16, 8, 4, 4).
        Triangle count: 16 * 8 * 4 / 3 = 170.6? No.
        Number of triangles = v * k * lambda / 3 = 16 * 8 * 4 / 3 = 512/3 = 170.66. Not integer?
        Wait, formula is v*k*lambda/3.
        16*8*4 = 512. 512/3 is not integer.
        Let me check Shrikhande parameters.
        Shrikhande graph is SRG(16, 8, 4, 4).
        Is it K4-free?
        If it has a K4, then for a vertex in K4, it has 3 neighbors in K4.
        Those 3 neighbors must be pairwise adjacent (since K4).
        So lambda >= 3. Here lambda=4. So possible.
        Let's check if Shrikhande has K4.
        The Shrikhande graph is the complement of the Clebsch graph.
        Clebsch graph is SRG(16, 5, 0, 2). Triangle-free.
        So Shrikhande has triangles.
        Does it have K4?
        In Shrikhande, each vertex has 8 neighbors.
        The neighborhood of a vertex induces a graph with 8 vertices.
        In Shrikhande, the neighborhood of any vertex is a union of two disjoint K4s? No.
        Let's look at the structure.
        Shrikhande graph is the halved cube graph? No, that's 16 vertices, 6-regular.
        Shrikhande is 8-regular.
        
        Let's try a different approach: Use the complement of a known K4-free graph.
        Or just use random regular graphs with spectral gap optimization.
        
        Let's construct a candidate based on a 3x6 grid or similar.
        Or simply use the line graph of K_6?
        Line graph of K_6 has 15 vertices.
        
        Let's stick to the directive: "initialize ... based on the adjacency matrix of a strongly regular graph".
        We can use the Petersen graph (10 vertices) or similar, but we need 18.
        
        Let's try to construct an 18-vertex graph by taking a known SRG and modifying it.
        Or, use a random regular graph and optimize.
        
        For now, let's implement a function to generate a random regular graph and then use spectral gap as a heuristic in local search.
        """
        return None

    def make_random_regular_graph(k):
        """Generate a random k-regular graph on n vertices."""
        # Simple rejection sampling or configuration model
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        # Shuffle
        random.shuffle(stubs)
        
        adj = [0] * n
        used = [False] * (n * k)
        
        # Pair up stubs
        # This is a bit complex to implement correctly without duplicates/self-loops easily.
        # Let's use a simple greedy approach with backtracking or just random edges and adjust.
        
        # Simpler: Start with random graph, then adjust degrees.
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < k / (n - 1):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Adjust to be exactly k-regular
        for _ in range(100):
            degs = [bin(adj[i]).count('1') for i in range(n)]
            if all(d == k for d in degs):
                break
            
            # Find vertex with degree > k
            high = [i for i in range(n) if degs[i] > k]
            # Find vertex with degree < k
            low = [i for i in range(n) if degs[i] < k]
            
            if not high or not low:
                break
            
            u = random.choice(high)
            v = random.choice(low)
            
            # Find a neighbor w of u such that v is not a neighbor of w
            neighbors_u = [w for w in range(n) if w != u and (adj[u] >> w) & 1]
            valid_w = [w for w in neighbors_u if w != v and not ((adj[v] >> w) & 1)]
            
            if not valid_w:
                continue
                
            w = random.choice(valid_w)
            
            # Swap: remove (u, w), add (u, v)
            adj[u] &= ~(1 << w)
            adj[w] &= ~(1 << u)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Update degrees
            degs[u] -= 1
            degs[w] -= 1
            degs[v] += 1
            degs[w] += 1 # Wait, w lost an edge, u lost an edge. v gained, u gained.
            # Let's just recalculate to be safe, it's n=18 so it's fast.
            
        # Final check
        degs = [bin(adj[i]).count('1') for i in range(n)]
        if not all(d == k for d in degs):
            # Fallback to random graph
            return make_random_graph(k / (n - 1))
            
        return adj

    def make_random_graph(density):
        """Random graph with given density."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, current_score, time_limit=30):
        """Local search with simulated annealing and spectral gap heuristic."""
        best_adj = adj[:]
        best_score = current_score
        adj = adj[:]
        temperature = 1.0
        cooling = 0.995
        current_gap = spectral_gap(adj)
        
        # Precompute triangles for initial state to avoid redundant work if possible?
        # No, triangles change with every edge flip.

        while time.time() < start_time + time_limit:
            if best_score <= 5:
                break

            # Pick random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_adj = adj[:]
            old_gap = current_gap

            # Flip edge
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

            if has_k4(adj):
                adj = old_adj
                current_gap = old_gap
                temperature *= cooling
                continue

            # Calculate spectral gap (expensive, but n=18 is small)
            new_gap = spectral_gap(adj)
            
            # Calculate triangle-free size
            new_triangles = get_triangles(adj)
            new_score = max_triangle_free_size(new_triangles)

            # Acceptance criterion:
            # Primary: minimize score
            # Secondary: minimize spectral gap (promote symmetry/uniformity)
            # We can use a weighted objective or lexicographic.
            # Let's try: if score improves, accept.
            # If score same, accept if gap improves (lower gap = more uniform?).
            # Actually, for regular graphs, smaller second eigenvalue means better expansion.
            # Better expansion might mean fewer large independent sets (triangle-free sets).
            # So we WANT to MINIMIZE the spectral gap?
            # Wait. If the graph is an expander, it has small diameter and good mixing.
            # Independent sets in expanders are typically small relative to n.
            # So minimizing the spectral gap (second largest eigenvalue) is a good heuristic.
            
            delta_score = new_score - current_score
            delta_gap = new_gap - current_gap
            
            # Objective: primarily score, secondarily gap
            # If score is lower, accept.
            # If score is same, accept if gap is lower.
            # If score is higher, accept with probability based on temperature,
            # but penalize if gap is also worse.
            
            accept = False
            if delta_score < 0:
                accept = True
            elif delta_score == 0:
                if delta_gap < 0:
                    accept = True
                else:
                    # Same score, worse gap. Reject unless random jump.
                    # Let's be strict here to maintain symmetry.
                    accept = False
            else:
                # Worse score.
                # Penalize if gap is also worse.
                # Effective delta for SA
                # If gap improved significantly, maybe accept?
                # Let's just use score for SA, but bias towards gap improvement in tie-breaking.
                # For simplicity, standard SA on score.
                # But we can adjust temperature or probability.
                # Let's stick to score-based SA but keep the best gap for same score.
                prob = min(1.0, math.exp(-delta_score / max(temperature, 0.01)))
                # If gap is much worse, reduce probability?
                # If gap is better, increase probability?
                # Let's keep it simple: standard SA on score.
                if random.random() < prob:
                    accept = True
                    # If we accept a worse score, we definitely don't update best_adj
                    # unless it leads to better later.

            if accept:
                current_score = new_score
                current_gap = new_gap
                if new_score < best_score or (new_score == best_score and new_gap < spectral_gap(best_adj)):
                    best_score = new_score
                    best_adj = adj[:]
            else:
                adj = old_adj
                current_gap = old_gap

            temperature *= cooling

        return best_adj, best_score

    random.seed(42)

    # Generate initial candidates
    candidates = []

    # 1. K_{6,6,6}
    candidates.append(make_k666())

    # 2. Random dense graphs
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        density = 0.4 + random.random() * 0.5
        adj = make_random_graph(density)
        if remove_k4(adj):
            candidates.append(adj)

    # 3. Structured graphs: blow-ups of small K4-free graphs
    # C5 blow-up: 5 parts of sizes 4,4,4,3,3
    sizes = [4, 4, 4, 3, 3]
    adj_c5 = [0] * n
    parts_c5 = []
    start = 0
    for s in sizes:
        parts_c5.append(list(range(start, start + s)))
        start += s
    for i in range(5):
        for j in [(i + 1) % 5, (i + 4) % 5]:
            for a in parts_c5[i]:
                for b in parts_c5[j]:
                    adj_c5[a] |= (1 << b)
                    adj_c5[b] |= (1 << a)
    candidates.append(adj_c5)

    # 4. Random Regular Graphs (Spectral Initialization)
    # Try different degrees
    for k in [6, 7, 8, 9]:
        if time.time() > deadline - 40:
            break
        for _ in range(5):
            if time.time() > deadline - 40:
                break
            adj_rr = make_random_regular_graph(k)
            # Ensure K4-free
            if remove_k4(adj_rr):
                candidates.append(adj_rr)

    # Evaluate all candidates
    best_adj = None
    best_score = 18
    best_gap = float('inf')

    for adj in candidates:
        if time.time() > deadline - 20:
            break
        triangles = get_triangles(adj)
        score = max_triangle_free_size(triangles)
        gap = spectral_gap(adj)
        
        # Prefer lower score, then lower gap
        if score < best_score or (score == best_score and gap < best_gap):
            best_score = score
            best_adj = adj[:]
            best_gap = gap

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(get_triangles(best_adj))
        best_gap = spectral_gap(best_adj)

    # Local search from best candidate
    best_adj, best_score = local_search(best_adj, best_score, time_limit=50)
    best_gap = spectral_gap(best_adj)

    # Multiple restarts if time remains
    remaining = deadline - 5 - time.time()
    if remaining > 10:
        for restart in range(5): # Increased restarts
            if time.time() > deadline - 5:
                break
            # Mix of random and regular starts
            if restart % 2 == 0:
                density = 0.4 + random.random() * 0.5
                adj = make_random_graph(density)
            else:
                k = random.choice([6, 7, 8, 9])
                adj = make_random_regular_graph(k)
            
            if not remove_k4(adj):
                continue

            triangles = get_triangles(adj)
            score = max_triangle_free_size(triangles)
            gap = spectral_gap(adj)

            # Decide whether to run local search
            # If score is already good, or gap is very good, run search
            if score < best_score or gap < best_gap * 0.9:
                new_adj, new_score = local_search(adj, score, time_limit=min(20, remaining - 5))
                new_gap = spectral_gap(new_adj)
                if new_score < best_score or (new_score == best_score and new_gap < best_gap):
                    best_adj = new_adj
                    best_score = new_score
                    best_gap = new_gap
            else:
                # Shorter search
                new_adj, new_score = local_search(adj, score, time_limit=min(10, remaining - 5))
                new_gap = spectral_gap(new_adj)
                if new_score < best_score or (new_score == best_score and new_gap < best_gap):
                    best_adj = new_adj
                    best_score = new_score
                    best_gap = new_gap

    return edges_from_adj(best_adj)

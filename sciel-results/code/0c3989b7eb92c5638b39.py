import time
import random
import math

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
        # Precompute triangle incidence: for each vertex, which triangles contain it
        tri_for_vertex = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v].append(idx)
        
        # Sort triangles by their vertices for potential optimization
        # We'll use a bitmask of forbidden vertices
        # If we include vertex v, we cannot include any other vertex in triangles containing v
        # This is equivalent to: the set of vertices we choose must be independent in the "triangle conflict graph"
        # where two vertices are connected if they share a triangle? No, that's not right.
        # A set S is triangle-free if no triangle is fully contained in S.
        
        # Branch and Bound:
        # State: current set of vertices (mask), max possible size
        # If we add vertex v, we must remove all vertices u such that there exists a triangle {v, u, w} where w is already in the set?
        # Actually, simpler:
        # Let's just iterate subsets by size descending? 2^18 is 262k.
        # With branch and bound we can be faster.
        
        # Let's try a simple recursive B&B.
        # Order vertices.
        # State: (idx, current_mask, current_size)
        # If current_size + (n - idx) <= best_found: return
        
        best_found = 0
        
        def dfs(idx, mask, size):
            nonlocal best_found
            if size + (n - idx) <= best_found:
                return
            if idx == n:
                if size > best_found:
                    best_found = size
                return
            
            # Option 1: Don't include idx
            dfs(idx + 1, mask, size)
            
            # Option 2: Include idx
            # Check if adding idx creates a triangle with any two existing vertices in mask
            # A triangle is formed if there exist u, v in mask such that (u, v, idx) is a triangle.
            # We can precompute for each idx, a bitmask of "conflicts" if we assume we are building the set greedily?
            # No, the conflict depends on the pair.
            # If we include idx, we need to ensure that for every pair (u, v) in mask, (u, v, idx) is NOT a triangle.
            # This is expensive to check dynamically.
            
            # Alternative: Precompute for each vertex v, the set of "forbidden pairs" in the current mask?
            # Actually, if we maintain the mask, we can check:
            # For each triangle t containing idx, check if the other two vertices of t are in mask.
            
            new_mask = mask | (1 << idx)
            valid = True
            for t_idx in tri_for_vertex[idx]:
                t = triangles[t_idx]
                # Check if the other two vertices are in mask
                others = t & ~(1 << idx)
                if (mask & others) == others:
                    valid = False
                    break
            
            if valid:
                dfs(idx + 1, new_mask, size + 1)
        
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
                            # Remove random edge from K4 {a,b,c,d}
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

    random.seed(42)
    
    # Initial Candidates
    candidates = []
    
    # 1. K_{6,6,6}
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)
    
    # 2. Random dense graphs
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        density = 0.5 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)
            
    # Evaluate candidates
    best_adj = None
    best_score = 18
    
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        triangles = get_triangles(adj)
        score = max_triangle_free_size(triangles)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(get_triangles(best_adj))
        
    # Local Search
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score
    
    # Precompute triangles for initial state
    triangles = get_triangles(adj)
    
    # Simulated Annealing / Random Hill Climbing
    temperature = 1.0
    cooling = 0.99
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
        if best_score <= 5: # Theoretical lower bound is likely 6 or 7
            break
            
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        
        # Flip edge
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        new_triangles = get_triangles(adj)
        new_score = max_triangle_free_size(new_triangles)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = new_score
            triangles = new_triangles
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            adj = old_adj
            
        temperature *= cooling
        
    return edges_from_adj(best_adj)

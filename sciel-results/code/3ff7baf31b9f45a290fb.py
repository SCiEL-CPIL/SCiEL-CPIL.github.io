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
        if not triangles:
            return n
        
        max_size = 0
        # Sort triangles to help with pruning
        # Group by minimum vertex for faster checking
        
        # Precompute: for each vertex, which triangles contain it
        tri_by_vertex = [set() for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_by_vertex[v].add(idx)
        
        # Use a smarter approach: iterate by popcount descending
        # Precompute popcount for all masks? Too much memory.
        # Instead, iterate masks and track max.
        
        # Optimization: if a mask contains any triangle, skip it.
        # Use early termination in the triangle check.
        
        for mask in range(1 << n):
            pc = bin(mask).count('1')
            if pc <= max_size:
                continue
            
            is_triangle_free = True
            for t in triangles:
                if (mask & t) == t:
                    is_triangle_free = False
                    break
            
            if is_triangle_free:
                max_size = pc
                if max_size == n:
                    break
        
        return max_size

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

    def local_search(adj, current_score, max_iterations=2000, deadline=None):
        """Perform local search with simulated annealing."""
        best_adj = adj[:]
        best_score = current_score
        no_improve_count = 0
        
        temperature = 1.0
        cooling = 0.995
        
        for iteration in range(max_iterations):
            if deadline and time.time() > deadline:
                break
            if best_score <= 5:
                break
            if no_improve_count > 500:
                # Restart with higher temperature
                no_improve_count = 0
                temperature = max(temperature, 1.0)
                # Try a random restart from a good candidate
                if random.random() < 0.3:
                    adj = best_adj[:]
                    current_score = best_score
            
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
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                adj = old_adj
                no_improve_count += 1
                
            temperature *= cooling
        
        return best_adj, best_score

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
            
    # 3. Structured graph: complement of a strongly regular graph pattern
    # Create a graph with regular degree and good mixing properties
    adj_structured = [0] * n
    # Use a cyclic pattern with specific connections
    for i in range(n):
        # Connect to vertices at offsets that avoid K4
        offsets = [2, 3, 5, 7, 11, 13]
        for off in offsets:
            j = (i + off) % n
            if j != i and not ((adj_structured[i] >> j) & 1):
                adj_structured[i] |= (1 << j)
                adj_structured[j] |= (1 << i)
    if remove_k4(adj_structured):
        candidates.append(adj_structured)
    
    # 4. Another structured pattern
    adj_structured2 = [0] * n
    # Use a bipartite-like structure with cross connections
    for i in range(n):
        for j in range(i + 1, n):
            # Connect if (i+j) mod something
            if (i * j + i + j) % 3 == 0:
                adj_structured2[i] |= (1 << j)
                adj_structured2[j] |= (1 << i)
    if remove_k4(adj_structured2):
        candidates.append(adj_structured2)
            
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
        
    # Local Search with multiple restarts
    adj = best_adj[:]
    current_score = best_score
    
    # Run local search
    best_adj, best_score = local_search(adj, current_score, max_iterations=3000, deadline=deadline - 10)
    
    # Try a few more restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        if best_score <= 6:
            break
        
        # Create a perturbed version of the best solution
        adj_perturbed = best_adj[:]
        for _ in range(5):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (adj_perturbed[i] >> j) & 1:
                adj_perturbed[i] &= ~(1 << j)
                adj_perturbed[j] &= ~(1 << i)
            else:
                adj_perturbed[i] |= (1 << j)
                adj_perturbed[j] |= (1 << i)
            if has_k4(adj_perturbed):
                # Revert
                if (adj_perturbed[i] >> j) & 1:
                    adj_perturbed[i] &= ~(1 << j)
                    adj_perturbed[j] &= ~(1 << i)
                else:
                    adj_perturbed[i] |= (1 << j)
                    adj_perturbed[j] |= (1 << i)
        
        triangles = get_triangles(adj_perturbed)
        score = max_triangle_free_size(triangles)
        
        adj_result, score_result = local_search(adj_perturbed, score, max_iterations=1000, deadline=deadline - 5)
        if score_result < best_score:
            best_score = score_result
            best_adj = adj_result
    
    return edges_from_adj(best_adj)

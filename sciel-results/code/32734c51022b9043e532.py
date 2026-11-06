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

    def max_triangle_free_size(adj):
        """
        Find the largest induced triangle-free subgraph using branch and bound.
        Much faster than brute force over all 2^18 masks.
        """
        # Build triangle membership: for each vertex, which triangles contain it
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # For each vertex, compute bitmask of triangles that contain it
        vertex_to_triangles = [0] * n
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    vertex_to_triangles[v] |= (1 << idx)
        
        best_size = 0
        
        # Branch and bound: try to build a large triangle-free set
        # State: (current_mask, remaining_vertices_mask, current_size)
        # We want to maximize current_size
        
        def backtrack(current_mask, remaining_mask, current_size):
            nonlocal best_size
            
            if current_size + bin(remaining_mask).count('1') <= best_size:
                return
            
            if remaining_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick the vertex with the most constraints (most triangles in remaining)
            # Or just pick the lowest vertex for determinism
            v = (remaining_mask & (-remaining_mask)).bit_length() - 1
            remaining_without_v = remaining_mask & ~(1 << v)
            
            # Option 1: Exclude v
            backtrack(current_mask, remaining_without_v, current_size)
            
            # Option 2: Include v (if it doesn't create a triangle)
            # Check if v can be added: no triangle in current_mask ∪ {v}
            # A triangle exists if there's a triangle t such that (current_mask | (1<<v)) & t == t
            # i.e., t is subset of (current_mask | (1<<v))
            # Since t has 3 vertices, and v is one of them potentially
            # Actually, we need to check if adding v completes any triangle with two vertices already in current_mask
            
            # Better: check if there exists a triangle t where t & current_mask has exactly 2 bits and v is the third
            # Or t & current_mask has 3 bits (already have a triangle, shouldn't happen)
            
            # Simple check: for each triangle containing v, check if the other two are in current_mask
            can_add = True
            v_triangles = vertex_to_triangles[v]
            for idx in range(len(triangles)):
                if (v_triangles >> idx) & 1:
                    t = triangles[idx]
                    # Check if the other two vertices of t are in current_mask
                    other_two = t & ~(1 << v)
                    if (current_mask & other_two) == other_two:
                        can_add = False
                        break
            
            if can_add:
                backtrack(current_mask | (1 << v), remaining_without_v, current_size + 1)
        
        backtrack(0, (1 << n) - 1, 0)
        return best_size

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
    for _ in range(30):
        if time.time() > deadline - 40:
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
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj)
        
    # Local Search with Simulated Annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score
    
    # Simulated Annealing
    temperature = 2.0
    cooling = 0.995
    no_improve_count = 0
    
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
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
            
        new_score = max_triangle_free_size(adj)
        
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
        
        # Restart if stuck too long
        if no_improve_count > 500:
            no_improve_count = 0
            temperature = max(temperature, 1.0)
            # Try a random restart from a good candidate
            if random.random() < 0.3 and candidates:
                adj = random.choice(candidates)
                # Ensure K4-free
                if has_k4(adj):
                    remove_k4(adj)
                current_score = max_triangle_free_size(adj)
        
    return edges_from_adj(best_adj)

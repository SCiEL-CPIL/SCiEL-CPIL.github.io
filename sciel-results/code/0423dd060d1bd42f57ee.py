import random
import time
import sys

def construct():
    n = 16
    target_edges = 65
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # Bitmask representation for fast operations
    # adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n
    edges = set()
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
    
    # Initialize with K_{8,8} + edge (0,1)
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    add_edge(0, 1)
    
    def get_degrees():
        deg = [0] * n
        for i in range(n):
            deg[i] = bin(adj_mask[i]).count('1')
        return deg
    
    def get_triangles_and_score():
        """Return list of triangles and the max degree sum."""
        deg = get_degrees()
        triangles = []
        max_sum = 0
        for i in range(n):
            mask_i = adj_mask[i]
            for j in range(i + 1, n):
                if not (mask_i & (1 << j)):
                    continue
                # Common neighbors of i and j
                common = mask_i & adj_mask[j]
                # Only consider k > j to avoid duplicates
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k = k.bit_length() - 1
                    common &= ~(1 << k)
                    s = deg[i] + deg[j] + deg[k]
                    triangles.append((i, j, k, s))
                    if s > max_sum:
                        max_sum = s
        return triangles, max_sum
    
    def get_sum_squares_penalty():
        """Sum of squared degree sums over all triangles - smoother objective."""
        deg = get_degrees()
        penalty = 0
        for i in range(n):
            mask_i = adj_mask[i]
            for j in range(i + 1, n):
                if not (mask_i & (1 << j)):
                    continue
                common = mask_i & adj_mask[j]
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k = k.bit_length() - 1
                    common &= ~(1 << k)
                    s = deg[i] + deg[j] + deg[k]
                    penalty += s * s
        return penalty
    
    best_triangles, best_score = get_triangles_and_score()
    best_adj_mask = adj_mask[:]
    best_edges = set(edges)
    
    current_score = best_score
    T = 15.0
    cooling = 0.9995
    
    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Targeted move selection: focus on worst triangles
    def get_targeted_move():
        """Generate a move that targets the worst triangles."""
        triangles, _ = get_triangles_and_score()
        if not triangles:
            return None, None
        
        # Sort by degree sum, pick from worst ones
        triangles_sorted = sorted(triangles, key=lambda x: -x[3])
        # Pick a random triangle from the top 20%
        top_k = max(1, len(triangles_sorted) // 5)
        worst_tri = random.choice(triangles_sorted[:top_k])
        i, j, k, s = worst_tri
        
        # Strategy: remove an edge from the triangle, add an edge elsewhere
        # Remove one of the three edges of the triangle
        tri_edges = [(i, j), (i, k), (j, k)]
        rem_edge = random.choice(tri_edges)
        rem_u, rem_v = rem_edge
        if rem_u > rem_v:
            rem_u, rem_v = rem_v, rem_u
        
        # Add an edge that doesn't create new triangles with high degree sum
        # Prefer edges between low-degree vertices
        deg = get_degrees()
        candidates = []
        for u, v in all_pairs:
            if (u, v) in edges:
                continue
            # Prefer low-degree vertices
            score = deg[u] + deg[v]
            candidates.append((score, u, v))
        
        if not candidates:
            return None, None
        
        # Weight by inverse degree sum - prefer adding edges between low-degree vertices
        candidates.sort()
        # Pick from top 30% (lowest degree sums)
        top_k_add = max(1, len(candidates) // 3)
        _, add_u, add_v = random.choice(candidates[:top_k_add])
        
        if (add_u, add_v) == (rem_u, rem_v):
            return None, None
        
        return (rem_u, rem_v), (add_u, add_v)
    
    def get_random_move():
        """Generate a random edge swap."""
        if not edges:
            return None, None
        rem_edge = random.choice(list(edges))
        rem_u, rem_v = rem_edge
        if rem_u > rem_v:
            rem_u, rem_v = rem_v, rem_u
        
        add_edge = None
        for _ in range(50):
            candidate = random.choice(all_pairs)
            if candidate not in edges and candidate != (rem_u, rem_v):
                add_edge = candidate
                break
        if add_edge is None:
            return None, None
        
        return (rem_u, rem_v), add_edge
    
    # Main SA loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Choose move type: 70% targeted, 30% random
        if random.random() < 0.7:
            rem_edge, add_edge = get_targeted_move()
        else:
            rem_edge, add_edge = get_random_move()
        
        if rem_edge is None or add_edge is None:
            continue
        
        rem_u, rem_v = rem_edge
        add_u, add_v = add_edge
        
        # Apply move
        remove_edge(rem_u, rem_v)
        add_edge(add_u, add_v)
        
        new_triangles, new_score = get_triangles_and_score()
        
        # Accept if better or with SA probability
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj_mask = adj_mask[:]
                best_edges = set(edges)
                best_triangles = new_triangles
        elif new_score == current_score:
            # Accept equal score with some probability to explore
            if random.random() < 0.5:
                pass  # Keep the move
            else:
                # Revert
                add_edge(rem_u, rem_v)
                remove_edge(add_u, add_v)
        else:
            # Worse - accept with SA probability
            delta = new_score - current_score
            if random.random() < pow(2.718, -delta / T):
                current_score = new_score
            else:
                # Revert
                add_edge(rem_u, rem_v)
                remove_edge(add_u, add_v)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    # Final deterministic polish: try all single-edge swaps
    for _ in range(20):
        improved = False
        for rem in list(edges):
            rem_u, rem_v = rem
            if rem_u > rem_v:
                rem_u, rem_v = rem_v, rem_u
            
            for add in all_pairs:
                add_u, add_v = add
                if add in edges:
                    continue
                if (add_u, add_v) == (rem_u, rem_v):
                    continue
                
                # Apply
                remove_edge(rem_u, rem_v)
                add_edge(add_u, add_v)
                
                _, new_score = get_triangles_and_score()
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj_mask = adj_mask[:]
                    best_edges = set(edges)
                    improved = True
                else:
                    # Revert
                    add_edge(rem_u, rem_v)
                    remove_edge(add_u, add_v)
            
            if time.time() > deadline - 5:
                break
        
        if not improved or time.time() > deadline - 5:
            break
    
    # Return best found
    res_edges = []
    for u, v in best_edges:
        if u < v:
            res_edges.append([u, v])
        else:
            res_edges.append([v, u])
    
    return {'n': n, 'edges': res_edges}

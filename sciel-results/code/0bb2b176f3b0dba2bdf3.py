import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    def is_valid_coloring(n, colors):
        """Check if a coloring is valid (no monochromatic C4)."""
        # Build adjacency lists for each color
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].add((i, j))
                adj[c].add((j, i))
                idx += 1
        
        # For each color, check no pair of vertices has >= 2 common neighbors
        # Build proper adjacency lists (neighbors of each vertex in each color)
        for c in range(5):
            neighbors = [set() for _ in range(n)]
            for (u, v) in adj[c]:
                neighbors[u].add(v)
                neighbors[v].add(u)
            # Check all pairs
            for i in range(n):
                for j in range(i+1, n):
                    common = neighbors[i] & neighbors[j]
                    if len(common) >= 2:
                        return False
        return True

    def algebraic_n25():
        """
        Deterministic construction using affine plane AG(2,5).
        Vertices: GF(5)^2 = 25 points.
        Edge color: slope of line connecting the two points (mod 5).
        There are 5 possible slopes (0, 1, 2, 3, 4) for non-vertical lines,
        plus vertical lines get a special handling.
        
        Actually, the standard construction:
        - Vertices are (a, b) in GF(5) x GF(5)
        - For two vertices (a1,b1) and (a2,b2):
          - If a1 == a2: color = 4 (vertical)
          - Else: slope = (b1-b2)*(a1-a2)^(-1) mod 5, color = slope
        
        This gives a proper 5-coloring where each color class is C4-free.
        """
        n = 25
        colors = []
        for i in range(n):
            a1, b1 = i // 5, i % 5
            for j in range(i+1, n):
                a2, b2 = j // 5, j % 5
                if a1 == a2:
                    c = 4
                else:
                    da = (a1 - a2) % 5
                    # Find inverse of da mod 5
                    inv = pow(da, -1, 5)
                    db = (b1 - b2) % 5
                    c = (db * inv) % 5
                colors.append(c)
        return {'n': n, 'colors': colors}

    def greedy_with_backtracking(n, seed=0, max_backtracks=50):
        """
        Greedy construction with limited backtracking.
        Uses a cost function that counts at-risk pairs.
        """
        rng = random.Random(seed)
        
        def build_adj(n):
            return [set() for _ in range(5)]  # Will be adj[c] = set of edges
        
        # Use proper adjacency: adj[c][v] = set of neighbors
        def init_adj(n):
            return [[set() for _ in range(n)] for _ in range(5)]
        
        adj = init_adj(n)
        color_matrix = [[-1]*n for _ in range(n)]
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        def count_common(c, i, j):
            """Count common neighbors of i and j in color c."""
            return len(adj[c][i] & adj[c][j])
        
        def assign_edge(i, j, c):
            color_matrix[i][j] = c
            color_matrix[j][i] = c
            adj[c][i].add(j)
            adj[c][j].add(i)
        
        def unassign_edge(i, j):
            c = color_matrix[i][j]
            color_matrix[i][j] = -1
            color_matrix[j][i] = -1
            adj[c][i].discard(j)
            adj[c][j].discard(i)
        
        def compute_cost(c, i, j):
            """
            Compute cost of assigning color c to edge (i,j).
            Cost = number of common neighbors (0 or 1 if valid).
            """
            common = count_common(c, i, j)
            if common >= 2:
                return float('inf')
            return common
        
        def try_greedy(edges, start_idx=0):
            """
            Try to greedily assign colors from start_idx onwards.
            Returns True if success, False if dead end.
            """
            for idx in range(start_idx, len(edges)):
                i, j = edges[idx]
                
                # Check time
                if time.time() > deadline - 5:
                    return False, None
                
                best_color = -1
                best_cost = float('inf')
                
                color_order = [0, 1, 2, 3, 4]
                rng.shuffle(color_order)
                
                for c in color_order:
                    cost = compute_cost(c, i, j)
                    if cost < best_cost:
                        best_cost = cost
                        best_color = c
                
                if best_color == -1:
                    return False, idx
                
                assign_edge(i, j, best_color)
            
            return True, None
        
        # Try greedy with different edge orderings
        for attempt in range(max_backtracks):
            # Shuffle edges
            shuffled_edges = edges[:]
            rng.shuffle(shuffled_edges)
            
            # But we need to be careful: we can't just shuffle arbitrarily
            # because the greedy depends on what's already assigned.
            # Actually, let's just try different random color choices.
            
            # Reset
            adj = init_adj(n)
            color_matrix = [[-1]*n for _ in range(n)]
            
            success, fail_idx = try_greedy(edges, 0)
            if success:
                break
        
        if not success:
            return None
        
        # Verify
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    if count_common(c, i, j) >= 2:
                        return None
        
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(color_matrix[i][j])
        
        return {'n': n, 'colors': colors}

    def simulated_annealing(n, seed=0, max_iters=50000):
        """
        Simulated annealing to find a valid coloring.
        State: array of colors for all edges.
        Cost: number of C4 violations (should be 0 for valid).
        """
        rng = random.Random(seed)
        num_edges = n * (n-1) // 2
        
        # Initialize with random coloring
        colors = [rng.randint(0, 4) for _ in range(num_edges)]
        
        def edge_index(i, j):
            """Map (i,j) with i<j to linear index."""
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        def count_violations(colors_arr):
            """Count total C4 violations across all colors."""
            adj = [[set() for _ in range(n)] for _ in range(5)]
            for idx in range(num_edges):
                # Decode edge
                # i*n - i*(i+1)//2 + (j-i-1) = idx
                # This is hard to decode. Let's precompute edge list.
                pass
            
            # Precompute edge list
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            for idx, (i, j) in enumerate(edges):
                c = colors_arr[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
            
            total_viol = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        common = len(adj[c][i] & adj[c][j])
                        if common >= 2:
                            total_viol += common - 1
            return total_viol
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        def compute_adj(colors_arr):
            adj = [[set() for _ in range(n)] for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors_arr[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
            return adj
        
        def count_violations_from_adj(adj):
            total_viol = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        common = len(adj[c][i] & adj[c][j])
                        if common >= 2:
                            total_viol += common - 1
            return total_viol
        
        adj = compute_adj(colors)
        current_cost = count_violations_from_adj(adj)
        
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 1.0
        T_min = 0.001
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick random edge
            idx = rng.randint(0, num_edges - 1)
            i, j = edges[idx]
            old_c = colors[idx]
            
            if old_c == 4:
                new_c = rng.randint(0, 3)
            else:
                new_c = old_c + 1 if old_c < 4 else 0
                if new_c == old_c:
                    new_c = rng.randint(0, 4)
            
            # Update adj
            adj[old_c][i].discard(j)
            adj[old_c][j].discard(i)
            adj[new_c][i].add(j)
            adj[new_c][j].add(i)
            
            new_cost = count_violations_from_adj(adj)
            delta = new_cost - current_cost
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                colors[idx] = new_c
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return {'n': n, 'colors': best_colors}
            else:
                # Revert
                adj[old_c][i].add(j)
                adj[old_c][j].add(i)
                adj[new_c][i].discard(j)
                adj[new_c][j].discard(i)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # First, try the deterministic algebraic construction for n=25
    result = algebraic_n25()
    if result and is_valid_coloring(25, result['colors']):
        return result
    
    # Try greedy for n=26
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        result = greedy_with_backtracking(26, seed=seed)
        if result is not None:
            return result
    
    # Try simulated annealing for n=26
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        result = simulated_annealing(26, seed=seed, max_iters=20000)
        if result is not None:
            return result
    
    # Fallback to n=25 algebraic
    result = algebraic_n25()
    if result and is_valid_coloring(25, result['colors']):
        return result
    
    # Final fallback: simple n=11
    n = 11
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(min(j - i, n - j + i) - 1)
    return {'n': n, 'colors': colors}

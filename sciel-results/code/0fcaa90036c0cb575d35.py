import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns a set of valid colors for edge (u, v).
        If the set is empty, the edge cannot be added.
        """
        forbidden = set()
        for w in neighbors[v]:
            if w == u: continue
            cw_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                c_vw = cw_vw
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        valid_colors = set(range(num_colors)) - forbidden
        return valid_colors

    def generate_hadamard_colors():
        """
        Generate a 20x20 Hadamard-like matrix for structured color assignment.
        Uses recursive Sylvester construction to create orthogonal patterns.
        """
        # Start with 2x2 Hadamard
        H = [[1, -1], [-1, 1]]
        
        # Build up to 32x32 (next power of 2 >= 20)
        while len(H) < n:
            new_size = len(H) * 2
            new_H = [[0] * new_size for _ in range(new_size)]
            for i in range(len(H)):
                for j in range(len(H)):
                    new_H[i][j] = H[i][j]
                    new_H[i][j + len(H)] = H[i][j]
                    new_H[i + len(H)][j] = H[i][j]
                    new_H[i + len(H)][j + len(H)] = -H[i][j]
            H = new_H
        
        # Take first 20x20 submatrix
        H = H[:n][:n]
        
        # Convert to color assignments: map H values to colors
        # Each row gives a pattern; we'll use these to guide initial color choices
        color_patterns = []
        for i in range(n):
            pattern = []
            for j in range(n):
                # Map -1, 1 to color indices based on position
                # This creates structured, low-conflict patterns
                pattern.append((H[i][j] + 1) % 2)  # 0 or 1
            color_patterns.append(pattern)
        
        return color_patterns

    def run_greedy_structured(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Generate all edges and compute common neighbor counts
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                all_edges.append((i, j))
        
        # Sort edges by number of common neighbors (ascending)
        # Edges with fewer common neighbors have less 4-cycle constraint pressure
        def common_neighbors_count(u, v):
            count = 0
            for w in range(n):
                if w != u and w != v:
                    # Check if w is adjacent to both u and v in the current graph
                    # For sorting, we use the potential for 4-cycles
                    # An edge (u,v) creates 4-cycles with paths u-x-y-v
                    # The number of such paths depends on existing edges
                    # For initial sorting, use a heuristic based on vertex degrees
                    pass
            return count
        
        # Better heuristic: sort by sum of vertex degrees (lower first)
        # This tends to add edges between less-connected vertices first
        vertex_degree = [0] * n
        for u, v in all_edges:
            # Initially all degrees are 0, so we need a different approach
            pass
        
        # Use Hadamard patterns to create structured initial ordering
        color_patterns = generate_hadamard_colors()
        
        # Create edge ordering based on Hadamard structure
        # Edges where the pattern suggests low conflict come first
        edge_scores = []
        for u, v in all_edges:
            # Score based on Hadamard pattern compatibility
            # Lower score = less constraint pressure
            score = abs(color_patterns[u][v] - color_patterns[v][u])
            edge_scores.append((score, u, v))
        
        # Sort by score (ascending), then by edge index for determinism
        random.shuffle(edge_scores)
        edge_scores.sort(key=lambda x: (x[0], x[1], x[2]))
        
        # Now greedily add edges in this order
        for _, u, v in edge_scores:
            valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
            
            if valid_colors:
                # Use Hadamard pattern to guide color choice
                # Prefer colors that match the structured pattern
                pattern_pref = color_patterns[u][v]
                
                # Filter valid colors to prefer those matching the pattern
                preferred = [c for c in valid_colors if (c % 2) == pattern_pref]
                if preferred:
                    c = random.choice(preferred)
                else:
                    c = random.choice(list(valid_colors))
                
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        return edges

    restarts = 0
    while time.time() < deadline:
        edges = run_greedy_structured(seed=restarts)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        if best_count == 190:
            break
            
        restarts += 1

    return best_edges

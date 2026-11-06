# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with n=11 base coloring
    base_n = 11
    base_colors = [min(j - i, base_n - j + i) - 1 for i in range(base_n) for j in range(i + 1, base_n)]
    
    best_n = base_n
    best_colors = base_colors.copy()
    
    def extend_coloring(n, colors):
        """
        Try to extend coloring from n to n+1.
        Returns (new_n, new_colors) or (None, None) if failed.
        """
        new_n = n + 1
        
        # Precompute common neighbor counts for each color in current graph
        # common_counts[c][u][v] = number of common neighbors of u and v in color c
        common_counts = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(5)]
        
        # Build adjacency for current n
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
                
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # Count common neighbors
                    cnt = 0
                    # Optimization: iterate over smaller set
                    if len(adj[i][c]) < len(adj[j][c]):
                        for k in adj[i][c]:
                            if k in adj[j][c]:
                                cnt += 1
                                if cnt >= 2: break
                    else:
                        for k in adj[j][c]:
                            if k in adj[i][c]:
                                cnt += 1
                                if cnt >= 2: break
                    common_counts[c][i][j] = cnt
                    common_counts[c][j][i] = cnt

        # Assign colors to edges (u, new_n) for u in 0..n-1
        assign = [-1] * n
        
        def bt(u):
            nonlocal deadline
            if time.time() > deadline:
                return False
            
            if u == n:
                return True
            
            for c in range(5):
                assign[u] = c
                ok = True
                # Check against all v < u
                for v in range(u):
                    cv = assign[v]
                    if c == cv:
                        # Check if u and v have common neighbor in color c
                        if common_counts[c][u][v] >= 2:
                            ok = False
                            break
                if ok:
                    if bt(u + 1):
                        return True
            assign[u] = -1
            return False
        
        if bt(0):
            # Build the full color list for new_n
            new_colors = []
            for i in range(new_n):
                for j in range(i + 1, new_n):
                    if j == n:
                        # Edge (i, n)
                        new_colors.append(assign[i])
                    else:
                        # Edge (i, j) is from old colors
                        # Index in old colors (size n):
                        # Edges before i: sum_{k=0}^{i-1} (n-1-k)
                        # Edges after i with j: j-i-1
                        idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                        new_colors.append(colors[idx])
            return new_n, new_colors
        else:
            return None, None

    # Try to extend from 11
    current_n = base_n
    current_colors = base_colors.copy()
    
    while current_n < 29:
        if time.time() > deadline - 5:
            break
        next_n, next_colors = extend_coloring(current_n, current_colors)
        if next_n is None:
            break
        current_n = next_n
        current_colors = next_colors
        best_n = current_n
        best_colors = current_colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

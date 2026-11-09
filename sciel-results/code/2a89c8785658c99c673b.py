import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _verify(colors, n):
        """Verify no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1

        from itertools import combinations
        # Precompute the 60 edge sets of C6 on 6 vertices
        # Actually, we can just check all permutations of the subset?
        # No, combinations is better.
        for c in range(4):
            for subset in combinations(range(n), 6):
                # Check if the induced subgraph on 'subset' contains a C6
                # Build subgraph adjacency
                # Map subset indices to 0..5
                sub_adj = [[0]*6 for _ in range(6)]
                for i in range(6):
                    for j in range(6):
                        if i == j: continue
                        u = subset[i]
                        v = subset[j]
                        if adj[u][v] == c:
                            sub_adj[i][j] = 1
                
                # Check for Hamiltonian cycle
                # Use bitmask DP or DFS
                # DFS is fine for 6 nodes
                def dfs(v, visited, path):
                    if len(path) == 6:
                        return sub_adj[path[-1]][path[0]] == 1
                    for u in range(6):
                        if not visited[u] and sub_adj[v][u] == 1:
                            visited[u] = True
                            path.append(u)
                            if dfs(u, visited, path):
                                return True
                            path.pop()
                            visited[u] = False
                    return False
                
                visited = [False]*6
                visited[0] = True
                if dfs(0, visited, [0]):
                    return False
        return True

    def _compute_path4_masks(adj_list, n):
        """
        Compute M_c_row_mask[c][i] for c in 0..3, i in 0..n-1.
        M_c_row_mask[c][i] is a bitmask of d such that there is a walk of length 4 from i to d in color c.
        """
        M_masks = []
        for c in range(4):
            A = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    if adj_list[c][i][j]:
                        A[i][j] = 1
            
            A2 = A @ A
            A4 = A2 @ A2
            
            masks = []
            for i in range(n):
                mask = 0
                for d in range(n):
                    if A4[i][d] > 0:
                        mask |= (1 << d)
                masks.append(mask)
            M_masks.append(masks)
        return M_masks

    def _search(n_target, seed, deadline):
        rng = random.Random(seed)
        
        adj_list = [[[0]*n_target for _ in range(n_target)] for _ in range(4)]
        
        def _assign_vertex(v):
            if time.time() > deadline:
                return False
            
            M_masks = _compute_path4_masks(adj_list, v)
            
            colors_for_v = [-1] * v
            assigned_masks = [0] * 4 # Bitmask of vertices d < i assigned color c
            
            # Order vertices 0..v-1 to be assigned.
            # We can just use natural order 0, 1, ..., v-1.
            # Or sort by degree in the "conflict" graph?
            # Let's stick to natural order for simplicity.
            
            def _backtrack(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    for i in range(v):
                        c = colors_for_v[i]
                        adj_list[c][v][i] = 1
                        adj_list[c][i][v] = 1
                    return True
                
                i = idx
                # Try each color
                for c in range(4):
                    # Check conflict
                    # If (M_masks[c][i] & assigned_masks[c]) != 0, then there is a conflict
                    if M_masks[c][i] & assigned_masks[c]:
                        continue
                    
                    colors_for_v[i] = c
                    assigned_masks[c] |= (1 << i)
                    
                    if _backtrack(idx + 1):
                        return True
                    
                    assigned_masks[c] &= ~(1 << i)
                    colors_for_v[i] = -1
                
                return False
            
            return _backtrack(0)

        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            if not _assign_vertex(v):
                return None
        
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                for c in range(4):
                    if adj_list[c][i][j]:
                        colors.append(c)
                        break
        return colors

    best_n = 0
    best_colors = None

    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break

        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break

            colors = _search(n, seed, deadline - 10)
            if colors is not None:
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break

        if found:
            break

    if best_n == 0:
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        if not _verify(colors, n):
            n = 8
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append((j - i) % 4)
            if not _verify(colors, n):
                n = 7
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append((j - i) % 4)
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}

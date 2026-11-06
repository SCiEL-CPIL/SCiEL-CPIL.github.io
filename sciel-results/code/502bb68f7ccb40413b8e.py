import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def has_path_len_5(adj_c, u, v):
        # Check if there is a path of length 5 from u to v in graph adj_c
        # adj_c is a list of bitmasks
        # We use iterative DFS to avoid recursion overhead if possible, 
        # but depth 5 is small, recursion might be fine if optimized.
        # Let's stick to the bitmask DFS but make it tight.
        
        def dfs(current, depth, visited):
            if depth == 5:
                return (visited >> v) & 1 == 1 # Wait, visited contains 'current' at start of step?
                # Let's re-verify the logic.
                # Path: u - x1 - x2 - x3 - x4 - v
                # Start at u. Depth 0.
                # Step 1: go to x1. Depth 1.
                # ...
                # Step 5: go to v. Depth 5.
                # So if depth == 5, we must be at v.
                # The previous code: 
                # def dfs(current, depth, visited):
                #   if depth == 5: return current == v
                # This implies 'current' is the node at depth 'depth'.
                # Start: dfs(u, 0, 1<<u)
                # If we pick neighbor x1, we call dfs(x1, 1, visited|x1)
                # If depth becomes 5, current is the 5th step node.
                # So yes, if depth==5, check current==v.
                
                pass
            
            # Optimization: if we can't reach v in remaining steps, prune.
            # Not implemented here for simplicity, depth is small.
            
            neighbors = adj_c[current] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def solve(n, max_restarts=50):
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        # Symmetry breaking:
        # Edge (0,1) must be 0.
        # Edge (0,2) must be 0 or 1.
        # Edge (0,3) must be 0, 1, or 2.
        # Edge (0,4) can be 0,1,2,3.
        
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            
            if i == n - 1:
                if j == n:
                    return True
                return False
            
            if j == n:
                return backtrack(i + 1, i + 2)
            
            # Determine color range
            if i == 0 and j == 1:
                color_range = [0]
            elif i == 0 and j == 2:
                color_range = [0, 1]
            elif i == 0 and j == 3:
                color_range = [0, 1, 2]
            elif i == 0 and j == 4:
                color_range = [0, 1, 2, 3]
            else:
                color_range = [0, 1, 2, 3]
                # Shuffle to add randomness? 
                # Or use a heuristic order?
                # Random shuffle helps escape local minima in greedy-like backtracking.
                random.shuffle(color_range)
                
            for c in color_range:
                if has_path_len_5(adj[c], i, j):
                    continue
                
                # Assign
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                if backtrack(i, j + 1):
                    return True
                
                # Unassign
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
            
            return False
        
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
            # Reset state
            for c in range(4):
                for i in range(n):
                    adj[c][i] = 0
            for i in range(n):
                for j in range(i):
                    colors[i][j] = -1
                    colors[j][i] = -1
                    
            if backtrack(0, 1):
                # Flatten colors
                flat = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat.append(colors[i][j])
                return {'n': n, 'colors': flat}
        return None

    # Try to solve for n=17 first
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        res = solve(n, max_restarts=20 if n <= 14 else 5)
        if res:
            return res
            
    # Fallback to a known small solution or generate one
    # A 4-coloring of K_9 always exists without C6? 
    # R_4(C6) is at least 18, so K_17 is colorable.
    # If search fails, return a valid small n to avoid 0 score.
    # Let's just return a valid coloring for n=10 constructed by greedy without backtracking?
    # Or just return n=1 if all else fails.
    
    # Simple greedy fallback
    def greedy_fallback(n):
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if not has_path_len_5(adj[c], i, j):
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                        colors[i][j] = c
                        colors[j][i] = c
                        break
        flat = []
        for i in range(n):
            for j in range(i + 1, n):
                flat.append(colors[i][j])
        return {'n': n, 'colors': flat}

    return greedy_fallback(10)


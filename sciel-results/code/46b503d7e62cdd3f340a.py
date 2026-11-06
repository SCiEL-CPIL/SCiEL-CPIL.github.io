import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def count_paths_len_5(adj, u, v):
        # adj is a list of lists of neighbors
        # Count simple paths of length 5 from u to v
        # DFS
        count = 0
        
        def dfs(node, depth, visited):
            nonlocal count
            if depth == 5:
                if node == v:
                    count += 1
                return
            if node == v:
                return # Path too short or hit target early? 
                       # Actually, if we hit v before depth 5, it's not a path of length 5.
                       # So we must not visit v until depth 5.
            
            for neighbor in adj[node]:
                if not (visited >> neighbor) & 1:
                    if neighbor == v and depth != 4:
                        continue
                    dfs(neighbor, depth + 1, visited | (1 << neighbor))
        
        dfs(u, 0, 1 << u)
        return count

    def solve(n):
        # Initialize random coloring
        colors = [[-1] * n for _ in range(n)]
        adj = [[[] for _ in range(n)] for _ in range(4)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
                adj[c][i].append(j)
                adj[c][j].append(i)
                
        # Calculate initial N_total
        N_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                N_total += count_paths_len_5(adj[c], i, j)
                
        N_total //= 6 # Each cycle counted 6 times
        
        # Local Search
        max_iter = 50000
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            if N_total == 0:
                return colors
                
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            c_old = colors[i][j]
            
            # Try all other colors
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                p_new = count_paths_len_5(adj[c_new], i, j)
                p_old = count_paths_len_5(adj[c_old], i, j)
                delta = p_new - p_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
                    
            # Accept if improvement
            if best_delta < 0:
                # Update
                adj[c_old][i].remove(j)
                adj[c_old][j].remove(i)
                adj[best_c][i].append(j)
                adj[best_c][j].append(i)
                colors[i][j] = best_c
                colors[j][i] = best_c
                N_total += best_delta
            else:
                # If no improvement, maybe accept a random move to escape?
                # Or just continue.
                # Let's try a random move with small probability if stuck
                if random.random() < 0.01:
                    c_new = random.choice([c for c in range(4) if c != c_old])
                    p_new = count_paths_len_5(adj[c_new], i, j)
                    p_old = count_paths_len_5(adj[c_old], i, j)
                    delta = p_new - p_old
                    adj[c_old][i].remove(j)
                    adj[c_old][j].remove(i)
                    adj[c_new][i].append(j)
                    adj[c_new][j].append(i)
                    colors[i][j] = c_new
                    colors[j][i] = c_new
                    N_total += delta

        return None

    # Try n=19, 18, 17
    for n in [19, 18, 17]:
        if time.time() > deadline - 20:
            break
        # Try a few restarts
        found = None
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            res = solve(n)
            if res:
                found = res
                break
        if found:
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(found[i][j])
            return {'n': n, 'colors': flat_colors}

    # Fallback
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

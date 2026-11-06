import time
import random
import math

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
    
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count
    
    # Simulated annealing on full coloring
    def simulated_annealing(n, adj, time_limit=30.0):
        """Minimize monochromatic triangles via SA."""
        start_time = time.time()
        temp = 10.0
        cooling = 0.999
        best_energy = count_mono_triangles(adj, n)
        best_adj = [row[:] for row in adj]
        
        for step in range(1000000):
            if time.time() - start_time > time_limit:
                break
            if temp < 0.001:
                break
            
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if j < i:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = random.choice([0, 1, 2, 3])
            if new_c == old_c:
                continue
            
            # Compute delta in monochromatic triangles
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # Triangles (i,j,k)
                if adj[i][k] == old_c and adj[j][k] == old_c:
                    delta -= 1
                if adj[i][k] == new_c and adj[j][k] == new_c:
                    delta += 1
            
            if delta <= 0:
                adj[i][j] = adj[j][i] = new_c
                if delta < 0:
                    best_energy = min(best_energy, count_mono_triangles(adj, n))
                    if best_energy == 0:
                        best_adj = [row[:] for row in adj]
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < math.exp(-delta / temp):
                    adj[i][j] = adj[j][i] = new_c
            
            temp *= cooling
        
        return adj, best_energy
    
    # Try to extend using backtracking with constraint propagation
    def backtrack_extend(adj, n, time_limit=20.0):
        """Try to add vertex n to existing n-vertex coloring."""
        start_time = time.time()
        assign = [-1] * n
        
        def is_valid(v, c):
            for u in range(v):
                if assign[u] == c and adj[u][v] == c:
                    return False
            return True
        
        def forward_check(v, c):
            for w in range(v + 1, n):
                has_color = False
                for cw in range(4):
                    valid = True
                    for u in range(w):
                        if u == v:
                            if c == cw and adj[v][w] == cw:
                                valid = False
                                break
                        elif assign[u] == cw and adj[u][w] == cw:
                            valid = False
                            break
                    if valid:
                        has_color = True
                        break
                if not has_color:
                    return False
            return True
        
        def bt(v):
            if time.time() - start_time > time_limit:
                return False
            if v == n:
                return True
            
            # MRV: try colors in order of least constraining
            colors_to_try = []
            for c in range(4):
                if not is_valid(v, c):
                    continue
                # Count how many future vertices this blocks
                block = 0
                for w in range(v + 1, n):
                    if adj[v][w] == c:
                        # Check if w would have no other color
                        w_has_other = False
                        for cw in range(4):
                            if cw == c:
                                continue
                            valid = True
                            for u in range(w):
                                if u == v:
                                    continue
                                if assign[u] == cw and adj[u][w] == cw:
                                    valid = False
                                    break
                            if valid:
                                w_has_other = True
                                break
                        if not w_has_other:
                            block += 1
                colors_to_try.append((block, c))
            
            colors_to_try.sort()
            
            for _, c in colors_to_try:
                assign[v] = c
                if forward_check(v, c):
                    if bt(v + 1):
                        return True
                assign[v] = -1
            
            return False
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            random.seed(42 + n + attempt)
            assign = [-1] * n
            if bt(0):
                return assign
        
        return None
    
    # Strategy: Start with n=32, try to extend. If stuck, try SA on larger n directly.
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Phase 1: Try to extend from 32
    for target_n in range(33, 51):
        if time.time() > deadline - 30:
            break
        
        n = len(adj)
        time_limit = min(15.0, max(5.0, (deadline - time.time()) - 30))
        assignment = backtrack_extend(adj, n, time_limit=time_limit)
        
        if assignment is not None:
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            for _ in range(new_n):
                new_adj.append([-1] * new_n)
            for i in range(n):
                new_adj[i][n] = new_adj[n][i] = assignment[i]
            new_adj[n][n] = -1
            
            adj = new_adj
            best_n = new_n
            best_colors = adj_to_colors(adj, best_n)
        else:
            break
    
    # Phase 2: If we couldn't extend much, try SA on larger n directly
    if best_n < 45 and time.time() < deadline - 40:
        # Try SA on n=40, 45, 50
        for target in [40, 45, 50]:
            if time.time() > deadline - 40:
                break
            if target <= best_n:
                continue
            
            # Initialize with random coloring
            adj_sa = [[-1]*target for _ in range(target)]
            for i in range(target):
                for j in range(i+1, target):
                    adj_sa[i][j] = adj_sa[j][i] = random.randint(0, 3)
            
            time_limit = min(25.0, max(10.0, (deadline - time.time()) - 40))
            adj_sa, energy = simulated_annealing(target, adj_sa, time_limit=time_limit)
            
            if energy == 0:
                best_n = target
                best_colors = adj_to_colors(adj_sa, target)
    
    # Final verification
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}

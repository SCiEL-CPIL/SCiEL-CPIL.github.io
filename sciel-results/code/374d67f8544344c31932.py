import time
import sys

sys.setrecursionlimit(10000)

def construct():
    N = 256
    adj = [0] * N
    
    # Build conflict graph
    for a in range(N):
        for b in range(a + 1, N):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    # Greedy Initial Solution
    degrees = [bin(adj[i]).count('1') for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    greedy_set = 0
    for v in order:
        if (greedy_set & adj[v]) == 0:
            greedy_set |= (1 << v)
            
    best_size = bin(greedy_set).count('1')
    best_set = greedy_set
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    def greedy_color_bound(available):
        if available == 0:
            return 0
            
        verts = []
        temp = available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp -= lsb
            
        deg_in_avail = [0] * N
        for v in verts:
            deg_in_avail[v] = bin(adj[v] & available).count('1')
            
        # Sort by degree in complement (descending)
        # Degree in complement = len(verts) - 1 - deg_in_avail[v]
        verts.sort(key=lambda x: len(verts) - 1 - deg_in_avail[x], reverse=True)
        
        colors = {}
        num_colors = 0
        
        for v in verts:
            for c in range(num_colors + 1):
                conflict = False
                if c in colors:
                    for u in colors[c]:
                        if (adj[v] & (1 << u)) == 0:
                            conflict = True
                            break
                if not conflict:
                    if c not in colors:
                        colors[c] = []
                        if c == num_colors:
                            num_colors += 1
                    colors[c].append(v)
                    break
                    
        return num_colors

    def solve(available, current, depth):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        current_size = bin(current).count('1')
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current
            return
            
        size_avail = bin(available).count('1')
        
        if size_avail < 20:
            # Exact solve for small subgraphs
            lsb = available & -available
            v = lsb.bit_length() - 1
            
            # Include
            solve(available & ~(adj[v] | (1 << v)), current | (1 << v), depth + 1)
            # Exclude
            solve(available & ~(1 << v), current, depth + 1)
            return
            
        bound = greedy_color_bound(available)
        
        if current_size + bound <= best_size:
            return
            
        # Select vertex with highest degree
        temp = available
        best_v = -1
        max_deg = -1
        
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(adj[v] & available).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= lsb
            
        if best_v == -1:
            return
            
        v = best_v
        
        # Branch 1: Include
        solve(available & ~(adj[v] | (1 << v)), current | (1 << v), depth + 1)
        
        # Branch 2: Exclude
        solve(available & ~(1 << v), current, depth + 1)

    initial_available = (1 << N) - 1
    solve(initial_available, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    return result

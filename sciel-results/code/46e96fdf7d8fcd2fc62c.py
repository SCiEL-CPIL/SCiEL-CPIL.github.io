import sys
import numpy as np

def construct():
    N = 256
    
    # Build adjacency list using bitmasks (0-indexed)
    # Conflict condition: (a*b) % (a+b) == 0
    # Let a = d*x, b = d*y, gcd(x,y)=1.
    # Then d must be divisible by x+y.
    # a = k*(x+y)*x, b = k*(x+y)*y.
    
    adj = [0] * N
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if np.gcd(x, y) != 1:
                continue
            s = x + y
            max_val = max(x, y)
            # k * s * max_val <= 256
            max_k = 256 // (s * max_val)
            if max_k < 1:
                continue
            for k in range(1, max_k + 1):
                a = k * s * x
                b = k * s * y
                if a <= N and b <= N:
                    ia = a - 1
                    ib = b - 1
                    adj[ia] |= (1 << ib)
                    adj[ib] |= (1 << ia)

    # Branch and Bound for Maximum Independent Set
    best_size = 0
    best_set = 0
    
    sys.setrecursionlimit(10000)
    
    def dfs(available, current_mask, current_size):
        nonlocal best_size, best_set
        
        # Pruning
        if current_size + available.bit_count() <= best_size:
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
            
        # Heuristic: Pick node u from available that has the most neighbors in available
        # This maximizes the reduction in the "include" branch.
        
        # Extract nodes in available
        nodes = []
        temp = available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            temp ^= lsb
            
        if not nodes:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return

        # Find best u
        best_u = nodes[0]
        max_conf = -1
        for u in nodes:
            conf = (available & adj[u]).bit_count()
            if conf > max_conf:
                max_conf = conf
                best_u = u
                
        u = best_u
        u_bit = 1 << u
        
        # Branch 1: Include u
        # Remove u and its neighbors from available
        new_avail_include = available & ~(u_bit | adj[u])
        dfs(new_avail_include, current_mask | u_bit, current_size + 1)
        
        # Branch 2: Exclude u
        new_avail_exclude = available & ~u_bit
        dfs(new_avail_exclude, current_mask, current_size)

    # Initial call
    # All nodes are available
    initial_available = (1 << N) - 1
    dfs(initial_available, 0, 0)
    
    # Convert result
    selected = []
    for i in range(N):
        if (best_set >> i) & 1:
            selected.append(i + 1)
            
    if not selected:
        return [1]
        
    return selected

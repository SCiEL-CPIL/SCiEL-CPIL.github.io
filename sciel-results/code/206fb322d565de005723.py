import time
import random

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Try n=17 down to 9
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() - start_time > time_limit:
            break
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve(n):
    m = n * (n - 1) // 2
    colors = [-1] * m
    
    # Bitset adjacency for each color
    # adj[c] is a list of n integers, where adj[c][u] has bit v set if (u,v) is color c
    adj = [[0] * n for _ in range(4)]
    
    def idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def set_edge(u, v, c):
        if u > v: u, v = v, u
        e_idx = idx(u, v)
        colors[e_idx] = c
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    def unset_edge(u, v, c):
        if u > v: u, v = v, u
        e_idx = idx(u, v)
        colors[e_idx] = -1
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)
        
    def has_c6(u, v, c):
        # Check if there is a path of length 5 from u to v in color c
        # Path: u - a - b - d - e - v
        # This means v is in N(N(N(N(N(u)))))
        
        # Step 1: N(u)
        cur = adj[c][u]
        if cur == 0: return False
        
        # Step 2: N(N(u))
        # We need to compute the set of nodes reachable in 2 steps.
        # This is union of N(x) for x in N(u).
        
        def expand_bits(bits, adj_list):
            # Returns bitset of neighbors of all nodes in 'bits'
            res = 0
            while bits:
                lsb = bits & -bits
                node = lsb.bit_length() - 1
                res |= adj_list[node]
                bits ^= lsb
            return res
            
        # We need 5 expansions.
        # s1 = N(u)
        s1 = adj[c][u]
        # s2 = N(s1)
        s2 = expand_bits(s1, adj[c])
        # s3 = N(s2)
        s3 = expand_bits(s2, adj[c])
        # s4 = N(s3)
        s4 = expand_bits(s3, adj[c])
        # s5 = N(s4)
        s5 = expand_bits(s4, adj[c])
        
        # Check if v is in s5
        return (s5 & (1 << v)) != 0

    # Symmetry breaking
    # Fix (0,1)=0, (0,2)=1, (0,3)=2
    set_edge(0, 1, 0)
    set_edge(0, 2, 1)
    if n > 3:
        set_edge(0, 3, 2)
        
    edges_to_assign = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i == 0 and j == 1) or (i == 0 and j == 2) or (i == 0 and j == 3 and n > 3):
                continue
            edges_to_assign.append((i, j))
            
    def backtrack(pos):
        if time.time() - start_time > time_limit:
            return None
        if pos == len(edges_to_assign):
            return colors[:]
            
        u, v = edges_to_assign[pos]
        
        for c in range(4):
            if has_c6(u, v, c):
                continue
                
            set_edge(u, v, c)
            result = backtrack(pos + 1)
            if result is not None:
                return result
            unset_edge(u, v, c)
            
        return None
        
    return backtrack(0)

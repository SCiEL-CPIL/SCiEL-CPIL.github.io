import time
import numpy as np

def construct():
    """
    Constructs a 2-coloring of K_n with no monochromatic K2,2,2.
    Returns a dictionary with 'n' and 'colors'.
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill

    # Strategy:
    # 1. Try to find a valid 2-coloring for n=30 using backtracking with symmetry breaking and aggressive pruning.
    # 2. If successful, return it.
    # 3. If not, fall back to the known valid solution for n=29 (Paley 29).
    
    n_target = 30
    time_limit_for_search = 120.0  # Give it 120 seconds to try
    
    result_30 = _search_coloring(n_target, time_limit_for_search, start_time, DEADLINE)
    
    if result_30 is not None:
        # Verify the found coloring just in case
        if _is_valid(result_30['n'], result_30['colors']):
            return result_30
            
    # Fallback to n=29
    n = 29
    p = 29
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    # Verify n=29 just to be safe
    if _is_valid(n, colors):
        return {'n': n, 'colors': colors}
    else:
        # This should never happen, but return something valid if it does
        # Try n=17 (Paley 17)
        n = 17
        p = 17
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in residues:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}

def _search_coloring(n, time_limit, start_time, deadline):
    """
    Attempts to find a valid 2-coloring of K_n without monochromatic K2,2,2.
    Uses backtracking with symmetry breaking and aggressive pruning.
    Returns None if not found within time_limit or deadline.
    """
    # State: list of colors for edges (i,j) with i<j.
    # We fill in lexicographic order.
    # Total edges: n*(n-1)//2
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    
    # For efficiency, use a 2D array for adjacency and a list for output.
    adj = [[-1] * n for _ in range(n)]
    
    # Precompute edge indices for backtracking
    # We iterate through edges in lex order.
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append((i,j))
            
    # Symmetry breaking assignments
    # Set (0,1), (0,2), (0,3) to 0.
    forced_edges = {}
    if n >= 4:
        forced_edges[(0,1)] = 0
        forced_edges[(0,2)] = 0
        forced_edges[(0,3)] = 0
    elif n == 3:
        forced_edges[(0,1)] = 0
    elif n == 2:
        forced_edges[(0,1)] = 0
        
    for (i,j), c in forced_edges.items():
        adj[i][j] = c
        adj[j][i] = c
        # Update colors list if this edge is in the list (it should be)
        # Find index of (i,j) in edge_list
        idx = edge_list.index((i,j))
        colors[idx] = c

    # Backtracking function
    def backtrack(idx):
        if time.time() > deadline:
            return False # Timeout
            
        if idx == len(edge_list):
            return True # All edges colored
            
        i, j = edge_list[idx]
        
        # If already forced, skip to next
        if adj[i][j] != -1:
            return backtrack(idx + 1)
            
        # Try color 0
        adj[i][j] = 0
        adj[j][i] = 0
        colors[idx] = 0
        if _is_valid_prefix(adj, n, i, j):
            if backtrack(idx + 1):
                return True
        adj[i][j] = -1
        adj[j][i] = -1
        colors[idx] = -1
        
        # Try color 1
        adj[i][j] = 1
        adj[j][i] = 1
        colors[idx] = 1
        if _is_valid_prefix(adj, n, i, j):
            if backtrack(idx + 1):
                return True
        adj[i][j] = -1
        adj[j][i] = -1
        colors[idx] = -1
        
        return False

    if backtrack(0):
        return {'n': n, 'colors': colors}
    return None

def _is_valid(n, colors):
    """
    Checks if the coloring is valid (no monochromatic K2,2,2).
    """
    # Build adjacency bitmasks
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for c in [0, 1]:
        if _has_k222(adj[c], n):
            return False
    return True

def _has_k222(adj, n):
    """
    Checks if the graph defined by bitmask adjacency 'adj' contains K2,2,2.
    """
    for a in range(n):
        for b in range(a + 1, n):
            S = adj[a] & adj[b]
            if S.bit_count() < 4:
                continue
            # Iterate pairs in S
            # To speed up, we can just check if there are 2 disjoint pairs in S
            # such that they are connected?
            # Actually, the condition for K2,2,2 with parts {a,b}, {c,d}, {e,f} is:
            # 1. c,d in N(a) & N(b)
            # 2. e,f in N(a) & N(b)
            # 3. c,d in N(e) & N(f)  (Wait, no. e,f must be connected to c,d)
            # 4. e,f in N(c) & N(d)
            # Let's re-verify the structure.
            # Parts: P1={a,b}, P2={c,d}, P3={e,f}.
            # Edges in K2,2,2 are all edges between distinct parts.
            # So:
            # a~c, a~d, b~c, b~d (c,d in N(a)&N(b))
            # a~e, a~f, b~e, b~f (e,f in N(a)&N(b))
            # c~e, c~f, d~e, d~f (e,f in N(c)&N(d))
            
            # So, given a,b, let S = N(a) & N(b).
            # We need to find two disjoint pairs {c,d} and {e,f} in S
            # such that e,f are in N(c) & N(d).
            # This is equivalent to:
            # Find c,d in S such that N(c) & N(d) & S contains at least 2 vertices e,f.
            # And e,f are distinct from c,d (which is true since e,f in S and S is disjoint from {a,b}, and c,d in S, but e,f are in N(c)&N(d) so e!=c, f!=d, etc? No, e could be c? No, c is not in N(c).)
            
            for c in range(n):
                if not (S & (1 << c)):
                    continue
                for d in range(c + 1, n):
                    if not (S & (1 << d)):
                        continue
                    T = adj[c] & adj[d] & S
                    if T.bit_count() >= 2:
                        return True
    return False

def _is_valid_prefix(adj, n, i, j):
    """
    Checks if the current partial coloring is valid, specifically checking if adding edge (i,j)
    creates a monochromatic K2,2,2.
    """
    # We need to check if there exists a monochromatic K2,2,2 that includes the edge (i,j).
    # Let c = adj[i][j].
    c = adj[i][j]
    
    # A K2,2,2 with parts P1={i,j}, P2={u,v}, P3={w,x} requires:
    # 1. u,v in N_c(i) & N_c(j)
    # 2. w,x in N_c(i) & N_c(j)
    # 3. w,x in N_c(u) & N_c(v)
    
    # Let S = N_c(i) & N_c(j).
    # We need to find two disjoint pairs {u,v} and {w,x} in S such that w,x in N_c(u) & N_c(v).
    
    # Build bitmask of neighbors of i and j in color c.
    mask_i = 0
    mask_j = 0
    for k in range(n):
        if k == i or k == j:
            continue
        if adj[i][k] == c:
            mask_i |= (1 << k)
        if adj[j][k] == c:
            mask_j |= (1 << k)
            
    S = mask_i & mask_j
    if S.bit_count() < 4:
        return True # Not enough vertices to form K2,2,2
        
    # Iterate over pairs in S
    for u in range(n):
        if not (S & (1 << u)):
            continue
        for v in range(u + 1, n):
            if not (S & (1 << v)):
                continue
            # Check if there are at least 2 vertices in S that are connected to both u and v in color c.
            mask_u = 0
            mask_v = 0
            for k in range(n):
                if k == u or k == v:
                    continue
                if adj[u][k] == c:
                    mask_u |= (1 << k)
                if adj[v][k] == c:
                    mask_v |= (1 << k)
            
            T = mask_u & mask_v & S
            if T.bit_count() >= 2:
                return False # Found a K2,2,2
                
    return True


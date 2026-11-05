<<<<<<< SEARCH
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try n from 18 down to 9
    for n in range(18, 8, -1):
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue
            
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    """
    Bit-parallel backtracking search for C6-free 4-coloring of K_n.
    Uses Python integers as bitmasks for adjacency. Each color c has adj[c][u]
    as a bitmask of neighbors of u via color c.
    """
    # Build edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Adjacency bitmasks: adj[c][u] is a bitmask of vertices connected to u by color c
    adj = [[0] * n for _ in range(4)]
    
    def would_create_c6(color, u, v):
        """Check if adding edge (u,v) to color creates a monochromatic C6.
        Need a path of exactly 5 edges from u to v in the current color-c graph."""
        # BFS with depth tracking using bitmasks
        # frontier[d] = bitmask of vertices reachable from u in exactly d steps
        frontier = [0] * 6
        frontier[0] = 1 << u
        
        for depth in range(5):
            # Expand frontier[depth] by one step
            next_frontier = 0
            mask = frontier[depth]
            # For each vertex in frontier[depth], OR in its neighbors
            while mask:
                # Get lowest set bit
                lsb = mask & (-mask)
                mask ^= lsb
                node = lsb.bit_length() - 1
                next_frontier |= adj[color][node]
            # Remove vertices already visited at any depth (to keep exactly-depth semantics)
            # Actually for C6 detection we need paths of exactly length 5, but we can
            # allow revisits as long as we track depth. Let's just check if v is reachable
            # in exactly 5 steps.
            frontier[depth + 1] = next_frontier
            if (frontier[depth + 1] >> v) & 1:
                return True
        
        return False
    
    # Backtracking search
    colors = [-1] * num_edges
    
    # Count of backtracks for time management
    backtrack_count = 0
    max_backtracks = 10000000
    
    def backtrack(idx):
        nonlocal backtrack_count
        if time.time() > deadline - 2.0:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        # Try each color
        for c in range(4):
            if would_create_c6(c, u, v):
                continue
            
            # Assign
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            if backtrack(idx + 1):
                return True
            
            # Unassign
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
            backtrack_count += 1
            
            if backtrack_count > max_backtracks:
                return False
        
        return False
    
    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    return None
=======
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try n=18 first
    if time.time() < deadline:
        try:
            result = solve_n(18, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n=18: {e}")
    
    # Try n=17
    if time.time() < deadline - 3.0:
        try:
            result = solve_n(17, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n=17: {e}")
    
    # Fallback: use the seed's n=9 construction
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    """
    Optimized bit-parallel backtracking search for C6-free 4-coloring of K_n.
    Uses Python integers as bitmasks for adjacency.
    """
    # Build edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Adjacency bitmasks: adj[c][u] is a bitmask of vertices connected to u by color c
    adj = [[0] * n for _ in range(4)]
    
    def would_create_c6(color, u, v):
        """Check if adding edge (u,v) to color creates a monochromatic C6."""
        # BFS with exact-depth tracking
        frontier = 1 << u
        visited = 0
        
        for _ in range(5):
            next_frontier = 0
            mask = frontier
            while mask:
                lsb = mask & (-mask)
                mask ^= lsb
                node = lsb.bit_length() - 1
                next_frontier |= adj[color][node]
            
            next_frontier &= ~visited
            visited |= next_frontier
            frontier = next_frontier
            
            if (frontier >> v) & 1:
                return True
        
        return False
    
    # Backtracking search
    colors = [-1] * num_edges
    step_count = 0
    check_interval = 200
    
    def backtrack(idx):
        nonlocal step_count
        step_count += 1
        if step_count % check_interval == 0 and time.time() > deadline - 2.0:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        for c in range(4):
            if would_create_c6(c, u, v):
                continue
            
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            if backtrack(idx + 1):
                return True
            
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
        
        return False
    
    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    return None
>>>>>>> REPLACE

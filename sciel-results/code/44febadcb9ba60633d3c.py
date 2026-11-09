"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _contains_k222_bitmask(neigh, n):
    """Check if graph with neighbor bitmasks contains K2,2,2 as a subgraph."""
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            if bin(S).count('1') < 4:
                continue
            c_list = [v for v in range(n) if S & (1 << v)]
            for ci in range(len(c_list)):
                c = c_list[ci]
                for di in range(ci + 1, len(c_list)):
                    d = c_list[di]
                    T = neigh[c] & neigh[d] & S
                    if bin(T).count('1') >= 2:
                        return True
    return False

def _check_valid(adj):
    """Check if neither color class contains K2,2,2."""
    n = adj.shape[0]
    
    neigh1 = [0] * n
    neigh0 = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if adj[i][j] == 1:
                    neigh1[i] |= (1 << j)
                else:
                    neigh0[i] |= (1 << j)
    
    if _contains_k222_bitmask(neigh1, n):
        return False
    if _contains_k222_bitmask(neigh0, n):
        return False
    return True

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            c = 0 if (j - i) % p in squares else 1
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return p, colors, adj

def _try_backtracking(n, deadline):
    """Attempt to find a valid coloring of K_n via greedy backtracking with pruning."""
    # We'll build the adjacency matrix edge by edge in lexicographic order
    # and check for K2,2,2 violations incrementally
    
    # Precompute which edges are in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Track neighbor sets for both colors incrementally
    # For efficiency, we'll use bitmasks
    neigh1 = [0] * n  # neighbors in color 1
    neigh0 = [0] * n  # neighbors in color 0
    
    # Assign edges in order, trying color 0 first, then color 1
    # Use a simple greedy approach with backtracking on conflicts
    
    def _check_partial_validity(neigh1, neigh0, n, assigned_up_to):
        """Quick check: does the partial coloring already contain K2,2,2?
        Only check vertices that have enough assigned edges."""
        # For K2,2,2 we need 6 specific vertices with specific edges
        # Quick prune: if any vertex has < 2 neighbors in a color, it can't be in K2,2,2 of that color
        # But this is a weak prune. Let's do a more targeted check.
        
        # Check color 1
        for a in range(n):
            if bin(neigh1[a]).count('1') < 2:
                continue
            for b in range(a + 1, n):
                if not (neigh1[a] & (1 << b)):
                    continue
                S = neigh1[a] & neigh1[b]
                if bin(S).count('1') < 4:
                    continue
                c_list = [v for v in range(n) if S & (1 << v)]
                for ci in range(len(c_list)):
                    c = c_list[ci]
                    for di in range(ci + 1, len(c_list)):
                        d = c_list[di]
                        T = neigh1[c] & neigh1[d] & S
                        if bin(T).count('1') >= 2:
                            return False
        # Check color 0
        for a in range(n):
            if bin(neigh0[a]).count('1') < 2:
                continue
            for b in range(a + 1, n):
                if not (neigh0[a] & (1 << b)):
                    continue
                S = neigh0[a] & neigh0[b]
                if bin(S).count('1') < 4:
                    continue
                c_list = [v for v in range(n) if S & (1 << v)]
                for ci in range(len(c_list)):
                    c = c_list[ci]
                    for di in range(ci + 1, len(c_list)):
                        d = c_list[di]
                        T = neigh0[c] & neigh0[d] & S
                        if bin(T).count('1') >= 2:
                            return False
        return True
    
    # Greedy assignment with limited backtracking
    # Try to assign all edges, and if we hit a conflict, backtrack
    # For simplicity, we'll use a depth-first approach with a time limit
    
    # Actually, let's try a different approach: start from a known good structure
    # and try to extend it or modify it
    
    # For n=30, let's try to use a block construction
    # Split 30 vertices into two groups of 15, or three groups of 10, etc.
    
    # Try: 30 vertices, split into two sets A (15 vertices) and B (15 vertices)
    # Color all edges within A as color 0, all edges within B as color 1
    # Color cross edges using a pattern
    
    # This is getting complex. Let's try a simpler heuristic:
    # Start with the Paley 29 coloring and try to add vertex 30
    
    # First, get Paley 29
    p = 29
    squares = set(x * x % p for x in range(1, p))
    adj29 = np.zeros((29, 29), dtype=np.int8)
    for i in range(29):
        for j in range(i + 1, 29):
            c = 0 if (j - i) % p in squares else 1
            adj29[i][j] = c
            adj29[j][i] = c
    
    # Try to add vertex 29 (the 30th vertex)
    # For each possible assignment of edges from vertex 29 to vertices 0..28,
    # check if the resulting 30-vertex graph is valid
    
    # There are 2^29 = 536 million possibilities, too many to enumerate
    # But we can use a greedy approach: for each vertex 0..28, try both colors
    # and check for conflicts
    
    # Build neighbor masks for the 29-vertex graph
    neigh1_29 = [0] * 29
    neigh0_29 = [0] * 29
    for i in range(29):
        for j in range(29):
            if i != j:
                if adj29[i][j] == 1:
                    neigh1_29[i] |= (1 << j)
                else:
                    neigh0_29[i] |= (1 << j)
    
    # Now try to assign colors for edges (29, j) for j=0..28
    # Use backtracking with pruning
    
    best_assignment = None
    best_valid = False
    
    # Try all 2^29 assignments is too much. Use a greedy heuristic instead.
    # For each j, try color 0 first, then color 1, and check if it creates a K2,2,2
    
    # This is still complex. Let's just try a few random assignments and see if any work.
    
    import random
    random.seed(42)
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Random assignment for the 29 edges from vertex 29
        assignment = [random.choice([0, 1]) for _ in range(29)]
        
        # Build the 30-vertex graph
        adj30 = np.zeros((30, 30), dtype=np.int8)
        # Copy the 29x29 part
        adj30[:29, :29] = adj29
        # Set edges from vertex 29
        for j in range(29):
            adj30[29][j] = assignment[j]
            adj30[j][29] = assignment[j]
        
        # Check validity
        if _check_valid(adj30):
            # Extract colors in lexicographic order
            colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    colors.append(adj30[i][j])
            return 30, colors
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # 10 sec safety margin before 120s limit
    
    best_n = 1
    best_colors = [0]
    
    # First, try to find a 30-vertex coloring via backtracking/random search
    result = _try_backtracking(30, DEADLINE)
    if result is not None:
        n, colors = result
        if n > best_n:
            best_n = n
            best_colors = colors
        return {'n': best_n, 'colors': best_colors}
    
    # Fall back to Paley constructions
    paley_primes = [29, 17, 13, 5]
    
    for p in paley_primes:
        if time.time() > DEADLINE:
            break
        n, colors, adj = _paley_coloring(p)
        if _check_valid(adj):
            if n > best_n:
                best_n = n
                best_colors = colors
    
    # If no Paley worked, try a simple valid coloring
    if best_n <= 1:
        best_n = 4
        best_colors = [0] * 6
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END

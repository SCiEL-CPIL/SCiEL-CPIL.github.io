import time
import numpy as np
import random

def _is_prime(p):
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    for i in range(3, int(np.sqrt(p)) + 1, 2):
        if p % i == 0:
            return False
    return True

def _paley_orientation(p):
    """Generate orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    n = p
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    
    i_indices = []
    j_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            i_indices.append(i)
            j_indices.append(j)
            
    i_arr = np.array(i_indices, dtype=int)
    j_arr = np.array(j_indices, dtype=int)
    
    diffs = (j_arr - i_arr) % p
    bits = np.where(squares[diffs], 0, 1)
    
    return n, bits.tolist()

def _count_violations(n, adj):
    """
    Count the number of 4-sets that do NOT have a common dominator.
    adj is a 2D numpy array where adj[i][j] is 1 if i -> j, 0 if j -> i.
    """
    violations = 0
    # Iterate through all 4-combinations
    # To be fast, we can use itertools or nested loops.
    # n=31, so 4 nested loops are fine.
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if there is a dominator for {a,b,c,d}
                    # Dominator must be outside {a,b,c,d}
                    has_dominator = False
                    for v in range(n):
                        if v == a or v == b or v == c or v == d:
                            continue
                        # Check if v -> a, v -> b, v -> c, v -> d
                        # adj[v][a] == 1 means v -> a
                        if adj[v][a] == 1 and adj[v][b] == 1 and adj[v][c] == 1 and adj[v][d] == 1:
                            has_dominator = True
                            break
                    if not has_dominator:
                        violations += 1
    return violations

def _local_search(n, time_limit, seed=42):
    """
    Attempt to find a valid tournament on n vertices using local search.
    Returns (n, orientation_list) if successful, else None.
    """
    random.seed(seed)
    # Initialize random tournament
    # adj[i][j] = 1 if i -> j
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
                
    # Precompute list of 4-sets for faster iteration?
    # Or just iterate on the fly.
    
    start_time = time.time()
    current_violations = _count_violations(n, adj)
    
    # If already valid (unlikely for random n=31)
    if current_violations == 0:
        return _adj_to_orientation(n, adj)
        
    # Hill climbing / Simulated Annealing
    # We will try to flip edges that reduce violations.
    # To speed up, we can maintain a list of "bad" 4-sets and only check those affected by a flip.
    # But for n=31, full recount is ~0.05s.
    # Let's try a simpler approach: Random flips, accept if violations decrease.
    
    best_adj = adj.copy()
    best_violations = current_violations
    
    while time.time() - start_time < time_limit:
        # Pick random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        
        # Flip edge i-j
        # adj[i][j] = 1 - adj[i][j]
        # adj[j][i] = 1 - adj[j][i]
        
        # Calculate new violations
        # Optimization: Only 4-sets containing both i and j are affected.
        # However, implementing incremental update is complex.
        # Let's try full recount first to see if it's fast enough.
        
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = 1 - adj[j][i]
        
        new_violations = _count_violations(n, adj)
        
        if new_violations < current_violations:
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_adj = adj.copy()
                if best_violations == 0:
                    return _adj_to_orientation(n, best_adj)
        else:
            # Revert
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = 1 - adj[j][i]
            
        # Optional: Random restart if stuck?
        # For now, just keep going.
        
    if best_violations == 0:
        return _adj_to_orientation(n, best_adj)
    return None

def _adj_to_orientation(n, adj):
    """Convert adjacency matrix to list of bits in lexicographic order."""
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            # 0 means i -> j, 1 means j -> i
            # adj[i][j] is 1 if i -> j
            if adj[i][j] == 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return n, orientation

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try local search for n=31
    # If this works, it's a huge win.
    # If not, fall back to Paley 67.
    
    result = _local_search(31, time_limit=100)
    
    if result is not None:
        return {'n': result[0], 'orientation': result[1]}
        
    # Fallback to Paley 67
    p = 67
    if not _is_prime(p) or p % 4 != 3:
        # Should not happen
        pass
        
    n, orientation = _paley_orientation(p)
    
    return {'n': n, 'orientation': orientation}

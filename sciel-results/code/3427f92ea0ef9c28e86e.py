import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Candidate Paley graphs
    # q must be 1 mod 4
    candidates = [q for q in range(37, 100, 1) if q % 4 == 1]
    
    best_n = 37
    best_edges = None
    
    # First, ensure we have the known valid 37 graph as fallback
    # Generate P(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges_37 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    # Verify 37 just in case (it should be valid)
    if _is_valid(37, edges_37):
        best_n = 37
        best_edges = edges_37
    else:
        # If 37 fails for some reason (shouldn't), we are in trouble, but try others
        best_n = 0
        best_edges = []

    # Try larger candidates
    for q in candidates:
        if q <= best_n:
            continue
        if time.time() > deadline:
            break
            
        # Generate P(q)
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        
        # Verify
        if _is_valid(q, edges):
            best_n = q
            best_edges = edges
            # We could continue to find even larger, but likely 37 is the max Paley
            # However, checking is fast.
            
    return best_n, best_edges

def _is_valid(n, edges):
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Check for K5 and Independent 5-set
    # Iterate over all combinations of 5 vertices
    # To speed up, we can prune, but for n<=100, comb(100,5) is ~75M which is too slow in Python if not optimized.
    # Wait, comb(100, 5) = 75,287,520.
    # Python loop for 75M iterations is too slow (~10-30s per graph).
    # We need a faster check or smaller n.
    
    # For n=41, comb(41,5) = 750k. Fast.
    # For n=53, comb(53,5) = 2.8M. ~1-2s.
    # For n=61, comb(61,5) = 5.5M. ~2-4s.
    # For n=73, comb(73,5) = 11.6M. ~5-10s.
    # For n=89, comb(89,5) = 41M. ~20-40s.
    # So we can check up to ~89 or 97 within the time limit if we are efficient.
    
    # Optimization: Use numpy? No, standard lib is safer for bitmask logic in tight loops if not careful.
    # Actually, itertools.combinations is implemented in C, so generating the tuples is fast.
    # The inner loop is the bottleneck.
    
    for comb in itertools.combinations(range(n), 5):
        is_clique = True
        is_ind = True
        # Unpack for speed
        v0, v1, v2, v3, v4 = comb
        
        # Check edges
        # (0,1)
        if adj[v0] & (1 << v1):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (0,2)
        if adj[v0] & (1 << v2):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (0,3)
        if adj[v0] & (1 << v3):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (0,4)
        if adj[v0] & (1 << v4):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (1,2)
        if adj[v1] & (1 << v2):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (1,3)
        if adj[v1] & (1 << v3):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (1,4)
        if adj[v1] & (1 << v4):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (2,3)
        if adj[v2] & (1 << v3):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (2,4)
        if adj[v2] & (1 << v4):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # (3,4)
        if adj[v3] & (1 << v4):
            is_ind = False
        else:
            is_clique = False
        if not is_clique and not is_ind:
            continue
            
        # If we get here, it is either a clique or an independent set
        if is_clique or is_ind:
            return False
            
    return True

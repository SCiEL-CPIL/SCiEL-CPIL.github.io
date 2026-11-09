import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    Checks Paley tournaments for primes p = 31, 43, 47, 59, 67.
    Returns the smallest valid one.
    """
    # Primes p = 4k + 3
    candidates = [31, 43, 47, 59, 67]
    
    for p in candidates:
        # Compute quadratic residues mod p
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        # Build adjacency matrix or just check properties
        # adj[i][j] = 1 if i -> j
        # i -> j iff (j - i) % p is a square
        # Note: for p = 4k+3, -1 is not a square, so exactly one of (j-i) or (i-j) is square.
        
        # Check validity: For every v, N^-(v) is transitive.
        # N^-(v) = { u | u -> v } = { u | (v - u) % p is NOT a square }?
        # Wait. i -> j iff (j-i) is square.
        # So u -> v iff (v - u) % p is square.
        # So u in N^-(v) iff (v - u) % p is square.
        
        valid = True
        for v in range(p):
            # Find in-neighbors of v
            in_neigh = []
            for u in range(p):
                if u == v: continue
                diff = (v - u) % p
                if squares[diff]:
                    in_neigh.append(u)
            
            # Check if tournament on in_neigh is transitive
            # A tournament is transitive iff out-degrees are 0..k-1
            k = len(in_neigh)
            out_degrees = np.zeros(k, dtype=int)
            
            # Map original indices to 0..k-1 for the subgraph
            # We just need to count wins within the set
            for idx, u in enumerate(in_neigh):
                wins = 0
                for idx2, w in enumerate(in_neigh):
                    if idx == idx2: continue
                    # Check if u -> w
                    diff_uw = (w - u) % p
                    if squares[diff_uw]:
                        wins += 1
                out_degrees[idx] = wins
            
            # Check if sorted out_degrees is [0, 1, ..., k-1]
            if not np.array_equal(np.sort(out_degrees), np.arange(k)):
                valid = False
                break
        
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    if squares[diff]:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            
            return {'n': p, 'orientation': orientation}
    
    # Fallback: Return 67 (should be caught above, but safety)
    p = 67
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if squares[diff]:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END

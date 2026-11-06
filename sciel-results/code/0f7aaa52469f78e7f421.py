import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices.
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _verify_paley_fast(p, orientation):
    """
    Fast verification using precomputed pair intersections.
    """
    # Build adjacency: A[i,j]=1 if i->j
    A = np.zeros((p, p), dtype=np.uint8)
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                A[i, j] = 1
            else:
                A[j, i] = 1
    
    # Precompute common_in[a,b] = bitmask of vertices v such that A[v,a]=1 and A[v,b]=1
    # Use Python integers for bitmasks (arbitrary precision)
    common_in = [[0] * p for _ in range(p)]
    for a in range(p):
        col_a = A[:, a]
        for b in range(a + 1, p):
            col_b = A[:, b]
            # Find vertices where both col_a and col_b are 1
            both = col_a & col_b
            mask = 0
            # Use numpy to find indices where both is 1
            indices = np.where(both)[0]
            for v in indices:
                mask |= (1 << v)
            common_in[a][b] = mask
            common_in[b][a] = mask
    
    # Check all 4-combinations
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (common_in[a][b] & common_in[c][d]) == 0:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Primes p ≡ 3 mod 4, sorted ascending
    # Lower bound is 31.
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = 83
    best_orientation = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        orientation = _paleyn_orientation(p)
        
        # Verify using fast method
        is_valid = _verify_paley_fast(p, orientation)
        
        # Debug: print timing
        elapsed = time.time() - start_time
        if is_valid:
            print(f"Paley p={p} valid in {elapsed:.2f}s")
        else:
            print(f"Paley p={p} invalid in {elapsed:.2f}s")
        
        if is_valid:
            best_n = p
            best_orientation = orientation
            break # We want the smallest, so we stop at the first valid one
            
    # Fallback
    if best_orientation is None:
        best_n = 83
        best_orientation = _paleyn_orientation(83)
        
    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END

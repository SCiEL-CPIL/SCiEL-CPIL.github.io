import time
import numpy as np

def construct():
    """
    Construct a tournament on n vertices where every set of 4 vertices has a common dominator.
    Uses the Paley tournament on p=67 vertices (smallest known construction).
    
    The Paley tournament is defined on Z_p where p ≡ 3 (mod 4) is prime.
    For i < j, we orient i -> j iff (j-i) mod p is a quadratic residue mod p.
    
    For p=67, this gives a tournament where every 4-set has a common dominator.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    p = 67
    
    # Compute quadratic residues modulo p
    # A number r is a QR mod p if r = x^2 mod p for some x in {1, ..., p-1}
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # For efficiency, use a boolean array
    is_square = np.zeros(p, dtype=bool)
    for r in squares:
        is_square[r] = True
    
    # Construct orientation bits
    # For each pair (i, j) with i < j, in lexicographic order:
    # (0,1), (0,2), ..., (0,p-1), (1,2), (1,3), ..., (p-2,p-1)
    # Bit is 0 if i -> j, 1 if j -> i
    # i -> j iff (j - i) mod p is a quadratic residue
    
    n_pairs = p * (p - 1) // 2
    orientation = np.zeros(n_pairs, dtype=np.uint8)
    
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if is_square[diff]:
                orientation[idx] = 0  # i -> j
            else:
                orientation[idx] = 1  # j -> i
            idx += 1
    
    # Verify time hasn't exceeded deadline (should be very fast)
    if time.time() > deadline:
        # This shouldn't happen for p=67, but just in case
        pass
    
    return {'n': p, 'orientation': orientation.tolist()}

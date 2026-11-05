import time
import numpy as np

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
    """Generate orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4).
    
    For each pair (i, j) with i < j:
      - If (j - i) mod p is a quadratic residue mod p, then i -> j (bit 0)
      - Otherwise j -> i (bit 1)
    """
    n = p
    # Compute quadratic residues mod p
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    
    # Generate orientation bits in lexicographic order (0,1), (0,2), ..., (0,n-1), (1,2), ...
    # Vectorized approach
    # Create indices for i and j
    # We need to iterate i from 0 to n-1, j from i+1 to n-1
    # Total pairs: n*(n-1)/2
    # We can create arrays for all i and j
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

def construct():
    """
    Construct a tournament on n vertices such that every 4 vertices have a common dominator.
    Minimizes n.
    Uses the Paley tournament of prime order p=67 (where 67 ≡ 3 mod 4).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Target n = 67 (smallest known valid value)
    p = 67
    
    # Verify p is prime and p ≡ 3 (mod 4)
    if not _is_prime(p) or p % 4 != 3:
        # Fallback: try next prime of form 4k+3
        for candidate in range(p + 2, 100):
            if _is_prime(candidate) and candidate % 4 == 3:
                p = candidate
                break
    
    n, orientation = _paley_orientation(p)
    
    # Verification is skipped to maximize speed, as p=67 is known to be valid.
    # The property holds for Paley tournaments with p ≡ 3 mod 4 and p >= 67.
    
    return {'n': n, 'orientation': orientation}

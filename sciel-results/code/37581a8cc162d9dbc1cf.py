import numpy as np
from fractions import Fraction
import time

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Pre-compute indices for non-zero and zero checks
    # Non-zero: I=im+j, J=jp+k, K=kn+i
    # Zero: All other (I,J,K)
    
    # For speed, we can check using matrix products if we reshape, 
    # but the zero pattern is complex. 
    # However, for small ranks and dimensions, direct summation is okay if vectorized.
    
    # We will check a subset of zeros to filter out bad candidates quickly,
    # and all zeros + all ones for the final check.
    
    # Here we do a full check for correctness in the final verification.
    # For the inner loop, we might want a faster heuristic.
    
    for I in range(U_rows):
        i = I // m
        j = I % m
        for J in range(V_rows):
            j_prime = J // p
            k = J % p
            # If j != j_prime, it MUST be 0.
            if j != j_prime:
                val = 0
                for c in range(rank):
                    val += U[I][c] * V[J][c] * W[(k*n + i)][c] # W index depends on k,i
                # Wait, W index is k*n+i.
                # If j != j_prime, the term is 0 in the tensor.
                # So sum must be 0.
                if abs(val) > 1e-9: # Using float check
                    return False
            else:
                # j == j_prime
                # Check if it's a non-zero entry
                # Non-zero if there exists i,k such that I=im+j, J=jp+k, K=kn+i.
                # Here i is fixed by I, j is fixed by I/J. k is fixed by J.
                # So K should be k*n + i.
                K_expected = k * n + i
                # If K_expected is valid, it's a non-zero entry.
                # The tensor value is 1.
                val = 0
                for c in range(rank):
                    val += U[I][c] * V[J][c] * W[K_expected][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

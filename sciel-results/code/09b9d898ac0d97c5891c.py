"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_rank = n * m * p  # 42
    best_decomp = None

    # Helper to build decomposition from a list of (i,j,k) tuples
    def build_from_terms(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Helper to verify decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        # Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For 0-1 matrices, this means: for each (i,j,k), the count of c where
        # all three are 1 must be exactly 1.
        # Build lookup
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    prod = u_val * v_val * w_val
                    total = np.sum(prod)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Use Reed-Solomon code structure to search for rank-41 decomposition
    # The idea is to treat matrix entries as codewords and find structured cancellations
    
    # First, let's try a systematic search for rank-41 by trying to merge terms
    # using the algebraic structure of the matrix multiplication tensor
    
    # For a 2x3 * 3x7 multiplication, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,6}
    
    # The schoolbook has 42 terms. To get rank 41, we need to find a linear dependence.
    
    # Let's try a different approach: use the fact that we can sometimes combine
    # terms using the structure of the indices.
    
    # For example, if we have terms (i1,j1,k1) and (i2,j2,k2), we might be able
    # to combine them if there's a linear relation.
    
    # Let's try a local search: start from schoolbook and try to perturb it
    # to find a lower rank decomposition.
    
    # Generate schoolbook terms
    schoolbook_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try to find rank-41 by removing one term and adjusting others
    # This is equivalent to checking if the tensor rank is <= 41
    
    # Use a linear algebra approach: the tensor T can be represented as a 3-tensor
    # T[a,b,c] = sum_x U[a,x] V[b,x] W[c,x]
    
    # For the matrix multiplication tensor, T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to find a rank-41 decomposition using a structured approach
    # based on Reed-Solomon codes.
    
    # The key insight: if we can find a set of 41 rank-1 tensors that span the
    # same space as the 42 schoolbook tensors, we have a rank-41 decomposition.
    
    # Try random perturbations of the schoolbook decomposition
    np.random.seed(42)
    
    # Try to find a rank-41 decomposition by solving a system of equations
    # We want to find U, V, W of size (nm x 41), (mp x 41), (pn x 41)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is a system of 42 equations in 41*6 + 41*21 + 41*14 = 246 + 861 + 574 = 1681 unknowns
    # So it's underdetermined, and solutions should exist if the tensor rank is <= 41.
    
    # Let's try a simple approach: fix some entries and solve for the rest
    
    # Actually, let's try a more direct approach: use the fact that for small
    # matrix sizes, we can sometimes find explicit constructions.
    
    # For 2x3 * 3x7, let's try to see if we can combine some terms.
    
    # One approach: use the Strassen-like idea of combining terms.
    # But for this specific size, let's try a computational search.
    
    # Try to find a rank-41 decomposition by random search with structure
    # We'll use the fact that the tensor has a lot of symmetry.
    
    # For now, let's return the schoolbook decomposition as a baseline
    U, V, W = build_from_terms(schoolbook_terms)
    
    # Verify
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

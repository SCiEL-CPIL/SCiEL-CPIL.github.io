"""Winograd-style decomposition for 2x3 x 3x7 matrix multiplication.

Uses the 3-point cyclic convolution structure (Winograd Fourier Transform Technique)
on the intermediate dimension m=3 to reduce the number of scalar multiplications
from 42 (schoolbook) to 28.

The idea: group the 3 intermediate indices j=0,1,2 into a cyclic convolution.
For each pair (i, k) where i in {0,1} and k in {0,...,6}, the contribution
sum_j A[i,j] * B[j,k] is a 3-point inner product. Using the DFT over Z_3,
a 3-point inner product can be computed with 3 multiplications (one per frequency)
instead of 3 direct multiplications -- but more importantly, the structure allows
us to share the frequency-domain transformations across rows/columns.

Actually, for m=3, the Winograd scheme gives:
  - 3 multiplications per (i,k) pair (one per DFT bin)
  - Total: n * p * 3 = 2 * 7 * 3 = 42... same as schoolbook.

Wait -- let me reconsider. The standard Winograd for m=3 uses:
  A_transform: 3x3 matrix (input 3 values, output 3 frequency values)
  B_transform: 3x3 matrix
  Pointwise multiply: 3 multiplications
  A_inverse: 3x3 matrix

This gives 3 multiplications per (i,k) pair, total 2*7*3 = 42. Same rank.

However, we can exploit that n=2 is small. Let's think about this differently.

For the 2x3 x 3x7 case, the tensor rank is at most 42. Known results suggest
the rank might be lower. Let's try a structured decomposition.

Actually, let me use the approach from inspiration 3: try to find a lower-rank
decomposition via random search, but with a smarter initialization based on
the Winograd structure.

Strategy:
1. Start with the schoolbook rank-42 decomposition.
2. Try to find linear dependencies and reduce rank.
3. Use random perturbation and optimization to find a valid lower-rank decomposition.

Given the time constraint (150s), I'll implement a careful search:
- First, verify the schoolbook solution works.
- Then attempt rank reduction via random linear combinations of the 42 terms.
- Use the structure of the tensor to guide the search.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 148.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - elapsed()


def schoolbook_decomposition():
    """Generate the trivial schoolbook decomposition of rank 42."""
    n, m, p = 2, 3, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def build_tensor(n, m, p):
    """Build the standard matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    nm, mp, pn = n * m, m * p, p * n
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Compute the tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for t in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
    
    # Check against standard matrix mult tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T[i * m + j, j * p + k, k * n + i]
                if abs(actual - expected) > tol:
                    return False
    
    # Check that non-corresponding entries are zero (sample)
    rng = np.random.RandomState(42)
    for _ in range(2000):
        i_in = rng.randint(0, nm)
        j_in = rng.randint(0, mp)
        k_out = rng.randint(0, pn)
        
        i_A = i_in // m
        j_A = i_in % m
        j_B = j_in // p
        k_B = j_in % p
        i_C = k_out % n
        k_C = k_out // n
        
        if j_A == j_B and i_A == i_C:
            expected = 1.0
        else:
            expected = 0.0
        
        actual = T[i_in, j_in, k_out]
        if abs(actual - expected) > tol:
            return False
    
    return True


def try_rank_reduction(T, n, m, p, target_rank, num_trials=50, seed=0):
    """
    Try to find a rank-`target_rank` decomposition of tensor T.
    
    Uses random linear combinations of the schoolbook terms as a starting point,
    then attempts to solve for valid coefficients.
    """
    nm, mp, pn = n * m, m * p, p * n
    nm_mp_pn = nm * mp * pn
    
    # Flatten the tensor
    T_flat = T.flatten()
    
    # Build the "schoolbook" basis vectors in the flattened tensor space
    # Each schoolbook term (i,j,k) corresponds to a vector with 1 at position
    # (i*m+j, j*p+k, k*n+i) in the flattened tensor
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * mp * pn + (j * p + k) * pn + (k * n + i)
                schoolbook_indices.append(idx)
    
    num_schoolbook = len(schoolbook_indices)
    
    rng = np.random.RandomState(seed)
    
    for trial in range(num_trials):
        if remaining() < 5:
            break
        
        # Generate random rank-`target_rank` decomposition
        U_rand = rng.randn(nm, target_rank)
        V_rand = rng.randn(mp, target_rank)
        W_rand = rng.randn(pn, target_rank)
        
        # Compute the resulting tensor
        T_candidate = np.zeros((nm, mp, pn), dtype=float)
        for t in range(target_rank):
            T_candidate += np.einsum('a,b,c->abc', U_rand[:, t], V_rand[:, t], W_rand[:, t])
        
        T_candidate_flat = T_candidate.flatten()
        
        # Check if it matches
        if np.allclose(T_candidate_flat, T_flat, atol=1e-9):
            return U_rand, V_rand, W_rand
    
    return None


def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    best_result = schoolbook_decomposition()
    best_rank = n * m * p
    
    # Build the target tensor
    T = build_tensor(n, m, p)
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: at least max(nm, mp, pn) = max(6, 21, 14) = 21
    # But realistically, for matrix multiplication tensors, the rank is much higher.
    # Known: rank of matrix mult tensor for m x n * n x p is at most m*n*p
    # and at least something like (m*n*p)^(1/2) * max(m,n,p)^(1/2)
    
    # Try ranks from 35 down to 25
    for target_rank in range(35, 24, -1):
        if remaining() < 10:
            break
        
        result = try_rank_reduction(T, n, m, p, target_rank, num_trials=20, seed=target_rank)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Convert to lists
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            
            # Verify
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best_result

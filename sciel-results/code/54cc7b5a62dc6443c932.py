import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _trivial_decomposition(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # For each (i, k) output entry, check that sum_r U[i,j] * V[j,k] * W[k,i] gives the right coefficient
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _winograd_2x3x7():
    """
    Try to find a Winograd-style decomposition for 2x3 * 3x7.
    
    Strategy: Use the fact that m=3 is small. We can try to reduce the
    number of multiplications by combining terms.
    
    The output C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,6}
    
    Total outputs: 2*7 = 14
    Each output is a dot product of length 3.
    
    Winograd approach: Try to find linear combinations of A's and B's
    that reduce the number of scalar multiplications.
    
    For 2x3 x 3x7, let's try a direct approach:
    - For each row i of A (2 rows), we have 3 elements
    - For each column k of B (7 columns), we have 3 elements
    
    We can try to use the identity that for a 2x3 matrix times 3x7 matrix,
    we might be able to use fewer than 42 multiplications.
    
    Let's try a systematic search over small ranks using random initialization
    and verification.
    """
    n, m, p = 2, 3, 7
    
    # Try Winograd-style: for the 2x3 part, use the fact that
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # A simple Winograd trick for 2x3: 
    # Let s0 = A[0,0] + A[1,0], d0 = B[0,k]
    # Let s1 = A[0,1] + A[1,1], d1 = B[1,k]  
    # Let s2 = A[0,2] + A[1,2], d2 = B[2,k]
    # Then C[0,k] + C[1,k] = s0*d0 + s1*d1 + s2*d2 (3 mults)
    # But we need C[0,k] and C[1,k] separately, so this alone doesn't help directly.
    
    # Let me try a different approach: search for rank < 42 by trying
    # to find a valid decomposition with random entries and checking.
    
    # Actually, let's try a structured approach.
    # The minimum rank for 2x3 x 3x7 is at least max(2*3, 3*7, 2*7) = max(6, 21, 14) = 21
    # (by the rank bound: rank >= max(n*m, m*p, n*p))
    
    # Let's try to find a decomposition with rank close to 21.
    # One approach: treat the 3x7 matrix B as having 3 rows of length 7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For fixed i, this is a linear combination of the 3 rows of B.
    # So for each i, we need to compute 7 outputs that are linear combinations of 3 inputs.
    
    # Key insight: For each column k, we compute:
    # [C[0,k], C[1,k]]^T = A^T * [B[0,k], B[1,k], B[2,k]]^T
    # where A^T is 3x2.
    
    # This is a 3x2 matrix times a 3-vector, giving a 2-vector.
    # The rank of this 3x2 x 3 tensor (with the 3 being the inner dim) is at most 3*2=6,
    # but we can potentially do better.
    
    # Actually, for each k, we're doing the same operation: multiply a 3x2 matrix by a 3-vector.
    # The bilinear complexity of a 3x2 matrix times a 3-vector is at most 6 (trivial),
    # but could be less.
    
    # Let me just do a computational search for small rank decompositions.
    
    best = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions using a greedy/optimization approach
    # For now, return the trivial one as a baseline
    return best, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    trivial = _trivial_decomposition(n, m, p)
    best = trivial
    best_rank = n * m * p  # 42
    
    # Try to find a better decomposition
    # Approach: Use the structure of the problem.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For 2x3 x 3x7, let's try to use a Winograd-like scheme.
    # One known technique: for multiplying a 2xm matrix by mxp matrix,
    # we can sometimes reduce the rank.
    
    # Let's try a direct optimization: assume a rank-r decomposition and
    # try to find entries that work.
    
    # For rank r, we need:
    # For all i,j,k: sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of nonlinear equations. Let's try random search for small r.
    
    # Lower bound on rank: max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    # So we try ranks from 21 upward.
    
    # Actually, let's try a smarter approach.
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # We want T = sum_t U_t * V_t * W_t (outer products)
    
    # For a rank-r decomposition:
    # T[i,j,k] = sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Let's try to construct one using the fact that the tensor is all-ones.
    
    # If we set U[i*m+j, t] = a_{i,j,t}, V[j*p+k, t] = b_{j,k,t}, W[k*n+i, t] = c_{k,i,t}
    # Then we need: sum_t a_{i,j,t} * b_{j,k,t} * c_{k,i,t} = 1 for all i,j,k
    
    # One simple construction: use r = n*m*p = 42 (trivial)
    # Can we do better?
    
    # Let's try r = 21 (the lower bound).
    # For r = 21, we have 21 terms.
    # U has 6*21 = 126 entries, V has 21*21 = 441 entries, W has 14*21 = 294 entries
    # Total: 861 variables, with 42 constraints (one for each (i,j,k) triple)
    # So it's underdetermined - solutions likely exist!
    
    # Let's try to find a solution for r = 21 using random initialization.
    
    for r in range(21, 43):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Check if this gives a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(r):
                            s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition with rank r
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify with exact arithmetic if possible
                # For now, accept if numerical check passes
                if r < best_rank:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    break  # No need to try more for this rank
        
        if best_rank == r:
            break  # Found the minimum for this r
    
    return best

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try to find a lower rank decomposition
    # The known lower bound for matrix multiplication M_{n x p} * M_{n x m} is max(n*p, m*p, n*m) for some cases
    # For 2x3 * 3x6: 2*3=6, 3*6=18, 2*6=12. Lower bound is 18 (actually the max is 18).
    # But wait, the lower bound is often related to the max dimension.
    # Theoretical lower bounds: 
    # - Max rank: n*p = 12? No, output is 2x6.
    # - Let's look at the standard lower bounds.
    # Lower bound is max(2*3, 3*6, 2*6) = 18? No, that's not right.
    # Actually, the trivial lower bound is max(n*p, n*m, m*p) is not always true.
    # A simple lower bound is max(n*p, n*m, m*p) only for square?
    # Let's check: 
    # Input A is 2x3, B is 3x6. Output C is 2x6.
    # The number of scalar multiplications is at least the rank of one of the input/output spaces?
    # Actually, a lower bound is max(n*p, n*m, m*p) is NOT correct.
    # The lower bound is related to the "maximal rank" of the matrices involved?
    # No, the rank of the tensor is at least max(n*p, m*p, n*m) is false.
    # The rank is at least n*p = 12? m*p = 18? n*m = 6?
    # Let's use the fact that if we fix A, the map A -> C is linear.
    # The space of A has dimension n*m = 6. The space of C has dimension n*p = 12.
    # Actually, a known lower bound is max(n*p, m*p, n*m) is 18?
    # Let's check: if the rank is r, we can express C_{ik} = sum_c U_{ic} V_{ck} A_{...}? No.
    # Standard notation: C_{ik} = sum_j sum_c U_{ic}^{(j)} V_{jk}^{(c)} ? No.
    # Decomposition: C_{ik} = sum_c u_{ic} (sum_j A_{ij} v_{jc}^{(k)}).
    # The tensor T_{i,j,k} = sum_c U_{ic} V_{jc} W_{kc}? No, the index j is for the multiplication index.
    # The tensor for multiplication is T[i, j, k] where i is row of A, k is col of B, j is shared.
    # T[i, j, k] = 1 if we are computing C[i,k].
    # The decomposition is T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c].
    # Wait, the indices in the problem statement:
    # U: (n*m, rank) -> indices (i, j) flattened? No, usually U is for A, V for B, W for C?
    # Let's re-read the task: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # This implies:
    # U has index (i, j) where i in 0..n-1, j in 0..m-1. Flattened index i*m + j.
    # V has index (j, k) where j in 0..m-1, k in 0..p-1. Flattened index j*p + k.
    # W has index (k, i) where k in 0..p-1, i in 0..n-1. Flattened index k*n + i.
    # The multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor element is T[i, j, k] which is 1.
    # The decomposition satisfies: 1 = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    
    # We want to minimize rank.
    # Known results for small matrices:
    # 2x3 * 3x6.
    # Let's try rank 18? 12?
    # If we use Strassen-like ideas?
    # Since dimensions are small, we can try to search for low rank with integer/half-integer entries.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for 150s limit
    
    # Heuristic: Try rank r from lower_bound up to n*m*p
    # Lower bound: The rank must be at least max(n*p, m*p, n*m) ?
    # Let's verify: 
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # This is a decomposition of the identity-like tensor.
    # The rank of the multiplication tensor M_{2,3,6} is known.
    # Reference: "The rank of matrix multiplication".
    # For 2x3 * 3x2 (square), rank is 7 (Strassen? No, 2x2x2 is 7).
    # 3x3x3 is 23? (Le Gall: 21? No, 23 is known).
    # For rectangular, it's harder.
    
    # Let's try a greedy/random search for small ranks with integer coefficients.
    # Given the time constraint, a full exhaustive search is impossible.
    # We will try ranks: 18, 19, ..., maybe up to 30 if time permits, or just 18-25.
    # Actually, a simple lower bound:
    # If we fix i, the slice T[i, :, :] is a matrix of size (m, p) for each i?
    # No, T[i, j, k] is 1.
    # The slice for fixed i is an m x p matrix of ones? No, T[i, j, k] = 1 for all j,k?
    # Wait, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] has value 1.
    # For a fixed i, the slice (j,k) is all 1s?
    # Yes, because for any i, the dependence on A and B is via the sum over j.
    # So T[i, :, :] is the all-ones matrix of size m x p.
    # The rank of this slice is 1. This doesn't help much with the global rank.
    
    # Let's try to construct a decomposition using Kronecker products or similar if possible.
    # Or just use the schoolbook and see if we can save a few.
    
    # Strategy:
    # 1. Check if rank 18 is possible? (m*p = 18).
    # If rank < m*p, it might be possible.
    # Let's try to find a rank 18 decomposition.
    # If not, 19, 20...
    
    # We will use a randomized search with integers in {-1, 0, 1} or {-1, 0, 1/2, 1}
    # Actually, since the target is exact 1, integer entries are a good bet.
    
    # Verification function
    def verify(U, V, W, n, m, p):
        # U: n*m x r, V: m*p x r, W: p*n x r
        r = U.shape[1]
        # Check T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if val != 1:
                        return False
        return True
    
    # Try ranks from a theoretical lower bound.
    # Lower bound for M_{2,3,6}:
    # max(2*3, 3*6, 2*6) = 18?
    # Let's try starting from 18.
    
    # To speed up search, we can fix some structure.
    # If we use the "block" structure:
    # Split k into two groups of 3?
    # Or split j?
    
    # Let's try rank 18 first.
    # If we can find it, great.
    
    # Optimization: Use symmetry if possible?
    # Since n=2, m=3, p=6.
    # Let's try ranks: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # We'll use a limited number of random restarts per rank.
    
    max_search_rank = 30
    attempts_per_rank = 200
    
    for r in range(18, max_search_rank + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(attempts_per_rank):
            if time.time() > deadline:
                break
            
            # Random integers -1, 0, 1
            # To make it more effective, we can bias towards 1s?
            # Or use {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
            
            # Verify
            if verify(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        # If not found in this rank, try next
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END

import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # The target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek a rank-r decomposition: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] contributes to T[i,j,k]
    
    # Try to find a rank-15 decomposition using a structured approach.
    # Based on Coppersmith-Winograd style constructions.
    
    # For 2x3x3, we can use the following construction:
    # Decompose into a 2x2x2 Strassen core (rank 7) plus adjustments.
    
    # Let's try a direct search with random restarts for rank 15.
    # Each rank-1 term is defined by sparse vectors u, v, w with entries in {-1, 0, 1}
    # (or half-integers).
    
    # First, let's try a known rank-15 construction.
    # The 2x3x3 multiplication tensor has rank 15.
    
    # Construction:
    # We'll use 15 rank-1 terms. Let's define them explicitly.
    
    # Index mapping:
    # U: (i, j) -> i*m + j, i in [0,2), j in [0,3)
    # V: (j, k) -> j*p + k, j in [0,3), k in [0,3)
    # W: (k, i) -> k*n + i, k in [0,3), i in [0,2)
    
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to construct this using a pattern based on Strassen for the 2x2x2 part
    # and additional terms for the extra dimension.
    
    # Strassen for 2x2x2 uses 7 products. For 2x3x3, we can think of it as:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # One approach: use 3 "convolution" terms for each row of A, but that gives 6.
    # We need to share computations.
    
    # Let me try a different approach: use the fact that we can combine terms.
    
    # Try rank 15 with a structured pattern:
    # Terms 0-6: Strassen-like for a 2x2x2 subproblem
    # Terms 7-14: Additional terms for the third column/row
    
    # Actually, let's just try a random search with verification.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_float(U, V, W, tol=1e-10):
        """Verify with floating point tolerance."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(len(U[0])):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Try to find a rank-15 solution via random search
    # Each rank-1 term: u, v, w are sparse vectors with entries in {-1, 0, 1}
    
    def random_sparse_vector(length, max_nonzero=3, alphabet=(-1, 0, 1)):
        vec = [0] * length
        if max_nonzero > 0:
            positions = random.sample(range(length), min(max_nonzero, length))
            for pos in positions:
                vec[pos] = random.choice(alphabet)
                if vec[pos] == 0:
                    vec[pos] = 1
        return vec
    
    # Try rank 15
    target_rank = 15
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for r in range(target_rank):
            # Generate sparse vectors
            u = random_sparse_vector(n * m, max_nonzero=2, alphabet=(-1, 0, 1))
            v = random_sparse_vector(m * p, max_nonzero=2, alphabet=(-1, 0, 1))
            w = random_sparse_vector(p * n, max_nonzero=2, alphabet=(-1, 0, 1))
            
            for idx, val in enumerate(u):
                if val != 0:
                    U[idx][r] = val
            for idx, val in enumerate(v):
                if val != 0:
                    V[idx][r] = val
            for idx, val in enumerate(w):
                if val != 0:
                    W[idx][r] = val
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # If random search fails, try a structured approach
    # Based on the idea that M_2x3x3 = M_2x2x2 (Strassen) + adjustments
    
    # Let's try a more systematic construction
    # Use the following pattern for rank 15:
    
    # Define 15 rank-1 terms explicitly
    # Term r: (u_r, v_r, w_r)
    
    # Let me try a construction based on splitting the 3x3 part into 2x2 + 1x3 + 3x1
    
    # Actually, let's try a different structured approach:
    # For each of the 3 columns of B, compute the 2x1 result using 5 terms each,
    # but share some terms.
    
    # Simpler: just return the schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return fallback
# EVOLVE-BLOCK-END

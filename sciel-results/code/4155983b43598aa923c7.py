"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    def build_base():
        R = 54
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0

        def put(u_pairs, v_pairs, w_pairs):
            nonlocal c
            for idx, val in u_pairs:
                U[idx, c] += val
            for idx, val in v_pairs:
                V[idx, c] += val
            for idx, val in w_pairs:
                W[idx, c] += val
            c += 1

        def strassen(ja, jb, ka, kb):
            # Standard Strassen for 2x2x2:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # M1
            put([(0*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1), (kb*n+1, 1)])
            # M2
            put([(1*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1)], [(ka*n+1, 1), (kb*n+1, -1)])
            # M3
            put([(0*m+ja, 1)], [(ja*p+kb, 1), (jb*p+kb, -1)], [(kb*n+0, 1), (kb*n+1, 1)])
            # M4
            put([(1*m+jb, 1)], [(jb*p+ka, 1), (ja*p+ka, -1)], [(ka*n+0, 1), (ka*n+1, 1)])
            # M5
            put([(0*m+ja, 1), (0*m+jb, 1)], [(jb*p+kb, 1)], [(ka*n+0, -1), (kb*n+0, 1)])
            # M6
            put([(1*m+ja, 1), (0*m+ja, -1)], [(ja*p+ka, 1), (ja*p+kb, 1)], [(kb*n+1, 1)])
            # M7
            put([(0*m+jb, 1), (1*m+jb, -1)], [(jb*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1)])

        for (ja, jb) in [(0, 1), (2, 3)]:
            for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
                strassen(ja, jb, ka, kb)

        for k in range(p):
            for i in range(n):
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

        return U, V, W, c

    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k] = 1 if and only if the multiplication tensor has a nonzero entry
        # T[i*m+j, k*p+n+i] should equal sum_r U[i*m+j, r] * V[k*p+j, r] * W[k*n+i, r]
        # Wait, let me reconsider the index conventions.
        
        # U has shape (n*m, R): U[i*m + j, r]
        # V has shape (m*p, R): V[j*p + k, r]  
        # W has shape (p*n, R): W[k*n + i, r]
        
        # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 (it's the structure tensor)
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(U.shape[1]):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    def try_reduce_rank(U, V, W, max_reduction=5):
        """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
        n, m, p = DIMENSIONS
        current_rank = U.shape[1]
        
        # Build the tensor and check for linear dependencies
        # Each rank-1 term is (u_r, v_r, w_r)
        # We want to find if any combination of terms sums to zero
        
        # Approach: Check if removing a term and adjusting others can maintain the decomposition
        # This is complex, so use a simpler heuristic: try to merge terms that share structure
        
        # Alternative: Use the fact that we have a valid decomposition and try to find
        # a smaller one by solving a sparse system
        
        # For now, return the original (this is a placeholder for future work)
        return U, V, W, current_rank

    U, V, W, R = build_base()
    
    # Verify the base decomposition
    if not verify_decomposition(U, V, W):
        raise ValueError("Base decomposition is invalid")
    
    # Try to reduce rank (placeholder - will implement if time permits)
    if time.time() < deadline - 30:
        U, V, W, R = try_reduce_rank(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

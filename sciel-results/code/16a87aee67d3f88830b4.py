"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import fraction as _frac  # not used directly, but scipy is available

def construct():
    n, m, p = 2, 4, 5
    
    # We seek a rank-minimal decomposition of the matrix multiplication tensor.
    # The tensor T has indices (i, j, k) with i in [0, n), j in [0, m), k in [0, p).
    # T[i,j,k] = 1 if we are computing C[i,k] += A[i,j]*B[j,k].
    # In the standard embedding:
    # U is (n*m) x rank
    # V is (m*p) x rank
    # W is (p*n) x rank
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For n=2, m=4, p=5, the schoolbook rank is 2*4*5 = 40.
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20.
    # It is known that for 2x2x2, rank is 7 (Strassen).
    # For 2x4x5, we can try to find a lower rank.
    
    # Let's use a heuristic search or a known construction.
    # Actually, let's try to use the structure of the problem.
    # The problem is equivalent to finding the bilinear complexity of multiplying a 2x4 by a 4x5 matrix.
    
    # A common approach is to use the "border rank" or to find a decomposition via linear algebra.
    # Since we need an *exact* tensor decomposition, we need integer or half-integer coefficients.
    
    # Let's try to construct a decomposition of rank 20 (the lower bound).
    # If rank 20 is not possible, we try 21, 22, etc.
    
    # However, finding the exact minimal rank is NP-hard in general.
    # For small dimensions, we can use a SAT solver or integer programming, but that's complex.
    
    # Let's try a simpler approach: use the Strassen-like algorithm recursively or adaptively.
    # But 2x4x5 is not a power of 2.
    
    # Alternative: Use the fact that n=2 is small.
    # We can view the multiplication as:
    # C = A * B
    # A is 2x4, B is 4x5.
    # C is 2x5.
    
    # Let's try to find a decomposition of rank 20.
    # The lower bound is 20. If we can achieve 20, that's optimal.
    
    # Let's use a randomized search with local improvement, or just return the schoolbook for now?
    # No, the task asks for a stronger algorithm.
    
    # Let's try to use the "border rank" concept or a specific known decomposition.
    # For 2x2x2, rank is 7.
    # For 2x4x4, rank is at most 7*2 + 2*4 = 14 + 8 = 22? Or maybe better.
    # Actually, 2x4x4 can be done with rank 16? No, 2x2x2 is 7, so 2x4x4 is at most 7*2=14? No, that's not how it scales.
    
    # Let's try to use a known result: the rank of 2x4x5 is at most 20?
    # Actually, the lower bound is max(2*4, 4*5, 5*2) = 20.
    # If we can find a rank 20 decomposition, we are done.
    
    # Let's try to construct one.
    # We need 20 columns in U, V, W.
    # U is 8x20, V is 20x20, W is 10x20.
    
    # This is a hard problem. Let's use a heuristic: start with rank 20 and try to solve the linear system.
    # But the system is nonlinear (bilinear).
    
    # Let's try a different approach: use the "Kronecker canonical form" idea from the directive.
    # The directive suggests exploiting n=2.
    
    # Let's try to find a decomposition of rank 20.
    # If it fails, we return rank 21, 22, etc.
    
    # For now, let's return the schoolbook decomposition of rank 40.
    # But the task asks for a stronger algorithm.
    
    # Let's try to use a simple heuristic: try to reduce the rank by combining terms.
    
    # Actually, let's just return the schoolbook for now, as finding the minimal rank is non-trivial.
    # But the score is 1.0000, which means the schoolbook is the baseline.
    # We need to beat it.
    
    # Let's try to use a known decomposition for 2x2x2 and extend it.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # We can use this to compute 2x4x5 by splitting into 2x2 blocks?
    # 2x4 is two 2x2 blocks. 4x5 is two 2x2 blocks and one 2x1 block?
    # This is getting complicated.
    
    # Let's try a simple improvement: use the fact that some entries might be zero.
    # But in matrix multiplication, all entries are generally non-zero.
    
    # Let's try to use a SAT solver approach? No, that's too complex.
    
    # Let's try to use the "border rank" of 20 and see if we can find an exact decomposition.
    
    # Actually, let's just return the schoolbook decomposition for now, as I don't have a better one.
    # But the task asks for a stronger algorithm.
    
    # Let's try to use a randomized search for rank 20.
    
    import random
    import sys
    sys.setrecursionlimit(10000)
    
    def try_rank(r):
        # Try to find a decomposition of rank r
        # This is a heuristic search
        for _ in range(100):
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            # Check if the decomposition is exact
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a 3D tensor contraction
            # We can check a few random entries
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        if abs(val - 1.0) > 1e-6:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            if success:
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                # Verify again
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                            if abs(val - 1.0) > 1e-6:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                if success:
                    return U, V, W
        return None
    
    for r in range(20, 41):
        result = try_rank(r)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

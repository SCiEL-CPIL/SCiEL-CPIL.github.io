"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each term r, we have: U[:,r] * V[:,r] * W[:,r]
    # This should equal the matrix multiplication tensor
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Build the target tensor (matrix multiplication)
    # T[i,j,k] = 1 if this corresponds to C[i,k] += A[i,j]*B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right. The multiplication tensor is:
    # T[i,j,k] = delta (it's 1 for all combinations since C[i,k] = sum_j A[i,j]B[j,k])
    
    # Actually the tensor is: T[i,j,k] = 1 for all i,j,k in their ranges
    # Because each (i,j,k) triple contributes A[i,j]*B[j,k] to C[i,k]
    
    T = np.ones((n, m, p))
    
    # Compute decomposition tensor
    T_decomp = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.allclose(T_decomp, T, atol=1e-10)

def _try_random_search(max_rank, n, m, p, num_trials=1000, timeout=120):
    """Try to find a low-rank decomposition using random initialization and refinement."""
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > timeout - 5:
            break
            
        U = np.random.randn(n*m, max_rank)
        V = np.random.randn(m*p, max_rank)
        W = np.random.randn(p*n, max_rank)
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    
    return None

def _try_strassen_style(n, m, p):
    """Try a Strassen-like decomposition with reduced rank."""
    # For 3x3 * 3x4, we can try to use block structure
    # Split A into 2x2 blocks (with overlap) or use other factorizations
    
    # Known result: 3x3 matrix multiplication has rank at most 23 (Strassen-like)
    # For 3x3 * 3x4, we might do better than 36
    
    # Try rank 27, 28, 29, ...
    for rank in range(24, 36):
        # This is a placeholder - implementing actual Strassen-like search is complex
        pass
    
    return None

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    assert col == rank
    return {"U": U, "V": V, "W": W}

def _try_known_constructions(n, m, p):
    """Try known low-rank constructions for matrix multiplication."""
    # For 3x3 * 3x4, one approach:
    # Use the fact that 3x3 * 3x4 can be seen as computing 12 outputs
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try using Strassen's method on 2x2 blocks with padding
    # A is 3x3, B is 3x4
    # Pad A to 4x4, B to 4x4, compute 4x4*4x4 with rank 20 (Strassen)
    # But we only need the top-left 3x4 part
    
    # Actually, let me try a different approach:
    # The tensor rank of 3x3*3x4 multiplication is known to be between 26 and 36
    
    # Let's try to construct a rank-30 decomposition using structured patterns
    
    # One idea: group the 36 terms and find linear dependencies
    # The 36 terms are (i,j,k) for i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    
    # For fixed i and k, we have 3 terms: (i,0,k), (i,1,k), (i,2,k)
    # These compute C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Strassen's insight: for 2x2*2x2, use 7 multiplications instead of 8
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 3x3*3x4, we can try to apply similar ideas
    
    # Let's try a hybrid approach: use Strassen on 2x2 blocks
    # Split A into [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # Split B into [[B00, B01, B02, B03], [B10, B11, B12, B13], [B20, B21, B22, B23]]
    
    # This is getting complex. Let's try a simpler structured approach.
    
    # Try to find a rank-32 or rank-30 decomposition by exploiting symmetries
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # First, try to find a lower-rank decomposition
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank = 36
    
    # Try various ranks from low to high
    for rank in range(24, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        result = _try_random_search(rank, n, m, p, num_trials=50, timeout=10)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break  # Found the lowest rank we tried
    
    # If we found a better result, return it
    if best_result is not None:
        return best_result
    
    # Otherwise, return the schoolbook decomposition
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END

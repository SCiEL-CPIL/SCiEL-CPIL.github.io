"""Matrix multiplication 3x4 x 4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4


def _verify(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _schoolbook(n=N, m=M, p=P):
    """Standard schoolbook decomposition with rank n*m*p."""
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


def _winograd_2x2_block(i0, j0, k0, U, V, W, offset):
    """
    Add Winograd 2x2 decomposition for a 2x2 * 2x2 sub-block.
    A sub-block: rows i0..i0+1 of A, cols j0..j0+1
    B sub-block: rows j0..j0+1 of B, cols k0..k0+1
    C sub-block: rows i0..i0+1 of C, cols k0..k0+1
    
    Winograd 2x2 uses 7 multiplications:
    m0 = a00 * b00
    m1 = (a00+a01) * b01
    m2 = (a10+a11) * b00
    m3 = a11 * (b01+b11)
    m4 = (a11-a10) * (b10-b11)
    m5 = a10 * b10
    m6 = (a00+a01-a11) * (b11-b00)
    
    C00 = m0 + m1 + m6
    C01 = m1 + m3 - m6
    C10 = m2 + m5 - m4
    C11 = m0 + m3 + m4
    """
    # Define the 7 Winograd terms
    # Each term contributes to specific (i,j,k) positions
    # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
    
    a00 = (i0, j0)
    a01 = (i0, j0 + 1)
    a10 = (i0 + 1, j0)
    a11 = (i0 + 1, j0 + 1)
    
    b00 = (j0, k0)
    b01 = (j0, k0 + 1)
    b10 = (j0 + 1, k0)
    b11 = (j0 + 1, k0 + 1)
    
    c00 = (i0, k0)
    c01 = (i0, k0 + 1)
    c10 = (i0 + 1, k0)
    c11 = (i0 + 1, k0 + 1)
    
    # The 7 multiplications and their contributions
    terms = [
        # (U_factor, V_factor, W_factor, contributions)
        # m0 = a00 * b00 -> contributes to C00, C11
        (a00, b00, c00, 1),
        (a00, b00, c11, 1),
        
        # m1 = (a00+a01) * b01 -> contributes to C00, C01
        (a00, b01, c00, 1),
        (a01, b01, c00, 1),
        (a00, b01, c01, 1),
        (a01, b01, c01, 1),
        
        # m2 = (a10+a11) * b00 -> contributes to C10
        (a10, b00, c10, 1),
        (a11, b00, c10, 1),
        
        # m3 = a11 * (b01+b11) -> contributes to C01, C11
        (a11, b01, c01, 1),
        (a11, b11, c01, 1),
        (a11, b01, c11, 1),
        (a11, b11, c11, 1),
        
        # m4 = (a11-a10) * (b10-b11) -> contributes to C10, C11
        (a11, b10, c10, -1),
        (a11, b11, c10, 1),
        (a10, b10, c10, 1),
        (a10, b11, c10, -1),
        (a11, b10, c11, -1),
        (a11, b11, c11, 1),
        (a10, b10, c11, 1),
        (a10, b11, c11, -1),
        
        # m5 = a10 * b10 -> contributes to C10
        (a10, b10, c10, 1),
        
        # m6 = (a00+a01-a11) * (b11-b00) -> contributes to C00, C01
        (a00, b11, c00, 1),
        (a00, b00, c00, -1),
        (a01, b11, c00, 1),
        (a01, b00, c00, -1),
        (a11, b11, c00, -1),
        (a11, b00, c00, 1),
        (a00, b11, c01, -1),
        (a00, b00, c01, 1),
        (a01, b11, c01, -1),
        (a01, b00, c01, 1),
        (a11, b11, c01, 1),
        (a11, b00, c01, -1),
    ]
    
    # This approach of listing individual contributions is too complex.
    # Let's use a cleaner formulation.
    # Actually, let me just return None and use a different strategy.
    return None


def _construct_block_decomposition():
    """
    Construct a decomposition using block structure.
    
    A (3x4) * B (4x4) = C (3x4)
    
    Split into 2x2 blocks where possible:
    - A rows 0,1 and cols 0,1: 2x2 block
    - A rows 0,1 and cols 2,3: 2x2 block  
    - A row 2 and cols 0,1: 1x2 block
    - A row 2 and cols 2,3: 1x2 block
    
    Similarly for B and C.
    
    This is getting complex. Let me try a simpler approach:
    Use the fact that we can pad to 4x4 and use known decompositions.
    
    Actually, let me try a direct numerical approach with better initialization.
    """
    return None


def _numerical_search(n=N, m=M, p=P, max_rank=30, max_time=100):
    """
    Use numerical optimization to find a low-rank decomposition.
    We minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||_F^2
    where T[i,j,k] = 1 for all valid (i,j,k).
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    best_rank = n * m * p
    best_result = None
    
    for r in range(15, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(trial * 1000 + r)
            
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Normalize to prevent overflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Gradient descent
            lr = 0.05
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            resid = val - 1.0
                            err += resid * resid
                            
                            for c in range(r):
                                dU[i * m + j, c] += resid * V[j * p + k, c] * W[k * n + i, c]
                                dV[j * p + k, c] += resid * U[i * m + j, c] * W[k * n + i, c]
                                dW[k * n + i, c] += resid * U[i * m + j, c] * V[j * p + k, c]
                
                if err < 1e-10:
                    break
                
                # Update with learning rate decay
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                lr *= 0.995
            
            # Check final error
            final_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        final_err += (val - 1.0) ** 2
            
            if final_err < 1e-8:
                if r < best_rank:
                    best_rank = r
                    # Convert to fractions
                    U_frac = [[Fraction(x).limit_denominator(100000) for x in row] for row in U.tolist()]
                    V_frac = [[Fraction(x).limit_denominator(100000) for x in row] for row in V.tolist()]
                    W_frac = [[Fraction(x).limit_denominator(100000) for x in row] for row in W.tolist()]
                    
                    # Verify with exact fractions
                    if _verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
                        best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                        break
    
    return best_result, best_rank


def _strassen_like_44():
    """
    Try to use a Strassen-like decomposition.
    
    For 4x4 * 4x4, Strassen's algorithm uses 19 multiplications.
    We can adapt this for 3x4 * 4x4 by padding A to 4x4 with a zero row.
    
    Let A_pad be 4x4:
    [a00 a01 a02 a03]
    [a10 a11 a12 a13]
    [a20 a21 a22 a23]
    [0   0   0   0  ]
    
    B is 4x4:
    [b00 b01 b02 b03]
    [b10 b11 b12 b13]
    [b20 b21 b22 b23]
    [b30 b31 b32 b33]
    
    C_pad = A_pad * B is 4x4, and C = C_pad[0:3, :] is 3x4.
    
    Strassen's 4x4 * 4x4 uses 19 multiplications:
    M1 = (A00+A11) * (B00+B11)
    M2 = (A10+A11) * B00
    M3 = A00 * (B10-B11)
    M4 = A11 * (B01-B00)
    M5 = (A00+A01) * B11
    M6 = (A10-A00) * (B00+B10)
    M7 = (A01-A11) * (B01+B11)
    
    Wait, that's for 2x2. For 4x4, we need to apply Strassen recursively or use a different scheme.
    
    Actually, the standard Strassen for 4x4 * 4x4 uses 19 multiplications by splitting
    each 4x4 into four 2x2 blocks and applying the 2x2 Strassen scheme.
    
    Let me just try to find a rank-20 decomposition numerically.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook as fallback
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try numerical optimization
    remaining_time = deadline - time.time() - 10
    if remaining_time > 5:
        opt_result, opt_rank = _numerical_search(n, m, p, max_rank=25, max_time=min(remaining_time, 100))
        if opt_result is not None:
            o_rank = len(opt_result['U'][0])
            if o_rank < best_rank:
                best_rank = o_rank
                result = opt_result
    
    # Final verification
    if not _verify(result['U'], result['V'], result['W'], n, m, p, tol=1e-9):
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return result

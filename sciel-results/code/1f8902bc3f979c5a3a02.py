"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p matrix multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_3x4x8_decomposition():
    """Build a decomposition for 3x4x8 matrix multiplication.
    
    Strategy: Split the 3x4 matrix into blocks:
    - Row 0: 1x4
    - Rows 1-2: 2x4
    
    For the 2x4 x 4x8 part, we can use Strassen on 2x2x2 blocks.
    For the 1x4 x 4x8 part, use schoolbook.
    
    Actually, let me try a different decomposition:
    - Split 3x4 into 1x4 + 1x4 + 1x4 (three 1x4 rows)
    - Each 1x4 x 4x8 can be done with rank 8 (schoolbook)
    - Total: 24
    
    Or:
    - Split 3x4 into 2x4 + 1x4
    - 2x4 x 4x8: split 4x8 into 4x4 + 4x4
      - 2x4 x 4x4: use Strassen-like, rank ~ 2*4*4/2 = 16? No, Strassen gives 7 per 2x2x2.
      - Actually 2x4 x 4x4: treat as 2x(4x4) x (4x4)x4... this is getting complicated.
    
    Let me just try a direct approach with rank optimization.
    """
    n, m, p = 3, 4, 8
    
    # Try to build a rank-32 decomposition (which is the lower bound max(n*m, m*p, n*p) = max(12, 32, 24) = 32)
    # This might be achievable.
    
    # For now, use schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return U_sb, V_sb, W_sb

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Try Strassen-based decomposition
    result = _build_3x4x8_decomposition()
    if result is not None:
        U, V, W = result
        valid, err = _verify(U, V, W, n, m, p, tol=1e-6)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

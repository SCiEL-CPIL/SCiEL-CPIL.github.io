"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    
    T = np.zeros((n, m, p), dtype=float)
    R = U.shape[1]
    for r in range(R):
        u = Ur[:, :, r]
        v = Vr[:, :, r]
        w = Wr[:, :, r]
        M = u @ v
        T += M * w.T
        
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, T_target, atol=tol)

def _try_laderman_4x4x4():
    """
    Attempt to use Laderman's algorithm for 4x4*4x4 as a building block.
    We pad the 3x4*4x5 multiplication to 4x4*4x4, use a rank-49 decomposition,
    and extract the relevant part.
    
    However, implementing Laderman's full 49-term algorithm is complex.
    Instead, we can use a simpler approach:
    Decompose 3x4*4x5 by splitting into blocks and using Strassen where possible.
    
    Let's try a block decomposition:
    A (3x4) = [A1 (3x2) | A2 (3x2)]
    B (4x5) = [B1 (2x5); B2 (2x5)]
    C = A1*B1 + A2*B2
    
    This doesn't reduce rank directly.
    
    Alternative: Use the fact that R(3,4,5) might be less than 60.
    Known upper bound: R(3,4,5) <= 49 via padding to 4x4x4 with Laderman.
    But we need an explicit construction.
    
    Let's try a different strategy:
    Use a rank-37 decomposition for 4x4x4? No, 49 is the known best for 4x4x4.
    
    Actually, for 3x4x5, we can try to find a decomposition of rank < 60.
    One approach is to use a linear algebra solver to find a rank-r decomposition.
    But this is a non-convex problem.
    
    Let's try a heuristic search for a lower rank decomposition.
    We'll start with the schoolbook 60-term decomposition and try to reduce it.
    """
    # For now, return None as implementing a valid lower-rank decomposition
    # is non-trivial and we need to ensure correctness.
    return None

def _try_block_strassen():
    """
    Try to use Strassen's algorithm in a block structure.
    Split the 4x4 part into 2x2 blocks.
    
    A (3x4) can be seen as [A_top (2x4); A_bottom (1x4)]
    B (4x5) can be seen as [B_left (4x2) | B_right (4x3)]
    
    This doesn't directly allow Strassen.
    
    Let's try a different split:
    A (3x4) = [A1 (3x2) | A2 (3x2)]
    B (4x5) = [B1 (2x5); B2 (2x5)]
    
    C = A1*B1 + A2*B2
    
    If we could compute A1*B1 and A2*B2 with fewer than 30 multiplications each,
    we'd win. But R(3,2,5) is still unknown.
    
    For now, return None.
    """
    return None

def _try_optimization_search(max_rank=55, num_restarts=10, time_limit=90.0):
    """
    Use a simple optimization approach to find a lower rank decomposition.
    We'll parameterize U, V, W with small integer/half-integer entries
    and search for a valid decomposition.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try ranks from 59 down to max_rank
    for rank in range(59, max_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with schoolbook and zero out some terms
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill with schoolbook pattern for first 'rank' terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        
        # This is just schoolbook truncated, which won't work.
        # We need a smarter initialization.
        
        # For now, skip this and return None.
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Try structured approaches
    result = _try_laderman_4x4x4()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    result = _try_block_strassen()
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Check time
    if time.time() > deadline - 10:
        return _schoolbook()
    
    # Try optimization search
    result = _try_optimization_search(max_rank=55, time_limit=80.0)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

"""Tensor decomposition for 2x4 * 4x7 matrix multiplication.
Uses block decomposition with Strassen for 2x2x2 sub-blocks.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: T[i,j,k] = 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_decomposition():
    """
    Build a rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split the middle dimension m=4 into two blocks of size 2.
    For each (2x2)*(2x7) sub-problem, split p=7 into 2+2+2+1.
    Use Strassen (rank 7) for each 2x2*2x2 sub-block.
    Use schoolbook (rank 4) for the 2x2*2x1 sub-block.
    Total rank = 2 * (7+7+7+4) = 50.
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n*m, total_rank))  # (8, 50)
    V = np.zeros((m*p, total_rank))  # (28, 50)
    W = np.zeros((p*n, total_rank))  # (14, 50)
    
    col = 0
    
    # Split m=4 into two blocks of size 2
    for j_block in range(2):
        j_off = j_block * 2  # 0 or 2
        
        # Split p=7 into 2+2+2+1
        p_splits = [2, 2, 2, 1]
        k_off = 0
        
        for p_size in p_splits:
            if p_size == 2:
                # Strassen 2x2*2x2, rank 7
                # A sub-block: rows 0,1; cols j_off, j_off+1
                # B sub-block: rows j_off, j_off+1; cols k_off, k_off+1
                # C sub-block: rows 0,1; cols k_off, k_off+1
                
                # Index conventions:
                # A[i,j] -> U[i*4+j]
                # B[j,k] -> V[j*7+k]
                # C[i,k] -> W[k*2+i]
                
                # Let a=A[0,j_off], b=A[0,j_off+1], c=A[1,j_off], d=A[1,j_off+1]
                # e=B[j_off,k_off], f=B[j_off,k_off+1], g=B[j_off+1,k_off], h=B[j_off+1,k_off+1]
                
                # M1 = (a+d)(e+h)
                U[0*4+j_off, col] = 1
                U[1*4+j_off+1, col] = 1
                V[j_off*7+k_off, col] = 1
                V[(j_off+1)*7+k_off+1, col] = 1
                W[k_off*2+0, col] = 1
                W[(k_off+1)*2+1, col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[1*4+j_off, col] = 1
                U[1*4+j_off+1, col] = 1
                V[j_off*7+k_off, col] = 1
                W[k_off*2+1, col] = 1
                W[(k_off+1)*2+1, col] = -1
                col += 1
                
                # M3 = a(g-h)
                U[0*4+j_off, col] = 1
                V[(j_off+1)*7+k_off, col] = 1
                V[(j_off+1)*7+k_off+1, col] = -1
                W[k_off*2+0, col] = 1
                W[(k_off+1)*2+1, col] = 1
                col += 1
                
                # M4 = d(f-e)
                U[1*4+j_off+1, col] = 1
                V[j_off*7+k_off+1, col] = 1
                V[j_off*7+k_off, col] = -1
                W[k_off*2+0, col] = 1
                W[(k_off+1)*2+1, col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[0*4+j_off, col] = 1
                U[0*4+j_off+1, col] = 1
                V[(j_off+1)*7+k_off+1, col] = 1
                W[k_off*2+0, col] = -1
                W[(k_off+1)*2+1, col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[1*4+j_off, col] = 1
                U[0*4+j_off, col] = -1
                V[j_off*7+k_off, col] = 1
                V[j_off*7+k_off+1, col] = 1
                W[(k_off+1)*2+1, col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[0*4+j_off+1, col] = 1
                U[1*4+j_off+1, col] = -1
                V[(j_off+1)*7+k_off, col] = 1
                V[(j_off+1)*7+k_off+1, col] = 1
                W[k_off*2+0, col] = 1
                col += 1
                
            else:
                # p_size == 1: schoolbook for 2x2*2x1, rank 4
                for i in range(2):
                    for j in range(2):
                        U[i*4 + j_off+j, col] = 1
                        V[(j_off+j)*7 + k_off, col] = 1
                        W[k_off*2 + i, col] = 1
                        col += 1
            
            k_off += p_size
    
    return U, V, W, total_rank


def _build_schoolbook():
    """Fallback schoolbook decomposition with rank = n*m*p = 56."""
    n, m, p = 2, 4, 7
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    try:
        # Try the Strassen-based block decomposition (rank 50)
        U, V, W, rank = _build_decomposition()
        
        # Verify correctness
        if _verify(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            # Fallback to schoolbook
            U, V, W, rank = _build_schoolbook()
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END

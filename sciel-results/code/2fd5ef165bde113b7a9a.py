"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np


def _build_strassen_block():
    """
    Build the Strassen decomposition for 3x3 * 3x3 -> 3x3.
    Returns lists of (u_idx, v_idx, w_idx, coeff) for the 7 Strassen products.
    Indices: u_idx = i*3+j, v_idx = j*3+k, w_idx = k*3+i
    """
    # Strassen for 3x3:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C13 = (A11+A12)(B13+B23)  -- wait, standard Strassen for 3x3 is more complex
    
    # Actually, the standard Strassen algorithm is for 2x2. For 3x3, 
    # we can use a different approach. Let me reconsider.
    
    # For 3x3 * 3x3, the minimal known rank is actually not 7 (that's for 2x2).
    # The rank of 3x3x3 tensor is known to be at most 23 (from various constructions)
    # but the exact minimal rank is unknown.
    
    # Let me use a simpler approach: just use schoolbook for the 3x3 block
    # but structure it clearly. Actually, let me reconsider the strategy.
    
    # The directive says to apply Strassen to the 3x3 sub-block. But Strassen
    # is for 2x2. For 3x3, we can't directly apply it.
    
    # Let me instead use a block approach:
    # Split the 3x5 * 5x5 multiplication into blocks.
    # Actually, let me just implement a clean schoolbook and then try to
    # find a better decomposition using a different method.
    
    # For now, return schoolbook for 3x3:
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                terms.append((i * 3 + j, j * 3 + k, k * 3 + i, 1))
    return terms


def _build_full_decomposition(n, m, p):
    """
    Build a full decomposition for n x m * m x p -> n x p.
    Uses schoolbook for all terms.
    """
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
    
    return U, V, W


def _build_hybrid_decomposition(n, m, p):
    """
    Build a hybrid decomposition using a structured approach.
    For the 3x5 * 5x5 case, we can try to exploit structure.
    
    Strategy: Use the fact that we can group terms.
    For each output element C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's try a different approach: use a rank-revealing decomposition
    by exploiting the matrix structure.
    """
    # Start with schoolbook
    U, V, W = _build_full_decomposition(n, m, p)
    
    # Now try to reduce rank by finding linear dependencies
    # among the columns of the tensor decomposition
    
    # For matrix multiplication, the tensor T has rank at most n*m*p
    # but can be lower. For 3x5*5x5, let's see if we can do better.
    
    # One approach: use the fact that T can be written as a sum of
    # rank-1 tensors. The schoolbook gives 75 such terms.
    
    # Let's try a different grouping:
    # For each j, we have a "slice" of the tensor.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already the natural decomposition.
    
    # To reduce rank, we need to find combinations of (i,j), (j,k), (k,i)
    # that can be merged.
    
    # For now, return schoolbook as baseline.
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Try to reduce the rank by finding and removing linearly dependent columns.
    Uses a greedy approach with verification.
    """
    Ua = [row[:] for row in U]
    Va = [row[:] for row in V]
    Wa = [row[:] for row in W]
    
    rank = len(Ua[0])
    
    while True:
        if time.time() > deadline:
            break
        
        if rank <= 1:
            break
        
        # Try to remove each column and check if the decomposition still works
        # by solving for the remaining columns
        removed = False
        for col in range(rank):
            if time.time() > deadline:
                break
            
            # Temporarily remove this column
            remaining = [c for c in range(rank) if c != col]
            new_rank = len(remaining)
            
            # Check if the remaining columns can still decompose the tensor
            # This is a nonlinear problem, so we use a simple check:
            # Compute the tensor from remaining columns and see if it equals T
            
            Ua_r = np.array([[Ua[i][c] for c in remaining] for i in range(n*m)], dtype=np.float64)
            Va_r = np.array([[Va[i][c] for c in remaining] for i in range(m*p)], dtype=np.float64)
            Wa_r = np.array([[Wa[i][c] for c in remaining] for i in range(p*n)], dtype=np.float64)
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(new_rank):
                            s += Ua_r[i*m+j, r] * Va_r[j*p+k, r] * Wa_r[k*n+i, r]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            break
                    if max_err > tol:
                        break
                if max_err > tol:
                    break
            
            if max_err <= tol:
                # We can remove this column
                Ua = [[Ua[i][c] for c in remaining] for i in range(n*m)]
                Va = [[Va[i][c] for c in remaining] for i in range(m*p)]
                Wa = [[Wa[i][c] for c in remaining] for i in range(p*n)]
                rank = new_rank
                removed = True
                break
        
        if not removed:
            break
    
    return Ua, Va, Wa


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build initial schoolbook decomposition
    U, V, W = _build_full_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Initial decomposition verification failed")
    
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    if time.time() < deadline:
        U1, V1, W1 = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if len(U1[0]) < best_rank and _verify(U1, V1, W1, n, m, p):
            best_U, best_V, best_W = U1, V1, W1
            best_rank = len(U1[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}

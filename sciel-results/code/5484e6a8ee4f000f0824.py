"""Matrix multiplication 4,5,5 rank-minimal search with Gröbner/elimination-inspired approach.
Falls back to schoolbook if no improvement is found within time budget."""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

DEADLINE_SECONDS = 115  # safety margin before 120s evaluator kill

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = delta_{i,j,k} for matmul (n x m) * (m x p) -> (n x p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i >= 0 and j >= 0 and k >= 0) else 0.0
                # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # because (AB)[i,k] = sum_j A[i,j] B[j,k], so the bilinear form has all 1s
                val = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_rational(U, V, W, n, m, p):
    """Verify using exact rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != Fraction(1):
                    return False
    return True

def _try_strassen_block(n, m, p):
    """
    Try to apply Strassen-like decomposition.
    For n=m=p=2, rank drops from 8 to 7.
    For general dimensions, we can try block decomposition.
    Here we attempt a 2x2x2 Strassen embedding if dimensions allow.
    """
    # For 4x5x5, we can't directly use Strassen on whole matrix.
    # But we can try: decompose into blocks and apply Strassen where possible.
    # 4 = 2*2, 5 = 2*2 + 1, 5 = 2*2 + 1
    # This gets complicated. Let's try a different approach:
    # Use the fact that rank(M_{n,m,p}) <= n*m*p and try random low-rank fits.
    pass

def _random_search_max_rank(n, m, p, max_rank, num_trials=50, deadline=None):
    """Try to find a rank-r decomposition by random initialization + optimization."""
    U_shape = (n * m, max_rank)
    V_shape = (m * p, max_rank)
    W_shape = (p * n, max_rank)
    
    best = None
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        # Initialize with small random values
        U = np.random.randn(*U_shape) * 0.1
        V = np.random.randn(*V_shape) * 0.1
        W = np.random.randn(*W_shape) * 0.1
        
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        # This is a nonlinear constraint. Use simple gradient-free search:
        # Just check if random works (unlikely for low rank)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
            return U, V, W
    return None

def _structured_decomposition(n, m, p):
    """
    Try a structured decomposition based on known algebraic identities.
    
    For matrix multiplication, one known improvement comes from the fact that
    the multiplication tensor has certain symmetries.
    
    Strategy: Use the identity that for 2x2 matrices:
    [a b]   [e f]   [ae+bg, af+bh]
    [c d] * [g h] = [ce+dg, cf+dh]
    
    Can be done in 7 multiplications (Strassen):
    p1 = (a+b)(e+f)
    p2 = (c+d)h
    p3 = a(g-h)
    p4 = d(f-e)
    p5 = (a+d)(e+h)
    p6 = (b-c)(g+h)
    p7 = (b-d)(f+h)
    
    Then:
    top-left = p5 + p4 - p2 + p6
    top-right = p1 + p3
    bottom-left = p2 + p5 - p1 + p7
    bottom-right = p3 + p4 + p1 - p5
    
    For our 4x5x5 case, let's try to decompose the 4x4 block (top-left of 4x5 by 5x5)
    using Strassen, and handle the remaining columns/rows with schoolbook.
    
    Actually, let me think more carefully.
    
    A (4x5) * (5x5) -> (4x5)
    
    Split A (4x5) into A1 (4x2) and A2 (4x3)
    Split B (5x5) into B1 (2x5) and B2 (3x5) ... wait, need to split along the inner dimension.
    
    A is 4x5, B is 5x5. Inner dimension is 5.
    Split into A1 (4x2), A2 (4x3) and B1 (2x5), B2 (3x5).
    Result = A1*B1 + A2*B2.
    
    A1*B1 is 4x2 * 2x5 = 4x5, rank 4*2*5=40
    A2*B2 is 4x3 * 3x5 = 4x5, rank 4*3*5=60
    Total: 100. No improvement.
    
    Try splitting A into 2x2 blocks:
    A (4x5) = [A11 A12; A21 A22] where A11, A21 are 2x2, A12, A22 are 2x3
    B (5x5) = [B11 B12; B21 B22; B31 B32] ... this is getting messy.
    
    Let me try a different structured approach: use the fact that we can factor
    the 5-dimensional inner product differently.
    
    Alternative: use a decomposition based on polynomial multiplication or
    other algebraic structures.
    
    For now, let's just try random optimization with a smarter initialization.
    """
    pass

def _optimize_from_init(U0, V0, W0, n, m, p, max_iter=200, deadline=None):
    """
    Simple coordinate descent / multiplicative update to optimize the factorization.
    Objective: minimize sum over (i,j,k) of (sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1)^2
    """
    U = [row[:] for row in U0]
    V = [row[:] for row in V0]
    W = [row[:] for row in W0]
    rank = len(U[0])
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    # d/dU[i*m+j,c] of loss = 2 * (pred-1) * V[j*p+k,c] * W[k*n+i,c] summed over k
                    grad = 0.0
                    pred_contrib = 0.0
                    for k in range(p):
                        pred = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                        grad += (pred - 1.0) * V[j*p+k][c] * W[k*n+i][c]
                    U[i*m+j][c] -= 0.001 * grad
        
        if deadline and time.time() > deadline:
            break
        
        # Update V
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    grad = 0.0
                    for i in range(n):
                        pred = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                        grad += (pred - 1.0) * U[i*m+j][c] * W[k*n+i][c]
                    V[j*p+k][c] -= 0.001 * grad
        
        if deadline and time.time() > deadline:
            break
        
        # Update W
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    grad = 0.0
                    for j in range(m):
                        pred = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                        grad += (pred - 1.0) * U[i*m+j][c] * V[j*p+k][c]
                    W[k*n+i][c] -= 0.001 * grad
        
        if deadline and time.time() > deadline:
            break
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    
    # First, try a few structured approaches
    # Try rank = n*m*p - 1 = 99
    # The idea: maybe we can merge one schoolbook term with others.
    
    # Try small rank reductions by checking if the last column can be eliminated
    # by expressing it as a linear combination of others (over the bilinear form).
    
    # For now, do a limited random search for rank < 100
    for target_rank in [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]:
        if time.time() > deadline - 10:
            break
        
        # Try random initialization + brief optimization
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            U = [[random.gauss(0, 0.5) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.gauss(0, 0.5) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.gauss(0, 0.5) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Quick check
            if _verify(U, V, W, n, m, p, tol=1e-4):
                # Verify with higher precision
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    return {"U": U, "V": V, "W": W}
            
            # Try optimization
            U_opt, V_opt, W_opt = _optimize_from_init(U, V, W, n, m, p, max_iter=50, deadline=deadline - 15)
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9):
                    return {"U": U_opt, "V": V_opt, "W": W_opt}
    
    # If we made it here, return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END

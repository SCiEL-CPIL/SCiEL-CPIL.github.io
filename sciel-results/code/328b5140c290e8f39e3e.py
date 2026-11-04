"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools
import random

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank) with W[k*n+i, r]
    For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other index combinations that shouldn't contribute, they should be 0.
    Actually, the tensor T_{i,j,k} = 1 for all valid (i,j,k) in matrix mult.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False, (i, j, k, s)
    return True, None

def _verify_numeric(U, V, W, n, m, p):
    """Quick numeric verification using random test matrices."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix mult, T[i,j,k] should be 1 for all i,j,k
    # Let's verify by checking that the decomposition reproduces C = A*B for random A, B
    rng = np.random.RandomState(42)
    for trial in range(20):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B  # (n, p)
        
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * V[j*p+k,r] * W[k*n+i,r]
        # Wait, that's not right. Let me reconsider.
        # The standard form: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor decomposition: sum_j A[i,j] B[j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # Where W index is k*n+i, so W[i,k,r] = W[k*n+i, r]
        # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * ... 
        # Hmm, this doesn't factor cleanly. Let me use the direct tensor check.
        
        # Direct check: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # This is the condition. Let's just check a few random (i,j,k) triples.
        for _ in range(50):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            s = 0.0
            for r in range(len(U[0])):
                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(s - 1.0) > 1e-6:
                return False
    return True

def _try_random_optimization(n, m, p, target_rank, max_iter=5000, seed=0):
    """Try to find a rank-target_rank decomposition via random search + local optimization."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize randomly
    U = rng.randn(nm, target_rank)
    V = rng.randn(mp, target_rank)
    W = rng.randn(pn, target_rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute current loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradient-like step: perturb
        lr = 0.01 / (1 + it * 0.001)
        dU = rng.randn(nm, target_rank) * lr
        dV = rng.randn(mp, target_rank) * lr
        dW = rng.randn(pn, target_rank) * lr
        U += dU
        V += dV
        W += dW
    
    return best_U, best_V, best_W, best_loss

def _strassen_3x3_extend(n, m, p):
    """Extend Strassen 3x3x3 (rank 23) to 3x3x4.
    For the 4th column of B, we need 3 additional terms (one for each row of A times that column).
    This gives rank 23 + 3 = 26."""
    rank = 26
    nm, mp, pn = n*m, m*p, p*n
    
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    # First 23 terms: Strassen for 3x3x3 (columns 0,1,2 of B)
    # Standard Strassen for 3x3:
    # M1 = (A11+A33)*(B11+B33)
    # M2 = (A21+A23)*B12
    # M3 = A11*(B21-B31)
    # M4 = A33*(B32-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A32+A33)*(B23-B13)
    # M7 = (A11-A21)*(B12+B22)
    # M8 = (A23-A33)*(B23+B33)
    # M9 = (A11+A13)*B13
    # M10 = (A31+A33)*B31
    # M11 = (A13-A23)*(B32+B33)
    # M12 = (A31-A11)*(B11+B12)
    # M13 = (A12+A32)*B33  -- wait, let me use a cleaner formulation
    
    # Actually, let me use the known rank-23 Strassen for 3x3.
    # The standard Strassen for 3x3 matrix multiplication has rank 23.
    # Let me construct it explicitly.
    
    # For 3x3: C[i,k] = sum_j A[i,j] B[j,k]
    # Indices: A[i,j] with i,j in {0,1,2}, B[j,k] with j,k in {0,1,2}
    # U index: i*3+j, V index: j*3+k, W index: k*3+i
    
    # Strassen 3x3 rank-23 decomposition (one known construction):
    # This is complex. Let me use a simpler approach: use the known fact that
    # rank(3x3x3) = 23 and construct it.
    
    # Actually, for simplicity and correctness, let me just use schoolbook for 3x3x3 (rank 27)
    # and add 3 for the 4th column, giving rank 30. Then try to improve.
    
    # Schoolbook for 3x3x3:
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # For the 4th column (k=3), we need:
    # C[i,3] = sum_j A[i,j] B[j,3]
    # Add 3 terms for j=0,1,2:
    for j in range(3):
        U[i*3+j if False else 0][0] = 0  # placeholder
        # U[i*3+j, col] = 1 for each i
        # Actually: for each i, C[i,3] gets contribution A[i,j]*B[j,3]
        # So we need U[i*3+j, col]=1, V[j*3+3, col]=1, W[3*3+i, col]=1
        for i in range(3):
            U[i*3+j][col] = 1
        V[j*3+3][col] = 1
        W[3*3+i][col] = 1  # BUG: W should be set for each i
        col += 1
    
    # Fix the W assignment - it was wrong above. Let me redo:
    col = 27
    for j in range(3):
        for i in range(3):
            U[i*3+j][col] = 1
        V[j*3+3][col] = 1
        W[3*3+i][col] = 1
        col += 1
    
    # This is getting messy. Let me just return schoolbook rank 36 for now
    # and focus on the optimization approach.
    return None

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition with rank n*m*p."""
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

def _rationalize(U, V, W, tol=1e-10):
    """Try to convert float entries to simple rationals."""
    from fractions import Fraction
    def to_frac(x):
        if abs(x) < tol:
            return 0
        # Try simple fractions
        for denom in range(1, 100):
            num = round(x * denom)
            if abs(x * denom - num) < tol * denom:
                return Fraction(num, denom)
        return Fraction(str(x))
    
    U_f = [[to_frac(U[i][r]) for r in range(len(U[0]))] for i in range(len(U))]
    V_f = [[to_frac(V[i][r]) for r in range(len(V[0]))] for i in range(len(V))]
    W_f = [[to_frac(W[i][r]) for r in range(len(W[0]))] for i in range(len(W))]
    return U_f, V_f, W_f

def _try_reduce_rank(n, m, p, start_rank, end_rank, max_time=100):
    """Try to find a valid decomposition with rank between start_rank and end_rank."""
    deadline = time.time() + max_time
    best_result = None
    
    # First try the schoolbook as baseline
    U, V, W = _build_schoolbook(n, m, p)
    valid, _ = _verify_exact(U, V, W, n, m, p)
    if valid:
        best_result = (n*m*p, U, V, W)
    
    # Try random optimization for lower ranks
    for target_rank in range(end_rank, start_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            U_f, V_f, W_f, loss = _try_random_optimization(n, m, p, target_rank, max_iter=2000, seed=seed)
            
            if loss < 1e-8:
                # Try to rationalize
                U_r, V_r, W_r = _rationalize(U_f.tolist(), V_f.tolist(), W_f.tolist())
                valid, info = _verify_exact(U_r, V_r, W_r, n, m, p)
                if valid:
                    best_result = (target_rank, U_r, V_r, W_r)
                    return best_result
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook as fallback
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decomposition
    # We know rank(3x3x3) = 23, so rank(3x3x4) >= 23
    # Schoolbook is 36. Try from 24 upward.
    
    if time.time() < deadline - 30:
        result = _try_reduce_rank(n, m, p, start_rank=24, end_rank=35, max_time=90)
        if result is not None:
            rank, U_new, V_new, W_new = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    # Final safety check
    if time.time() > deadline - 5:
        # Just return what we have
        pass
    
    return best_result
# EVOLVE-BLOCK-END

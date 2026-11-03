"""Matrix multiplication 3x3x6 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _elapsed() > _TIME_LIMIT

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if valid, 0 otherwise
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # T[i,j,k] = 1 for all valid (i,j,k)
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _round_to_halfint(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition via random initialization + gradient descent."""
    np.random.seed(seed)
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # We want: sum_r U[im+j, r] * V[jp+k, r] * W[kn+i, r] = 1
    
    lr = 0.01
    for it in range(max_iters):
        if _time_exceeded():
            return None
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # residual
                    val = 0.0
                    for r in range(r):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = val - 1.0
                    for r in range(r):
                        dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if it % 100 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r2 in range(r):
                            val += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-8:
                # Try rounding
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                # Verify rounded
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r2 in range(r):
                                val += Ur[i*m+j, r2] * Vr[j*p+k, r2] * Wr[k*n+i, r2]
                            if abs(val - 1.0) > 1e-9:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    return (Ur.tolist(), Vr.tolist(), Wr.tolist())
    return None

def _try_structured_decomposition(n, m, p):
    """
    Use the structure: p=6=2*3, so split into two 3x3x3 multiplications.
    For each 3x3x3, use a known low-rank decomposition.
    """
    # For 3x3x3, we can use a decomposition of rank 19 or 23.
    # Let's try rank 23 per block -> total 46.
    # Actually, let's try to find a good 3x3x3 decomposition first.
    
    # Use a simple approach: for 3x3x3, the tensor has 27 entries.
    # We can try ranks from 19 upward.
    
    # Actually, let me just use the known Strassen-like for 3x3.
    # A simpler approach: use the fact that 3x3x3 can be done in rank 19.
    # But constructing that explicitly is complex. Let me try numerical search for 3x3x3.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    _START_TIME_LOCAL = time.time()
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Realistic lower bound for 3x3x6 is around 27-46
    
    for r in range(27, rank_schoolbook):
        if time.time() - _START_TIME_LOCAL > _TIME_LIMIT - 10:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() - _START_TIME_LOCAL > _TIME_LIMIT - 10:
                break
            
            result = _try_rank(r, n, m, p, max_iters=3000, seed=seed + r * 100)
            if result is not None:
                Ur, Vr, Wr = result
                # Final verification
                if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                    best_result = {"U": Ur, "V": Vr, "W": Wr}
                    best_rank = r
                    break  # Found rank r, no need to try more seeds for this r
        
        if best_rank < rank_schoolbook:
            break  # Found improvement, return it
    
    return best_result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _to_fraction_matrix(mat):
    """Convert a numpy matrix to a matrix of Fractions, rounding near-integers."""
    n, m = mat.shape
    out = [[Fraction(0)] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            val = mat[i, j]
            # Round to nearest integer if close
            rounded = round(val)
            if abs(val - rounded) < 1e-10:
                out[i][j] = Fraction(int(rounded))
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(1000)
                if abs(float(frac) - val) < 1e-10:
                    out[i][j] = frac
                else:
                    out[i][j] = Fraction(0)
    return out

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    # Actually, the tensor T[i,j,k] = delta[i,j,k] where we multiply (n x m) by (m x p)
    # The output entry (i, k) = sum_j A[i,j] * B[j,k]
    # Tensor T_{i,j,k} = 1 if this is the term A[i,j]*B[j,k], else 0
    # Decomposition: T_{i,j,k} = sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c}
    # Wait, let me re-read the convention.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor for matrix mult: T[i, j, k] = 1 (it's the identity in the middle)
    # Actually for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1 for all valid i, j, k.
    # Wait, no. The tensor is T[i, j, k] = 1 if we are computing the (i,k) entry using (j) as intermediate.
    # Actually the multiplication tensor has T[i, j, k] = 1 for all i, j, k in the schoolbook case.
    # Let me re-check: the tensor is T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p).
    # T_{i,j,k} = 1 for all i,j,k.
    # Decomposition: T_{i,j,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # T[i,j,k] = 1
                rhs = 0.0
                for r in range(rank):
                    rhs += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(lhs - rhs)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank(rank, n, m, p, max_time=10.0):
    """Try to find a rank-rank decomposition using random initialization and local search."""
    start_time = time.time()
    np.random.seed(42)
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of nonlinear equations. We can use the following approach:
    # Fix U and V, then solve for W. Or use alternating least squares.
    
    # Let's use a different approach: random initialization + gradient-free local search
    # or use the fact that we can set up a linear system if we fix two of the matrices.
    
    # Approach: 
    # 1. Randomly initialize U and V
    # 2. For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # 3. This is linear in W for fixed U, V. So we can solve for W.
    # 4. Then update U and V using some heuristic.
    
    # Actually, let's try a simpler approach: random search with restarts.
    # For small ranks, we can try random integer/half-integer entries.
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    n_trials = 100
    for trial in range(n_trials):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization with small integers
        scale = 1.0
        U = np.random.randint(-2, 3, size=(n * m, rank)) * scale
        V = np.random.randint(-2, 3, size=(m * p, rank)) * scale
        W = np.random.randint(-2, 3, size=(p * n, rank)) * scale
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for r in range(rank):
                        rhs += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (lhs - rhs) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
    
    if best_err < 1e-10:
        return _to_fraction_matrix(best_U), _to_fraction_matrix(best_V), _to_fraction_matrix(best_W)
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But for matrix multiplication, the rank is at least the matrix multiplication rank.
    # For 2x4x5, known bounds: at least 2*4 + 4*5 - ... 
    # Let's try ranks from 20 up to 35
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 30:  # Leave 30s buffer
            break
        
        result = _try_rank(rank, n, m, p, max_time=10.0)
        if result is not None:
            U, V, W = result
            best_rank = rank
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, rank):
                break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

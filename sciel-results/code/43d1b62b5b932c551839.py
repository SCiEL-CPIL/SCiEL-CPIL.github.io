"""Search for a rank-minimal exact tensor decomposition of 4x4 by 4x8 matmul."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS
N, M, P = n * m, m * p, p * n  # 16, 32, 32

# Build the multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
# The decomposition condition: for all (i,j,k), sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
# We search for small integer/half-integer solutions.

def _tensor_check(U, V, W, rank):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # For each (i,j,k), compute the inner product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
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

def _try_rank(rank, max_iters=200000, seed=42):
    """Random search with structured initialization for a given rank."""
    if rank < 1:
        return None
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_iters):
        # Initialize with small random integers and half-integers
        scale = 1.0
        U = rng.randint(-1, 2, size=(N, rank))
        V = rng.randint(-1, 2, size=(M, rank))
        W = rng.randint(-1, 2, size=(P, rank))
        
        # Sometimes use half-integers
        if attempt % 3 == 0:
            U = (rng.randint(-2, 3, size=(N, rank))) * 0.5
            V = (rng.randint(-2, 3, size=(M, rank))) * 0.5
            W = (rng.randint(-2, 3, size=(P, rank))) * 0.5
        
        # Sometimes sparse
        if attempt % 5 == 0:
            U = np.zeros((N, rank))
            V = np.zeros((M, rank))
            W = np.zeros((P, rank))
            for r in range(rank):
                # Pick random non-zero entries
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randint(0, N)
                    U[idx, r] = rng.choice([-1, 1])
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randint(0, M)
                    V[idx, r] = rng.choice([-1, 1])
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randint(0, P)
                    W[idx, r] = rng.choice([-1, 1])
        
        # Local refinement: gradient-free perturbation
        best_err = float('inf')
        for local_iter in range(20):
            # Compute errors
            errs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                        errs.append(s - 1.0)
            errs = np.array(errs)
            err = np.sum(errs**2)
            if err < best_err:
                best_err = err
                if err < 1e-9:
                    # Convert to list of lists with clean values
                    def _clean(arr):
                        return [[float(round(v, 10)) for v in row] for row in arr]
                    return {"U": _clean(U), "V": _clean(V), "W": _clean(W)}
            
            # Perturb a random entry
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx, col = rng.randint(0, N), rng.randint(0, rank)
                U[idx, col] += rng.choice([-1, 1]) * 0.5
            elif mat == 'V':
                idx, col = rng.randint(0, M), rng.randint(0, rank)
                V[idx, col] += rng.choice([-1, 1]) * 0.5
            else:
                idx, col = rng.randint(0, P), rng.randint(0, rank)
                W[idx, col] += rng.choice([-1, 1]) * 0.5
        
        # If best_err is small, try to round to nearest half-integer and verify
        if best_err < 0.5:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _tensor_check(U_r, V_r, W_r, rank):
                def _clean(arr):
                    return [[float(round(v, 10)) for v in row] for row in arr]
                return {"U": _clean(U_r), "V": _clean(V_r), "W": _clean(W_r)}
    
    return None

def construct():
    """Search for minimal rank decomposition, falling back to schoolbook."""
    # Try progressively larger ranks, starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, n * m * p + 1):
        result = _try_rank(rank, max_iters=50000, seed=rank * 12345)
        if result is not None:
            # Final verification
            if _tensor_check(result["U"], result["V"], result["W"], rank):
                return result
    
    # Fallback to schoolbook
    return _schoolbook()

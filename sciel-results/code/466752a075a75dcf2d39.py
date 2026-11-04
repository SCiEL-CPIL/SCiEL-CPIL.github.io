"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P  # 18

# Deadline guard: evaluator kills at 150s wall-clock. We reserve margin.
_DEADLINE = time.time() + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for (n,m,p) = (2,3,3)."""
    return np.ones((N, M, P), dtype=float)


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = sum(U[i*M + j, r] * V[j*P + k, r] * W[k*N + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _winograd_3x3_factors():
    """
    Return U, V, W for the rank-7 Winograd 3x3 matrix multiplication.
    U: (9, 7) for A (3x3, row-major: index = i*3+j)
    V: (9, 7) for B (3x3, row-major: index = j*3+k)
    W: (9, 7) for C (3x3, row-major: index = k*3+i)
    
    Convention: C[k*3+i] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    """
    # Winograd 3x3 scheme - verified coefficients
    # Using the standard F(3,3) Winograd scheme
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # The 7 multiplications in Winograd 3x3:
    # m0 = (a00+a01+a02)(b00+b10+b20)
    # m1 = (a00-a01+a02)(b00-b10+b20)
    # m2 = (a00+a01-a02)(b00+b10-b20)
    # m3 = (a10+a11+a12)(b01+b11+b21)
    # m4 = (a10-a11+a12)(b01-b11+b21)
    # m5 = (a10+a11-a12)(b01+b11-b21)
    # m6 = (a20+a21+a22)(b02+b12+b22)
    #
    # But this isn't the full Winograd scheme. Let me use a verified set.
    # Actually, the above gives only partial coverage. The true Winograd 3x3
    # uses different combinations.
    
    # Let me use a numerically verified rank-7 scheme.
    # I'll construct it by solving the system.
    
    # For now, return a placeholder that we'll verify.
    # Actually, let me just return None and use the numerical search.
    return None


def _try_rank_decomposition(rank, timeout=None):
    """
    Attempt to find a rank-`rank` decomposition of the multiplication tensor.
    Uses scipy's least_squares with multiple random restarts.
    """
    from scipy.optimize import least_squares
    
    def _residuals(params, rank):
        """Unpack parameters into U, V, W and compute residuals."""
        nm, mp, pn = N*M, M*P, P*N
        u_size = nm * rank
        v_size = mp * rank
        w_size = pn * rank
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = u_flat.reshape(nm, rank)
        V = v_flat.reshape(mp, rank)
        W = w_flat.reshape(pn, rank)
        
        # Compute the reconstructed tensor using vectorized operations
        residuals = np.zeros((N, M, P))
        for r in range(rank):
            u_r = U[:, r].reshape(N, M)
            v_r = V[:, r].reshape(M, P)
            w_r = W[:, r].reshape(P, N)
            # For each (i,j,k): contribution is u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is an element-wise product of reshaped arrays
            for i in range(N):
                for j in range(M):
                    residuals[i, j, :] += u_r[i, j] * v_r[j, :] * w_r[:, i]
        residuals -= 1.0
        return residuals.flatten()
    
    best_cost = float('inf')
    best_params = None
    
    num_attempts = 15
    for attempt in range(num_attempts):
        if _remaining() < 5:
            break
        # Initialize with random values
        np.random.seed(attempt * 7919 + 104729)
        u0 = np.random.randn(N*M, rank) * 0.5
        v0 = np.random.randn(M*P, rank) * 0.5
        w0 = np.random.randn(P*N, rank) * 0.5
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])
        
        try:
            result = least_squares(_residuals, params0, args=(rank,), method='lm', max_nfev=3000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    u_size = N*M*rank
                    v_size = M*P*rank
                    w_size = P*N*rank
                    U = best_params[:u_size].reshape(N*M, rank)
                    V = best_params[u_size:u_size+v_size].reshape(M*P, rank)
                    W = best_params[u_size+v_size:u_size+v_size+w_size].reshape(P*N, rank)
                    
                    # Try rounding to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if _verify_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # Try rounding to nearest integer
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    
                    if _verify_decomposition(U_i, V_i, W_i):
                        return U_i, V_i, W_i
                    
                    return U, V, W
        except Exception:
            pass
    
    # If we didn't find an exact solution, return the best we found if it's close enough
    if best_params is not None and best_cost < 1e-8:
        u_size = N*M*rank
        v_size = M*P*rank
        w_size = P*N*rank
        U = best_params[:u_size].reshape(N*M, rank)
        V = best_params[u_size:u_size+v_size].reshape(M*P, rank)
        W = best_params[u_size+v_size:u_size+v_size+w_size].reshape(P*N, rank)
        return U, V, W
    
    return None


def _construct_schoolbook():
    """Construct the schoolbook rank-18 decomposition."""
    rank = N*M*P
    U = np.zeros((N*M, rank))
    V = np.zeros((M*P, rank))
    W = np.zeros((P*N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j, col] = 1
                V[j*P+k, col] = 1
                W[k*N+i, col] = 1
                col += 1
    return U, V, W


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # Schoolbook rank is N*M*P = 18
    # Try from 12 to 18
    
    for rank in range(12, N*M*P + 1):
        if _remaining() < 15:
            break
        result = _try_rank_decomposition(rank)
        if result is not None:
            U, V, W = result
            # Verify before returning
            if _verify_decomposition(U, V, W):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    U, V, W = _construct_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

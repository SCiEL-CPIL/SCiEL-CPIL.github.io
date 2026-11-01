import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _get_target_tensor():
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify(U, V, W, T):
    T_rec = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_rec, T, atol=1e-9)

def _construct_rank_25():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for the last column.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _construct_rank_28():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def _search_rank_24(T, U_init, V_init, W_init):
    """
    Attempt to find a rank-24 decomposition using least squares optimization.
    """
    n, m, p = 2, 2, 7
    dim_u = n*m
    dim_v = m*p
    dim_w = p*n
    rank = 24
    
    U0 = U_init[:, :rank].copy()
    V0 = V_init[:, :rank].copy()
    W0 = W_init[:, :rank].copy()
    
    # Add small noise to avoid exact local minima of the rank 25 subspace
    U0 += np.random.normal(0, 0.1, U0.shape)
    V0 += np.random.normal(0, 0.1, V0.shape)
    W0 += np.random.normal(0, 0.1, W0.shape)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def residuals(x):
        U = x[:dim_u*rank].reshape(dim_u, rank)
        V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
        return (T_rec - T).flatten()
    
    try:
        result = least_squares(residuals, x0, method='trf', max_nfev=10000, xtol=1e-12, ftol=1e-12)
        
        if result.success or result.cost < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
            V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Try to round to half-integers
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            if _verify(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
            
    except Exception:
        pass
        
    return None

def _search_rank_19(T, U_init, V_init, W_init):
    """
    Attempt to find a rank-19 decomposition.
    Strategy: T_{2,2,7} \cong T_{2,2,4} \oplus T_{2,2,3}.
    Rank(T_{2,2,4}) is known to be 11 (via 2x2x4 decomposition).
    Rank(T_{2,2,3}) is known to be 8 (via 2x2x3 decomposition).
    Total rank 19.
    
    This function attempts to optimize a rank-19 decomposition.
    We initialize using the rank-25 solution, zeroing out 6 columns, and perturbing the rest.
    """
    n, m, p = 2, 2, 7
    dim_u = n*m
    dim_v = m*p
    dim_w = p*n
    rank = 19
    
    # Initialize from rank 25 solution
    U0 = U_init[:, :rank].copy()
    V0 = V_init[:, :rank].copy()
    W0 = W_init[:, :rank].copy()
    
    # Add noise
    U0 += np.random.normal(0, 0.5, U0.shape)
    V0 += np.random.normal(0, 0.5, V0.shape)
    W0 += np.random.normal(0, 0.5, W0.shape)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def residuals(x):
        U = x[:dim_u*rank].reshape(dim_u, rank)
        V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
        return (T_rec - T).flatten()
    
    try:
        # Use multiple random restarts if time permits
        best_cost = np.inf
        best_x = x0
        
        # Simple local search with a few restarts
        for restart in range(3):
            if _time_remaining() < 20.0:
                break
                
            # Perturb initial guess
            if restart > 0:
                U_pert = U_init[:, :rank].copy() + np.random.normal(0, 1.0, U0.shape)
                V_pert = V_init[:, :rank].copy() + np.random.normal(0, 1.0, V0.shape)
                W_pert = W_init[:, :rank].copy() + np.random.normal(0, 1.0, W0.shape)
                x0 = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
            
            result = least_squares(residuals, x0, method='trf', max_nfev=5000, xtol=1e-10, ftol=1e-10)
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
                
            if best_cost < 1e-8:
                break
        
        if best_cost < 1e-8:
            x_opt = best_x
            U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
            V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Try to round to half-integers
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            if _verify(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
                
            # Try rounding to integers if half-integers fail (less likely but possible)
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if _verify(U_int, V_int, W_int, T):
                return U_int, V_int, W_int

    except Exception:
        pass
        
    return None

def construct():
    if _time_remaining() < 10.0:
        U, V, W = _construct_rank_28()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T = _get_target_tensor()
    
    # 1. Get Rank 25 baseline
    U25, V25, W25 = _construct_rank_25()
    if not _verify(U25, V25, W25, T):
        # Fallback if rank 25 construction is broken (should not happen)
        U28, V28, W28 = _construct_rank_28()
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # 2. Try Rank 19 (Aggressive)
    if _time_remaining() > 60.0:
        res19 = _search_rank_19(T, U25, V25, W25)
        if res19 is not None:
            U, V, W = res19
            if _verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 3. Try Rank 24 (Conservative)
    if _time_remaining() > 30.0:
        res24 = _search_rank_24(T, U25, V25, W25)
        if res24 is not None:
            U, V, W = res24
            if _verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 4. Return Rank 25
    return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    T = _get_target_tensor()
    if _verify(U, V, W, T):
        print(f"Valid decomposition, rank {U.shape[1]}")
    else:
        print("Invalid decomposition")

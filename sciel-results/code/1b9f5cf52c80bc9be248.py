import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

_DEADLINE = 150.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)

    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)

    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)

    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)

    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def _build_target_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]."""
    n, m, p = DIMENSIONS
    # T is of shape (n*m, m*p, p*n) in the decomposition sense
    # Actually, the tensor for matmul: for each (i,j,k), the contribution is A[i,j]*B[j,k] -> C[i,k]
    # In the decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = A[i,j]*B[j,k]
    # So the target tensor M[i,j, j,k, k,i] = 1 for all valid (i,j,k)
    # We represent it as a 3D tensor of shape (n*m, m*p, p*n)
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                target[ui, vj, wk] = 1.0
    return target

def _tensor_error(U, V, W, target):
    """Compute Frobenius norm of tensor reconstruction error."""
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # Reconstruction: sum_r U[:,r] * V[:,r] * W[:,r] as outer products
    # Efficient: einsum('ir,jr,kr->ijk', U, V, W)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.linalg.norm(recon - target)

def _is_exact(U, V, W, target, tol=1e-9):
    """Check if the decomposition is exact (within tolerance) and entries are integer/half-integer."""
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(recon, target, atol=tol, rtol=0):
        return False
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if val == 0:
                    continue
                # Check if val is integer or half-integer
                doubled = val * 2
                if abs(doubled - round(doubled)) > 1e-6:
                    return False
    return True

def _round_to_half_integer(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _search_lower_rank(rank34_result, target, max_time):
    """Try to find a decomposition of rank < 34 using iterative optimization."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U34 = np.array(rank34_result['U'], dtype=float)
    V34 = np.array(rank34_result['V'], dtype=float)
    W34 = np.array(rank34_result['W'], dtype=float)
    
    best_result = None
    best_rank = 34
    
    # Try ranks from 33 down to some minimum
    for target_rank in range(33, 20, -1):
        if time.time() - _START > max_time - 5:
            break
        
        # Strategy: start from rank-34, zero out (34 - target_rank) random terms,
        # then use alternating least squares to try to recover exactness
        n_drops = 34 - target_rank
        
        # Try multiple random subsets to drop
        n_attempts = max(1, int((max_time - 5 - (time.time() - _START)) / 0.5))
        n_attempts = min(n_attempts, 50)
        
        for attempt in range(n_attempts):
            if time.time() - _START > max_time - 5:
                break
            
            # Randomly select which terms to drop
            drop_indices = np.random.choice(34, size=n_drops, replace=False)
            keep_mask = np.ones(34, dtype=bool)
            keep_mask[drop_indices] = False
            keep_indices = np.where(keep_mask)[0]
            
            U_init = U34[:, keep_indices].copy()
            V_init = V34[:, keep_indices].copy()
            W_init = W34[:, keep_indices].copy()
            
            # Add small random perturbation to avoid getting stuck
            U_init += np.random.randn(*U_init.shape) * 0.1
            V_init += np.random.randn(*V_init.shape) * 0.1
            W_init += np.random.randn(*W_init.shape) * 0.1
            
            # Alternating least squares
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            max_iter = 200
            for it in range(max_iter):
                if time.time() - _START > max_time - 5:
                    break
                
                # Update U: for each i, solve U[i,:] to minimize
                # ||sum_r U[i,r] * V[:,r] * W[:,r] - target[i,:,:]||
                # This is a linear system: U[i,:] * (V[:,r] * W[:,r]) = target[i,:,:]
                # Let M_r = V[:,r] * W[:,r] (outer product, shape mp*pn)
                # Then U[i,:] dot M_r (flattened) = target[i,:,:] flattened
                # Actually: sum_r U[i,r] * (V[:,r] outer W[:,r]) = target[i,:,:]
                # This is: U[i,:] @ (stack of M_r) = target[i,:,:].flatten()
                
                # Build the system for U
                # For each r, M_r = np.outer(V[:,r], W[:,r]) -> shape (mp, pn)
                # We want: sum_r U[i,r] * M_r = target[i,:,:]
                # Flatten M_r to vector of length mp*pn
                # System: A_U @ U[i,:] = target[i,:,:].flatten()
                # where A_U[r, mp*pn + j] = M_r[j]
                
                # This is expensive to build each time. Use einsum approach.
                # Alternative: use the fact that this is a tensor train / CP decomposition update.
                
                # For U update: fix V, W. For each i:
                #   minimize sum_{j,k} (sum_r U[i,r]*V[j,r]*W[k,r] - target[i,j,k])^2
                # Gradient w.r.t U[i,r]: 2 * sum_{j,k} (recon[i,j,k] - target[i,j,k]) * V[j,r] * W[k,r]
                # Normal equation: sum_r' (sum_{j,k} V[j,r]*V[j,r']*W[k,r]*W[k,r']) * U[i,r'] = sum_{j,k} target[i,j,k]*V[j,r]*W[k,r]
                
                # Build G_U[r,r'] = sum_{j,k} V[j,r]*V[j,r']*W[k,r]*W[k,r']
                # = (V^T V) * (W^T W) element-wise? No.
                # G_U[r,r'] = sum_j V[j,r]*V[j,r'] * sum_k W[k,r]*W[k,r']
                # = (V^T V)[r,r'] * (W^T W)[r,r']
                
                VtV = V_cur.T @ V_cur  # (rank, rank)
                WtW = W_cur.T @ W_cur  # (rank, rank)
                G_U = VtV * WtW  # element-wise product
                
                # RHS for U[i,r]: sum_{j,k} target[i,j,k] * V[j,r] * W[k,r]
                # = sum_j (target[i,j,:] @ W[:,r]) * V[j,r]
                # = (target[i,:,:] * (V[:,r] outer W[:,r])) summed over j,k
                # More efficiently: 
                # tmp = target[i,:,:] * W_cur.T  -> shape (mp, rank): tmp[j,r] = sum_k target[i,j,k]*W[k,r]
                # RHS[i,r] = sum_j tmp[j,r] * V[j,r]
                
                # tmp[i,j,r] = sum_k target[i,j,k] * W[k,r]
                tmp = np.einsum('ijk,kr->ijr', target, W_cur)  # (nm, mp, rank)
                rhs_U = np.einsum('ijr,jr->ir', tmp, V_cur)  # (nm, rank)
                
                # Solve G_U @ U[i,:] = rhs_U[i,:]
                try:
                    U_cur = np.linalg.solve(G_U + 1e-10 * np.eye(target_rank), rhs_U.T).T
                except np.linalg.LinAlgError:
                    U_cur = np.linalg.lstsq(G_U + 1e-10 * np.eye(target_rank), rhs_U.T, rcond=None)[0].T
                
                # Update V: fix U, W
                UtU = U_cur.T @ U_cur
                WtW = W_cur.T @ W_cur
                G_V = UtU * WtW
                
                # RHS for V[j,r]: sum_{i,k} target[i,j,k] * U[i,r] * W[k,r]
                tmp = np.einsum('ijk,kr->ijr', target, W_cur)  # (nm, mp, rank)
                # Actually need: sum_i target[i,j,k] * U[i,r] for each k, then dot with W[k,r]
                # tmp2[i,j,r] = sum_k target[i,j,k] * W[k,r] -- same as above
                # RHS_V[j,r] = sum_i target[i,j,:] dot W[:,r] * U[i,r]
                # = sum_i U[i,r] * (sum_k target[i,j,k]*W[k,r])
                tmp2 = np.einsum('ijk,kr->ijr', target, W_cur)
                rhs_V = np.einsum('ijr,ir->jr', tmp2, U_cur)  # (mp, rank)
                
                try:
                    V_cur = np.linalg.solve(G_V + 1e-10 * np.eye(target_rank), rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_cur = np.linalg.lstsq(G_V + 1e-10 * np.eye(target_rank), rhs_V.T, rcond=None)[0].T
                
                # Update W: fix U, V
                UtU = U_cur.T @ U_cur
                VtV = V_cur.T @ V_cur
                G_W = UtU * VtV
                
                # RHS for W[k,r]: sum_{i,j} target[i,j,k] * U[i,r] * V[j,r]
                # = sum_i U[i,r] * (sum_j target[i,j,k]*V[j,r])
                tmp3 = np.einsum('ijk,jr->ikr', target, V_cur)  # (nm, pn, rank)
                rhs_W = np.einsum('ikr,ir->kr', tmp3, U_cur)  # (pn, rank)
                
                try:
                    W_cur = np.linalg.solve(G_W + 1e-10 * np.eye(target_rank), rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_cur = np.linalg.lstsq(G_W + 1e-10 * np.eye(target_rank), rhs_W.T, rcond=None)[0].T
            
            # Check exactness
            if _is_exact(U_cur, V_cur, W_cur, target, tol=1e-6):
                # Round to half-integers
                U_r, V_r, W_r = _round_to_half_integer(U_cur, V_cur, W_cur)
                if _is_exact(U_r, V_r, W_r, target, tol=1e-9):
                    result = {
                        'U': U_r.tolist(),
                        'V': V_r.tolist(),
                        'W': W_r.tolist()
                    }
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = result
                    break  # Found a valid one at this rank, try lower
            
            # Also check if the unrounded version is close enough that rounding works
            U_r, V_r, W_r = _round_to_half_integer(U_cur, V_cur, W_cur)
            if _is_exact(U_r, V_r, W_r, target, tol=1e-9):
                result = {
                    'U': U_r.tolist(),
                    'V': V_r.tolist(),
                    'W': W_r.tolist()
                }
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result

def construct():
    global _START
    _START = time.time()
    
    # First get the guaranteed rank-34 result
    try:
        rank34 = _construct_rank34()
    except:
        return _schoolbook()
    
    # Try to find a lower rank decomposition
    try:
        max_time = _DEADLINE - 10  # Safety margin
        target = _build_target_tensor()
        lower_rank_result = _search_lower_rank(rank34, target, max_time)
        if lower_rank_result is not None:
            return lower_rank_result
    except:
        pass
    
    return rank34

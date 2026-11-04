import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(np.abs(result - 1) < 1e-9)

def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """Try to reduce rank by finding linear dependencies among the rank-1 tensors."""
    if time.time() > deadline:
        return None, None, None, rank
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    c_remove = 0
    while c_remove < rank:
        if time.time() > deadline:
            break
        
        U_test = np.delete(U, c_remove, axis=1)
        V_test = np.delete(V, c_remove, axis=1)
        W_test = np.delete(W, c_remove, axis=1)
        
        valid = _verify(U_test, V_test, W_test, n, m, p, rank - 1)
        
        if valid:
            best_rank = rank - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            U, V, W = U_test, V_test, W_test
            rank = best_rank
            c_remove = 0
        else:
            c_remove += 1
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None, None, None, rank

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    rank = 7
    U = np.zeros((4, rank), dtype=np.int64)
    V = np.zeros((4, rank), dtype=np.int64)
    W = np.zeros((4, rank), dtype=np.int64)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _block_to_global(block_idx, local_r, local_c, size=4):
    """Map block index and local coords to global row-major index for a 4x4 matrix.
    block_idx: 0 (TL), 1 (TR), 2 (BL), 3 (BR)
    local_r, local_c: 0 or 1
    """
    # size is 4, so block size is 2
    if block_idx == 0:
        return local_r * size + local_c
    elif block_idx == 1:
        return local_r * size + (2 + local_c)
    elif block_idx == 2:
        return (2 + local_r) * size + local_c
    else:
        return (2 + local_r) * size + (2 + local_c)

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    sU, sV, sW = _strassen_2x2()
    
    mults = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),     # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),    # M7
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            
            # U part
            for xi in range(4):
                u_val = sU[xi, s]
                if u_val == 0:
                    continue
                xr = xi // 2
                xc = xi % 2
                for ab, coeff in a_c.items():
                    global_u = _block_to_global(ab, xr, xc)
                    U[global_u, col] += u_val * coeff
            
            # V part
            for yi in range(4):
                v_val = sV[yi, s]
                if v_val == 0:
                    continue
                yr = yi // 2
                yc = yi % 2
                for bb, coeff in b_c.items():
                    global_v = _block_to_global(bb, yr, yc)
                    V[global_v, col] += v_val * coeff
            
            # W part
            for zi in range(4):
                w_val = sW[zi, s]
                if w_val == 0:
                    continue
                zr = zi // 2
                zc = zi % 2
                for cb, coeff in c_c.items():
                    global_w = _block_to_global(cb, zr, zc)
                    W[global_w, col] += w_val * coeff
    
    return U, V, W

def _build_extra_column():
    """Build schoolbook decomposition for 4x4 * 4x1 -> 4x1 with rank 16."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _try_milp_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of rank <= target_rank using MILP.
    We use the existing columns as a pool of atoms.
    """
    if time.time() > deadline:
        return None
    
    atoms = []
    rank = U.shape[1]
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        if np.any(u != 0) and np.any(v != 0) and np.any(w != 0):
            atoms.append((u, v, w))
            
    if len(atoms) < target_rank:
        return None
        
    num_atoms = len(atoms)
    num_constraints = n * m * p
    
    A = np.zeros((num_constraints, num_atoms))
    for c, (u, v, w) in enumerate(atoms):
        u3 = u.reshape(n, m)
        v3 = v.reshape(m, p)
        w3 = w.reshape(p, n)
        t_c = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
        A[:, c] = t_c.flatten()
        
    b = np.ones(num_constraints)
    
    try:
        c_obj = np.ones(num_atoms)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_atoms)
        
        # We add a constraint to limit the number of atoms to target_rank
        # This helps the solver prune the search space
        # sum(x) <= target_rank
        # This is a linear constraint: [1, 1, ..., 1] x <= target_rank
        sum_constraint = LinearConstraint(np.ones((1, num_atoms)), -np.inf, target_rank)
        
        res = milp(c=c_obj,
                   constraints=[LinearConstraint(A, b, b), sum_constraint],
                   integrality=integrality,
                   bounds=bounds)
        
        if res.success:
            x = res.x
            x_int = np.round(x).astype(int)
            
            new_rank = np.sum(x_int)
            if new_rank > target_rank:
                return None
                
            U_new = np.zeros((n*m, new_rank), dtype=np.int64)
            V_new = np.zeros((m*p, new_rank), dtype=np.int64)
            W_new = np.zeros((p*n, new_rank), dtype=np.int64)
            
            idx = 0
            for c in range(num_atoms):
                if x_int[c] == 1:
                    u, v, w = atoms[c]
                    U_new[:, idx] = u
                    V_new[:, idx] = v
                    W_new[:, idx] = w
                    idx += 1
            
            if _verify(U_new, V_new, W_new, n, m, p, new_rank):
                return U_new, V_new, W_new, new_rank
            else:
                return None
        else:
            return None
            
    except Exception:
        return None

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115
    
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Copy Strassen part
    U[:, :49] = U_s
    for j in range(m):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
    
    # Copy extra column part
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
    
    if not _verify(U, V, W, n, m, p, total_rank):
        # Fallback
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try simple rank reduction first
    if time.time() < deadline - 10:
        U_red, V_red, W_red, new_rank = _try_rank_reduction(U, V, W, n, m, p, total_rank, deadline - 10)
        if U_red is not None and new_rank < total_rank:
            U, V, W, total_rank = U_red, V_red, W_red, new_rank
    
    # Try MILP search for lower rank
    if time.time() < deadline - 5:
        U_milp, V_milp, W_milp, rank_milp = _try_milp_reduction(U, V, W, n, m, p, total_rank - 1, deadline - 5)
        if U_milp is not None and rank_milp < total_rank:
            U, V, W, total_rank = U_milp, V_milp, W_milp, rank_milp
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

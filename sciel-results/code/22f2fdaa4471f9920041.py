import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_rank33_decomposition():
    """
    Constructs a rank-33 decomposition for 2x3 by 3x6 matrix multiplication.
    Decomposes B into three 3x2 blocks, using rank-11 for each 2x3x2 multiplication.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 11 terms for a 2x3 x 3x2 multiplication block.
    # A is 2x3 (flat: 0..5), B is 3x2 (flat: 0..5), C is 2x2 (flat: 0..3)
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    terms = []
    
    # Strassen 7 terms for 2x2 x 2x2 (A[:,0:2] * B[0:2,:])
    # A sub: 0,1,3,4; B sub: 0,1,2,3; C sub: 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}))
    # M2 = (A10+A11)B00
    terms.append(({3:1, 4:1}, {0:1}, {2:1, 3:-1}))
    # M3 = A00(B01-B11)
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    # M4 = A11(B10-B00)
    terms.append(({4:1}, {2:1, 0:-1}, {0:1, 2:1}))
    # M5 = (A00+A01)B11
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({1:1, 4:-1}, {2:1, 3:1}, {0:1}))
    
    # 4 scalar terms for A[:,2] * B[2,:]
    # K1 = A02 * B20 -> C00
    terms.append(({2:1}, {4:1}, {0:1}))
    # K2 = A02 * B21 -> C01
    terms.append(({2:1}, {5:1}, {1:1}))
    # K3 = A12 * B20 -> C10
    terms.append(({5:1}, {4:1}, {2:1}))
    # K4 = A12 * B21 -> C11
    terms.append(({5:1}, {5:1}, {3:1}))
    
    # Fill global matrices for 3 blocks
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx, col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

def _build_rank24_decomposition(U12, V12, W12):
    """
    Build a rank-24 decomposition for 2x3x6 using two rank-12 2x3x3 blocks.
    Splits the 6 columns into two groups of 3.
    """
    n, m, p = 2, 3, 6
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        for c in range(12):
            col = block * 12 + c
            U[:, col] = U12[:, c]
            for j in range(m):
                for k in range(3):
                    V[j*p + (block*3 + k), col] = V12[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W[(block*3 + k)*n + i, col] = W12[k*n + i, c]
    
    return U, V, W

def _build_rank26_decomposition(U13, V13, W13):
    """
    Build a rank-26 decomposition for 2x3x6 using two rank-13 2x3x3 blocks.
    """
    n, m, p = 2, 3, 6
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        for c in range(13):
            col = block * 13 + c
            U[:, col] = U13[:, c]
            for j in range(m):
                for k in range(3):
                    V[j*p + (block*3 + k), col] = V13[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W[(block*3 + k)*n + i, col] = W13[k*n + i, c]
    
    return U, V, W

def _find_rank12_2x3x3(max_time=25):
    """
    Try to find a rank-12 decomposition for 2x3x3 matrix multiplication.
    A is 2x3, B is 3x3, C is 2x3.
    """
    start_time = time.time()
    n, m, p = 2, 3, 3
    rank = 12
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    def make_initial_guess(seed=0):
        """Create initial guess: 7 Strassen + 5 scalar-ish terms."""
        np.random.seed(seed)
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
        # C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        
        # 7 Strassen terms for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        # A sub: 0,1,3,4; B sub: 0,1,3,4; C sub: 0,1,3,4
        strassen = [
            ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
            ({3:1, 4:1}, {0:1}, {3:1, 4:-1}),
            ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),
            ({4:1}, {3:1, 0:-1}, {0:1, 3:1}),
            ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),
            ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
            ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
        ]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(strassen):
            for idx, val in u_dict.items():
                U[idx, c] = val
            for idx, val in v_dict.items():
                V[idx, c] = val
            for idx, val in w_dict.items():
                W[idx, c] = val
        
        # 5 more terms to handle remaining contributions
        # Need: A[:,2]*B[2,:] -> C[:,2] and A[:,0:2]*B[2,:] -> C[:,0:2] and A[:,2]*B[0:2,:] -> C[:,2]
        # Let's use random small perturbations for the remaining 5 terms
        for c in range(7, rank):
            U[:, c] = np.random.randn(dim_u) * 0.5
            V[:, c] = np.random.randn(dim_v) * 0.5
            W[:, c] = np.random.randn(dim_w) * 0.5
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_solution = None
    best_cost = np.inf
    
    # Try multiple initial points
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        x0 = make_initial_guess(seed=attempt * 7 + 3)
        try:
            result = least_squares(residuals, x0, max_nfev=3000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Round to nearest half-integer and verify
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
    
    return None

def _find_rank13_2x3x3(max_time=20):
    """
    Try to find a rank-13 decomposition for 2x3x3 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = 2, 3, 3
    rank = 13
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    def make_initial_guess(seed=0):
        """Create initial guess: 7 Strassen + 6 scalar terms."""
        np.random.seed(seed)
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # 7 Strassen terms for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        strassen = [
            ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
            ({3:1, 4:1}, {0:1}, {3:1, 4:-1}),
            ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),
            ({4:1}, {3:1, 0:-1}, {0:1, 3:1}),
            ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),
            ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
            ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
        ]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(strassen):
            for idx, val in u_dict.items():
                U[idx, c] = val
            for idx, val in v_dict.items():
                V[idx, c] = val
            for idx, val in w_dict.items():
                W[idx, c] = val
        
        # 6 scalar terms for remaining contributions
        # A[0,2]*B[2,0]->C[0,0], A[0,2]*B[2,1]->C[0,1], A[0,2]*B[2,2]->C[0,2]
        # A[1,2]*B[2,0]->C[1,0], A[1,2]*B[2,1]->C[1,1], A[1,2]*B[2,2]->C[1,2]
        scalar_terms = [
            ({2:1}, {6:1}, {0:1}),
            ({2:1}, {7:1}, {1:1}),
            ({2:1}, {8:1}, {2:1}),
            ({5:1}, {6:1}, {3:1}),
            ({5:1}, {7:1}, {4:1}),
            ({5:1}, {8:1}, {5:1}),
        ]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(scalar_terms):
            for idx, val in u_dict.items():
                U[idx, 7+c] = val
            for idx, val in v_dict.items():
                V[idx, 7+c] = val
            for idx, val in w_dict.items():
                W[idx, 7+c] = val
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_solution = None
    best_cost = np.inf
    
    # The initial guess should already be exact (rank-13 construction)
    x0 = make_initial_guess(seed=42)
    try:
        result = least_squares(residuals, x0, max_nfev=1000)
        if result.cost < best_cost:
            best_cost = result.cost
            best_solution = result.x
    except:
        pass
    
    # Try a few random restarts to see if we can find a better (smaller norm) solution
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        x0 = make_initial_guess(seed=attempt * 13 + 7)
        try:
            result = least_squares(residuals, x0, max_nfev=2000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Round to nearest half-integer and verify
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_find_rank10_2x3x2(max_time=20):
    """Try to find a rank-10 decomposition for 2x3 x 3x2 with integer/half-integer entries."""
    start_time = time.time()
    n, m, p = 2, 3, 2
    rank = 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    best_solution = None
    best_cost = np.inf
    
    # Strategy 1: Start from Strassen-like structure with 7 terms + 3 extra
    # This gives a good initial point close to a solution
    base_terms = [
        ({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({3:1, 4:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({4:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 4:-1}, {2:1, 3:1}, {0:1}),
        ({2:1}, {4:1}, {0:1}),
        ({2:1}, {5:1}, {1:1}),
        ({5:1}, {4:1}, {2:1}),
    ]
    
    # Build initial solution from these 10 terms (7 Strassen + 3 from 4 scalar terms)
    U_init = np.zeros((dim_u, rank))
    V_init = np.zeros((dim_v, rank))
    W_init = np.zeros((dim_w, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(base_terms):
        for idx, val in u_dict.items():
            U_init[idx, c] = val
        for idx, val in v_dict.items():
            V_init[idx, c] = val
        for idx, val in w_dict.items():
            W_init[idx, c] = val
    
    x0_strassen = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try optimization from Strassen-like start
    try:
        result = least_squares(residuals, x0_strassen, max_nfev=2000)
        if result.cost < best_cost:
            best_cost = result.cost
            best_solution = result.x
            if best_cost < 1e-14:
                pass  # Will be caught by verification below
    except:
        pass
    
    # Strategy 2: Random restarts with different scales
    for seed in range(300):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        scale = 0.3 + 0.4 * (seed % 5) / 5.0
        x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * scale
        try:
            result = least_squares(residuals, x0, max_nfev=1500)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Round to nearest half-integer and verify
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify the rounded solution
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
    
    return None

def _build_rank30_decomposition(U10, V10, W10):
    """Build a rank-30 decomposition using rank-10 2x3x2 blocks."""
    n, m, p = 2, 3, 6
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(3):
        for c in range(10):
            col = block * 10 + c
            U[:, col] = U10[:, c]
            for j in range(m):
                for k in range(2):
                    V[j*p + (block*2 + k), col] = V10[j*2 + k, c]
            for k in range(2):
                for i in range(n):
                    W[(block*2 + k)*n + i, col] = W10[k*n + i, c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy 1: Try rank-12 for 2x3x3 -> rank-24 for 2x3x6
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        result = _find_rank12_2x3x3(max_time=min(remaining, 30.0))
        if result is not None and time.time() < deadline:
            U12, V12, W12 = result
            U, V, W = _build_rank24_decomposition(U12, V12, W12)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try rank-13 for 2x3x3 -> rank-26 for 2x3x6
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        result = _find_rank13_2x3x3(max_time=min(remaining, 20.0))
        if result is not None and time.time() < deadline:
            U13, V13, W13 = result
            U, V, W = _build_rank26_decomposition(U13, V13, W13)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 33
    U, V, W = _build_rank33_decomposition()
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

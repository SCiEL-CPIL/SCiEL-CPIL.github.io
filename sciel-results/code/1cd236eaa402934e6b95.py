"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _target_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D array T[a,b,c] where
    a = i*m+j (U index), b = j*p+k (V index), c = k*n+i (W index).
    T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _decompose_from_factors(U, V, W):
    """Given factor matrices, return the reconstructed tensor."""
    rank = U.shape[1]
    # T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly matches the target tensor."""
    T_target = _target_tensor(n, m, p)
    T_recon = _decompose_from_factors(U, V, W)
    return np.max(np.abs(T_target - T_recon)) < tol

def _try_rank(n, m, p, rank, n_trials=50, time_limit=12.0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    T_target = _target_tensor(n, m, p)
    T_flat = T_target.reshape(-1)
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    def objective(params):
        # params: [U_flat (dim_u*rank), V_flat (dim_v*rank), W_flat (dim_w*rank)]
        u = dim_u * rank
        v = u + dim_v * rank
        U = params[:u].reshape(dim_u, rank)
        V = params[u:v].reshape(dim_v, rank)
        W = params[v:].reshape(dim_w, rank)
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for c in range(rank):
            T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        diff = T.reshape(-1) - T_flat
        return np.sum(diff * diff)
    
    best_obj = np.inf
    best_params = None
    start = time.time()
    
    for trial in range(n_trials):
        if time.time() - start > time_limit:
            break
        # Random initialization
        scale = 1.0 / (rank ** 0.5)
        params0 = np.random.randn(dim_u * rank + dim_v * rank + dim_w * rank) * scale
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                if best_obj < 1e-12:
                    break
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    u = dim_u * rank
    v = u + dim_v * rank
    U = best_params[:u].reshape(dim_u, rank)
    V = best_params[u:v].reshape(dim_v, rank)
    W = best_params[v:].reshape(dim_w, rank)
    
    # Round to nearest half-integer and verify
    U_round = np.round(U * 2.0) / 2.0
    V_round = np.round(V * 2.0) / 2.0
    W_round = np.round(W * 2.0) / 2.0
    
    if _verify_exact(U_round, V_round, W_round, n, m, p, tol=1e-9):
        return (U_round, V_round, W_round)
    
    # Also try rounding to integers
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify_exact(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return (U_int, V_int, W_int)
    
    return None

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
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

def _hybrid_strassen_schoolbook(n, m, p):
    """Hybrid decomposition using Strassen for 2x2x2 blocks + schoolbook for rest.
    
    For 2x3x7: split j into {0,1} and {2}.
    - Block A: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (7 terms)
    - Block B: i in {0,1}, j in {2}, k in {0,1} -> 2x1x2, schoolbook (4 terms)  
    - Block C: i in {0,1}, j in {0,1,2}, k in {2,3,4,5,6} -> 2x3x5, schoolbook (30 terms)
    Total: 7 + 4 + 30 = 41. Not better than 42.
    
    Alternative: split k into {0,1} and {2,3,4,5,6}
    - Block A: i in {0,1}, j in {0,1,2}, k in {0,1} -> 2x3x2, schoolbook (12 terms)
    - Block B: i in {0,1}, j in {0,1,2}, k in {2,3,4,5,6} -> 2x3x5, schoolbook (30 terms)
    Total: 42. Same.
    
    Let's try a different approach: exploit that n=2 is small.
    For fixed k, we have a 2x3 matrix times 3x1 vector = 2x1 vector.
    This is just 6 multiplications per k, so 6*7=42 total. No savings.
    
    The Strassen insight: for 2x2x2, we can do it in 7 multiplications instead of 8.
    But we need to carefully handle the indexing.
    """
    # Let's construct a valid decomposition with rank < 42 if possible
    # For 2x3x7, the minimum known rank is likely around 30-35 based on literature
    # Let's try to find one via the search above
    
    # If search fails, return schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-START
def _strassen_schoolbook_hybrid(n, m, p):
    """Hybrid Strassen/Schoolbook: split inner dim into 2x2 blocks where possible."""
    # For 2x3x7: split m=3 into 2+1, use Strassen on 2x2x2 block, schoolbook on rest
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        rank += 1
    
    # Split: i in {0,1}, j in {0,1,2}, k in {0,...,6}
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> use Strassen (7 terms)
    # Block 2: i in {0,1}, j in {2}, k in {0,1} -> schoolbook (4 terms)
    # Block 3: i in {0,1}, j in {0,1,2}, k in {2,3,4,5,6} -> schoolbook (2*3*5=30 terms)
    
    # Strassen for 2x2x2 block (i=0,1; j=0,1; k=0,1)
    # Standard Strassen: 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our tensor: U index = i*m+j, V index = j*p+k, W index = k*n+i
    # For the 2x2x2 block with i in {0,1}, j in {0,1}, k in {0,1}:
    # U indices: 0*m+0=0, 0*m+1=1, 1*m+0=3, 1*m+1=4
    # V indices: 0*p+0=0, 0*p+1=1, 1*p+0=7, 1*p+1=8
    # W indices: 0*n+0=0, 1*n+0=2, 0*n+1=1, 1*n+1=3
    
    # Strassen term 1: (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # Contributes to C[0,0], C[0,1], C[1,0], C[1,1] with signs
    # This is complex; let's just use schoolbook for all and see if we can find
    # a better decomposition via lattice reduction
    
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)

def _try_lattice_reduction(n, m, p, target_rank, time_limit=30.0):
    """Use lattice reduction (LLL) to find sparse low-rank decompositions.
    
    Embed the tensor constraints into a lattice and use LLL to find
    short vectors that correspond to valid rank-1 terms.
    """
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_target = _target_tensor(n, m, p)
    T_flat = T_target.reshape(-1)
    
    # Strategy: for each candidate rank, try to find U,V,W with small integer/half-integer entries
    # that satisfy the tensor equation. Use random initialization + local optimization
    # with LLL-inspired sparsification.
    
    best_result = None
    start = time.time()
    
    for trial in range(20):
        if time.time() - start > time_limit:
            break
        
        # Random sparse initialization
        scale = 1.0
        U = np.zeros((dim_u, target_rank))
        V = np.zeros((dim_v, target_rank))
        W = np.zeros((dim_w, target_rank))
        
        for c in range(target_rank):
            # Random sparse vectors with small integer entries
            n_nonzero_u = np.random.randint(1, max(2, dim_u//2 + 1))
            n_nonzero_v = np.random.randint(1, max(2, dim_v//2 + 1))
            n_nonzero_w = np.random.randint(1, max(2, dim_w//2 + 1))
            
            u_idx = np.random.choice(dim_u, n_nonzero_u, replace=False)
            v_idx = np.random.choice(dim_v, n_nonzero_v, replace=False)
            w_idx = np.random.choice(dim_w, n_nonzero_w, replace=False)
            
            U[u_idx, c] = np.random.choice([-1, 1, 2, -2], n_nonzero_u)
            V[v_idx, c] = np.random.choice([-1, 1, 2, -2], n_nonzero_v)
            W[w_idx, c] = np.random.choice([-1, 1, 2, -2], n_nonzero_w)
        
        # Local optimization to reduce error
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            u_end = dim_u * target_rank
            v_end = u_end + dim_v * target_rank
            U_l = params[:u_end].reshape(dim_u, target_rank)
            V_l = params[u_end:v_end].reshape(dim_v, target_rank)
            W_l = params[v_end:].reshape(dim_w, target_rank)
            T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
            for c in range(target_rank):
                T += np.einsum('a,b,c->abc', U_l[:,c], V_l[:,c], W_l[:,c])
            diff = T.reshape(-1) - T_flat
            return np.sum(diff * diff)
        
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < 1e-8:
                u_end = dim_u * target_rank
                v_end = u_end + dim_v * target_rank
                U_opt = res.x[:u_end].reshape(dim_u, target_rank)
                V_opt = res.x[u_end:v_end].reshape(dim_v, target_rank)
                W_opt = res.x[v_end:].reshape(dim_w, target_rank)
                
                # Try to sparsify: round to nearest half-integer and zero out small values
                U_round = np.round(U_opt * 2.0) / 2.0
                V_round = np.round(V_opt * 2.0) / 2.0
                W_round = np.round(W_opt * 2.0) / 2.0
                
                # Zero out values below threshold
                threshold = 0.1
                U_round[np.abs(U_round) < threshold] = 0
                V_round[np.abs(V_round) < threshold] = 0
                W_round[np.abs(W_round) < threshold] = 0
                
                if _verify_exact(U_round, V_round, W_round, n, m, p, tol=1e-9):
                    return U_round, V_round, W_round
                
                # Try integer rounding
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if _verify_exact(U_int, V_int, W_int, n, m, p, tol=1e-9):
                    return U_int, V_int, W_int
                
        except Exception:
            continue
    
    return None

def _try_rank_improved(n, m, p, rank, n_trials=20, time_limit=15.0):
    """Improved rank search with better initialization and sparsification."""
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_target = _target_tensor(n, m, p)
    T_flat = T_target.reshape(-1)
    
    best_obj = np.inf
    best_params = None
    start = time.time()
    
    for trial in range(n_trials):
        if time.time() - start > time_limit:
            break
        
        # Mixed initialization: some random, some structured
        if trial % 3 == 0:
            # Structured: try to exploit tensor structure
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            for c in range(rank):
                # Random but with some structure
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                # Add random perturbations
                U[np.random.choice(dim_u, 3, replace=False), c] += np.random.randn(3) * 0.5
                V[np.random.choice(dim_v, 3, replace=False), c] += np.random.randn(3) * 0.5
                W[np.random.choice(dim_w, 3, replace=False), c] += np.random.randn(3) * 0.5
        elif trial % 3 == 1:
            # Random with small integers
            U = np.random.choice([-1, 0, 1, 2, -2], (dim_u, rank))
            V = np.random.choice([-1, 0, 1, 2, -2], (dim_v, rank))
            W = np.random.choice([-1, 0, 1, 2, -2], (dim_w, rank))
        else:
            # Pure random
            scale = 1.0 / (rank ** 0.5)
            U = np.random.randn(dim_u, rank) * scale
            V = np.random.randn(dim_v, rank) * scale
            W = np.random.randn(dim_w, rank) * scale
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            u_end = dim_u * rank
            v_end = u_end + dim_v * rank
            U_l = params[:u_end].reshape(dim_u, rank)
            V_l = params[u_end:v_end].reshape(dim_v, rank)
            W_l = params[v_end:].reshape(dim_w, rank)
            T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
            for c in range(rank):
                T += np.einsum('a,b,c->abc', U_l[:,c], V_l[:,c], W_l[:,c])
            diff = T.reshape(-1) - T_flat
            return np.sum(diff * diff)
        
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                if best_obj < 1e-12:
                    break
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    u_end = dim_u * rank
    v_end = u_end + dim_v * rank
    U = best_params[:u_end].reshape(dim_u, rank)
    V = best_params[u_end:v_end].reshape(dim_v, rank)
    W = best_params[v_end:].reshape(dim_w, rank)
    
    # Try multiple rounding strategies
    for mult in [1.0, 2.0]:
        U_round = np.round(U * mult) / mult
        V_round = np.round(V * mult) / mult
        W_round = np.round(W * mult) / mult
        
        # Zero out small values
        for threshold in [0.01, 0.05, 0.1]:
            U_t = U_round.copy()
            V_t = V_round.copy()
            W_t = W_round.copy()
            U_t[np.abs(U_t) < threshold] = 0
            V_t[np.abs(V_t) < threshold] = 0
            W_t[np.abs(W_t) < threshold] = 0
            
            if _verify_exact(U_t, V_t, W_t, n, m, p, tol=1e-9):
                return (U_t, V_t, W_t)
        
        if _verify_exact(U_round, V_round, W_round, n, m, p, tol=1e-9):
            return (U_round, V_round, W_round)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Known lower bound: max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, n*p)
    schoolbook_rank = n * m * p  # 42
    
    # First, try to find a good decomposition using the improved search
    # Start from lower bound and go up
    best_result = None
    
    # Try a range of ranks with improved search
    ranks_to_try = list(range(lower_bound, min(lower_bound + 15, schoolbook_rank + 1)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 5:
            break
        
        # Use improved search with time allocation
        time_alloc = min(remaining_time * 0.8, 20.0)
        result = _try_rank_improved(n, m, p, rank, n_trials=15, time_limit=time_alloc)
        if result is not None:
            U, V, W = result
            best_result = (rank, U, V, W)
            break  # Found a valid decomposition, return it
    
    # If no result from improved search, try lattice reduction approach
    if best_result is None:
        if time.time() < deadline - 15:
            for rank in range(lower_bound, min(lower_bound + 10, schoolbook_rank + 1)):
                if time.time() > deadline - 10:
                    break
                result = _try_lattice_reduction(n, m, p, rank, time_limit=10.0)
                if result is not None:
                    U, V, W = result
                    best_result = (rank, U, V, W)
                    break
    
    # If still no result, use the hybrid approach or schoolbook
    if best_result is not None:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition as a fallback."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1 k=0,1 (7 terms)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
    
    # Part 4: Strassen for j=0,1 k=2,3 (7 terms)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
    
    return U, V, W

def _tensor_to_vector(U, V, W, n, m, p):
    """Convert a decomposition to the full tensor vector."""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    tensor = np.zeros((n * m * p,))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                tensor[idx] = val
    return tensor

def _target_tensor(n, m, p):
    """The target multiplication tensor."""
    tensor = np.zeros((n * m * p,))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m * p + j * p + k] = 1.0
    return tensor

def _try_rank_reduction_from_rank28(n, m, p, target_rank, max_time=30):
    """
    Try to find a rank-target_rank decomposition starting from rank-28.
    Uses structured approach: try to merge/eliminate terms.
    """
    start_time = time.time()
    
    U28, V28, W28 = _build_rank28()
    U28_arr = np.array([[float(x) for x in row] for row in U28], dtype=float)
    V28_arr = np.array([[float(x) for x in row] for row in V28], dtype=float)
    W28_arr = np.array([[float(x) for x in row] for row in W28], dtype=float)
    
    # Try to find a rank-27 decomposition by optimizing
    # Strategy: take rank-28, zero out one term's contribution, and optimize
    # to see if the remaining 27 terms can compensate
    
    best_obj = np.inf
    best_params = None
    
    num_trials = 5
    for trial in range(num_trials):
        if time.time() - start_time > max_time - 5:
            break
        
        np.random.seed(trial * 7919 + 3)
        
        # Initialize from rank-28 with small perturbations
        # Use all 28 terms but allow the optimizer to drive one toward zero
        x0_scale = 1.0
        x0 = np.concatenate([
            U28_arr.flatten(),
            V28_arr.flatten(),
            W28_arr.flatten()
        ]) * x0_scale
        
        # Add small random perturbation to help escape
        x0 += np.random.randn(len(x0)) * 0.01
        
        def objective(params):
            U = params[:6 * target_rank].reshape(6, target_rank)
            V = params[6 * target_rank:6 * target_rank + 15 * target_rank].reshape(15, target_rank)
            W = params[6 * target_rank + 15 * target_rank:].reshape(10, target_rank)
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        residual += (val - 1.0) ** 2
            return residual
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                if res.fun < 1e-8:
                    break
        except Exception:
            continue
    
    if best_obj < 1e-6 and best_params is not None:
        U = best_params[:6 * target_rank].reshape(6, target_rank)
        V = best_params[6 * target_rank:6 * target_rank + 15 * target_rank].reshape(15, target_rank)
        W = best_params[6 * target_rank + 15 * target_rank:].reshape(10, target_rank)
        
        # Convert to list format
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def _try_structured_rank27(n, m, p, max_time=40):
    """
    Try structured approach for rank-27:
    The rank-28 has: 10 (j=2) + 4 (j=0,1,k=4) + 7 (Strassen j=0,1,k=0,1) + 7 (Strassen j=0,1,k=2,3)
    
    Idea: Maybe the 4 terms for k=4 can be reduced to 3 using a trick,
    or one of the Strassen blocks can be reduced.
    
    Alternative: Try to find if the 10 terms for j=2 can be reduced.
    For j=2, we need: sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1 for all i,k
    This is a 2x5 matrix of 1s, which has rank 1 in the usual sense,
    but in tensor decomposition it takes 10 terms in the naive approach.
    However, we can do it in fewer terms!
    
    Actually, for fixed j=2, we need U[i,2,r] * V[2,k,r] * W[k,i,r] to sum to 1.
    If we set U[i,2,r] = a_i, V[2,k,r] = b_k, W[k,i,r] = c_{k,i},
    then sum_r a_i b_k c_{k,i} = 1.
    
    With rank-1: a_i b_k c_{k,i} = 1 for all i,k. This requires a_i b_k c_{k,i} = 1.
    If we set a_i = 1, b_k = 1, c_{k,i} = 1, that works! But that's 1 term.
    Wait, no - each term r contributes U[i,2,r]*V[2,k,r]*W[k,i,r].
    If we have ONE term with U[i,2,0]=1 for all i, V[2,k,0]=1 for all k, W[k,i,0]=1 for all k,i,
    then sum = 1*1*1 = 1. So j=2 can be done in rank 1!
    
    But wait, that's not right either. U[i,2,0]=1 means the same value for all i.
    V[2,k,0]=1 means the same value for all k.
    W[k,i,0]=1 means the same value for all k,i.
    Then U[i,2,0]*V[2,k,0]*W[k,i,0] = 1*1*1 = 1. Yes! Rank 1 suffices for j=2.
    
    So the total rank should be 1 + rank(2x2x5 tensor).
    The 2x2x5 tensor: for i in {0,1}, j in {0,1}, k in {0,...,4}.
    We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    The minimal rank of 2x2x5 multiplication is what we need to find.
    """
    start_time = time.time()
    
    # The j=2 part takes rank 1
    # The j=0,1 part is a 2x2x5 multiplication tensor
    # We need to find the minimal rank of the 2x2x5 tensor
    
    # Try ranks from 14 upward (total rank = 1 + sub_rank)
    # 2x2x5 has 2*2*5 = 20 entries, each must equal 1
    # Lower bound: at least 20/rank terms on average contribute, but that's not tight
    
    # Let's search for the 2x2x5 decomposition
    for sub_rank in [13, 14, 15, 16, 17, 18]:
        if time.time() - start_time > max_time - 10:
            break
        
        result = _search_2x2x5(sub_rank, max_time=8)
        if result is not None:
            U_sub, V_sub, W_sub = result
            total_rank = 1 + sub_rank
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Rank-1 term for j=2
            for i in range(n):
                for k in range(p):
                    U[i * m + 2][0] = 1
                    V[2 * p + k][0] = 1
                    W[k * n + i][0] = 1
            
            # Sub-tensor terms
            for r in range(sub_rank):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][1 + r] = U_sub[i * 2 + j, r]
                for j in range(2):
                    for k in range(5):
                        V[j * p + k][1 + r] = V_sub[j * 5 + k, r]
                for k in range(5):
                    for i in range(2):
                        W[k * n + i][1 + r] = W_sub[k * 2 + i, r]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _search_2x2x5(target_rank, max_time=8):
    """Search for a rank-target_rank decomposition of the 2x2x5 multiplication tensor."""
    start_time = time.time()
    n_sub, m_sub, p_sub = 2, 2, 5
    
    def objective(params):
        U = params[:target_rank * n_sub * m_sub].reshape(target_rank, n_sub * m_sub).T
        V = params[target_rank * n_sub * m_sub : target_rank * n_sub * m_sub + target_rank * m_sub * p_sub].reshape(target_rank, m_sub * p_sub).T
        W = params[target_rank * n_sub * m_sub + target_rank * m_sub * p_sub :].reshape(target_rank, p_sub * n_sub).T
        
        residual = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    val = np.dot(U[i * m_sub + j], V[j * p_sub + k] * W[k * n_sub + i])
                    residual += (val - 1.0) ** 2
        return residual
    
    best_obj = np.inf
    best_params = None
    
    for trial in range(8):
        if time.time() - start_time > max_time - 2:
            break
        
        np.random.seed(trial * 104729 + 11)
        x0 = np.random.randn(target_rank * (n_sub * m_sub + m_sub * p_sub + p_sub * n_sub)) * 0.5
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                if res.fun < 1e-8:
                    break
        except Exception:
            continue
    
    if best_obj < 1e-6 and best_params is not None:
        U = best_params[:target_rank * n_sub * m_sub].reshape(target_rank, n_sub * m_sub).T
        V = best_params[target_rank * n_sub * m_sub : target_rank * n_sub * m_sub + target_rank * m_sub * p_sub].reshape(target_rank, m_sub * p_sub).T
        W = best_params[target_rank * n_sub * m_sub + target_rank * m_sub * p_sub :].reshape(target_rank, p_sub * n_sub).T
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Fallback: Rank 28
    U28, V28, W28 = _build_rank28()
    if _verify_decomposition(U28, V28, W28, n, m, p):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28
    else:
        raise ValueError("Fallback failed")
        
    # Try structured rank-27 search
    if time.time() - start_time < deadline - 50:
        result = _try_structured_rank27(n, m, p, max_time=50)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = len(U[0])
    
    # Try direct rank-27 optimization from rank-28
    if best_rank > 27 and time.time() - start_time < deadline - 30:
        result = _try_rank_reduction_from_rank28(n, m, p, 27, max_time=30)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = 27
    
    return {"U": best_U, "V": best_V, "W": best_W}

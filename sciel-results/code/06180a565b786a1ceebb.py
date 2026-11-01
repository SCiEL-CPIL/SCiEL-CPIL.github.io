import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_linear_system_for_rank_reduction(n, m, p, base_U, base_V, base_W, target_rank):
    """
    Build a linear system to find a rank-reduced decomposition.
    
    Given a rank-R decomposition (U0, V0, W0), we want to find a rank-r decomposition
    (U, V, W) with r < R. We parameterize U, V, W as linear combinations of the base
    columns and solve for coefficients that make the tensor sum equal to the identity.
    
    The key insight: if we can express the rank-r decomposition as a linear combination
    of the rank-R terms with some coefficients being zero, we've found a lower rank solution.
    """
    R = base_U.shape[1]
    
    # We'll try to find a rank-target_rank solution by finding a linear dependency
    # among the R terms that allows us to eliminate (R - target_rank) terms.
    
    # Strategy: Assume the rank-r solution has the form:
    # U[:, r'] = sum_{s=0}^{R-1} alpha[s] * U0[:, s] for r' in 0..r-1
    # V[:, r'] = sum_{s=0}^{R-1} beta[s] * V0[:, s] for r' in 0..r-1  
    # W[:, r'] = sum_{s=0}^{R-1} gamma[s] * W0[:, s] for r' in 0..r-1
    
    # But this doesn't directly work. Instead, let's use a different approach.
    
    # We want to find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_{r'=0}^{r-1} U[:, r'] ⊗ V[:, r'] ⊗ W[:, r'] = identity tensor
    
    # Let's parameterize this as a linear system. The tensor equation is:
    # For each (i,j,k): sum_{r'=0}^{r-1} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # But it's not linear! It's a system of polynomial equations.
    
    # Alternative approach: Use the fact that if we have a rank-R decomposition,
    # we can look for a rank-r decomposition by finding a subspace.
    
    # Let's try a direct approach: set up the equations and use SVD to find
    # a solution with the minimum number of non-zero terms.
    
    # Actually, let's use a smarter approach. We know that the multiplication tensor
    # has a certain structure. Let's exploit it.
    
    # For 2x3x5, the schoolbook rank is 30. We have a rank-28 solution.
    # Let's see if we can find a rank-27 solution by looking for a linear dependency.
    
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r where each term is a rank-1 tensor.
    # If we can find coefficients c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0,
    # then we can eliminate one term.
    
    # But this is not quite right either. Let me think differently.
    
    # Let's use the approach of finding a rank-r decomposition by solving:
    # min ||sum_{r=0}^{r-1} u_r ⊗ v_r ⊗ w_r - T||_F
    # subject to some constraints that force sparsity.
    
    # For now, let's return None and handle this in the search function.
    return None

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    col = add_strassen(0, 1, 0, col)
    
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)
    
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def search_lower_rank(n, m, p, base_U, base_V, base_W, target_rank, time_budget):
    """
    Search for a lower-rank decomposition using algebraic null-space analysis.
    
    Strategy: Model the tensor decomposition as a structured linear system and
    use SVD/null-space to find if a rank-(target_rank) solution exists by
    identifying linear dependencies among the rank-R terms.
    """
    start_time = time.time()
    
    if target_rank >= base_U.shape[1]:
        return None
    
    R = base_U.shape[1]
    
    # Approach 1: Direct optimization with sparsity encouragement
    # Initialize with a random subset of columns
    np.random.seed(42)
    indices = np.random.choice(R, target_rank, replace=False)
    U0 = np.array(base_U[:, indices], dtype=float)
    V0 = np.array(base_V[:, indices], dtype=float)
    W0 = np.array(base_W[:, indices], dtype=float)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    num_u = n * m * target_rank
    num_v = m * p * target_rank
    num_w = p * n * target_rank
    
    def reshape(x):
        U = x[:num_u].reshape(n * m, target_rank)
        V = x[num_u:num_u+num_v].reshape(m * p, target_rank)
        W = x[num_u+num_v:].reshape(p * n, target_rank)
        return U, V, W
    
    def compute_tensor(U, V, W):
        """Efficiently compute the tensor sum."""
        tensor = np.zeros((n, m, p))
        for r in range(target_rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Use outer products for efficiency
            for i in range(n):
                for j in range(m):
                    tensor[i, j, :] += u[i*m+j] * (v[j*p:(j+1)*p] * w[0::n])
        return tensor
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return np.inf
        U, V, W = reshape(x)
        tensor = compute_tensor(U, V, W)
        err = tensor - 1.0
        return np.sum(err**2)
    
    def gradient(x):
        """Compute gradient of the objective function."""
        U, V, W = reshape(x)
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(target_rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = u[i*m+j] * v[j*p+k] * w[k*n+i] - 1.0/target_rank
                        # Actually, we need the full tensor error
                        pass
        
        # Simplified gradient: use finite differences or analytical
        # For now, return zeros and let L-BFGS-B use numerical gradients
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    best_obj = objective(x0)
    best_x = x0.copy()
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        
        if trial > 0:
            indices = np.random.choice(R, target_rank, replace=False)
            U0 = np.array(base_U[:, indices], dtype=float)
            V0 = np.array(base_V[:, indices], dtype=float)
            W0 = np.array(base_W[:, indices], dtype=float)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
    
    if best_obj < 1e-9:
        U, V, W = reshape(best_x)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        valid, _ = verify_decomposition(U_list, V_list, W_list, n, m, p)
        if valid:
            return U_list, V_list, W_list
    
    # Approach 2: Algebraic null-space search
    # Try to find a rank-target_rank solution by solving a linear system
    if time.time() - start_time < time_budget:
        result = algebraic_null_space_search(n, m, p, base_U, base_V, base_W, target_rank, time_budget)
        if result is not None:
            return result
    
    return None

def algebraic_null_space_search(n, m, p, base_U, base_V, base_W, target_rank, time_budget):
    """
    Use SVD/null-space to find a lower-rank decomposition.
    
    Key idea: If we have a rank-R decomposition, we can look for a rank-r solution
    by finding a subspace of the term space where the tensor sum still equals the identity.
    """
    start_time = time.time()
    R = base_U.shape[1]
    
    # Build the tensor contribution matrix
    # Each term r contributes u_r ⊗ v_r ⊗ w_r to the tensor
    # We want to find a subset of size target_rank that still sums to the identity
    
    # Alternative: Find coefficients c_r such that sum_r c_r * T_r = T_identity
    # where T_r = u_r ⊗ v_r ⊗ w_r
    
    # This is a linear system: A * c = b
    # where A has columns T_r (flattened) and b is the identity tensor (flattened)
    
    # But we want a sparse solution with exactly target_rank non-zero entries.
    
    # Let's try: find the null space of A and see if we can express the identity
    # using fewer terms.
    
    # Actually, the right approach is:
    # We have R terms. We want to find r < R terms that sum to the identity.
    # This is equivalent to finding a vector c in R^R with at most r non-zero entries
    # such that sum_r c_r * T_r = T_identity.
    
    # Let's set up the linear system and use SVD to find a minimum-norm solution,
    # then check if it's sparse.
    
    # Build the matrix A where column r is the flattened tensor T_r
    A = np.zeros((n*m*p, R))
    for r in range(R):
        u = base_U[:, r]
        v = base_V[:, r]
        w = base_W[:, r]
        T_r = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_r[i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        A[:, r] = T_r.flatten()
    
    b = np.ones(n*m*p)
    
    # Solve A * c = b using least squares
    try:
        c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    # Check if the solution is close to sparse
    # Count non-zero entries
    nonzero = np.sum(np.abs(c) > 1e-6)
    
    if nonzero <= target_rank:
        # We found a sparse solution!
        active_indices = np.where(np.abs(c) > 1e-6)[0]
        if len(active_indices) <= target_rank:
            # Extract the active terms and scale them
            U_new = np.zeros((n*m, target_rank))
            V_new = np.zeros((m*p, target_rank))
            W_new = np.zeros((p*n, target_rank))
            
            for idx, r in enumerate(active_indices):
                if idx < target_rank:
                    U_new[:, idx] = base_U[:, r] * c[r]
                    V_new[:, idx] = base_V[:, r]
                    W_new[:, idx] = base_W[:, r]
            
            # Verify
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            valid, _ = verify_decomposition(U_list, V_list, W_list, n, m, p)
            if valid:
                return U_list, V_list, W_list
    
    # Try finding a different sparse solution using null space
    # The general solution is c = c_particular + N * d where N is the null space
    try:
        N = null_space(A, rcond=1e-10)
    except Exception:
        return None
    
    if N.shape[1] > 0:
        # Try random directions in the null space
        np.random.seed(123)
        for trial in range(20):
            if time.time() - start_time > time_budget:
                break
            
            d = np.random.randn(N.shape[1])
            c_trial = c + N @ d
            
            # Check sparsity
            nonzero = np.sum(np.abs(c_trial) > 1e-6)
            if nonzero <= target_rank:
                active_indices = np.where(np.abs(c_trial) > 1e-6)[0]
                if len(active_indices) <= target_rank:
                    U_new = np.zeros((n*m, target_rank))
                    V_new = np.zeros((m*p, target_rank))
                    W_new = np.zeros((p*n, target_rank))
                    
                    for idx, r in enumerate(active_indices):
                        if idx < target_rank:
                            U_new[:, idx] = base_U[:, r] * c_trial[r]
                            V_new[:, idx] = base_V[:, r]
                            W_new[:, idx] = base_W[:, r]
                    
                    U_list = U_new.tolist()
                    V_list = V_new.tolist()
                    W_list = W_new.tolist()
                    valid, _ = verify_decomposition(U_list, V_list, W_list, n, m, p)
                    if valid:
                        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_rank = float('inf')
    
    # First, try the known rank-28 solution
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = 28
    except Exception as e:
        pass
    
    # Try to find lower rank solutions
    if best_result is not None:
        for target_rank in [27, 26, 25]:
            if time.time() > deadline:
                break
            
            time_budget = min(deadline - time.time(), 40.0)
            if time_budget < 5.0:
                break
            
            try:
                result = search_lower_rank(n, m, p, 
                                          np.array(best_result['U']), 
                                          np.array(best_result['V']), 
                                          np.array(best_result['W']),
                                          target_rank, time_budget)
                if result is not None:
                    U, V, W = result
                    best_result = {'U': U, 'V': V, 'W': W}
                    best_rank = target_rank
                    break
            except Exception as e:
                pass
    
    # Fallback: Schoolbook (Rank 30)
    if best_result is None:
        rank_school = n * m * p
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}

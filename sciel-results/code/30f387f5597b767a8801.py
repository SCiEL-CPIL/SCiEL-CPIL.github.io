import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    # Check if U @ V^T approximates the tensor in W^T
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # We need T[k*n+i, c]... wait.
    # The tensor is defined by W[k*n+i, c].
    # The product C = A @ B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # The decomposition is T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}?
    # No, the standard format is:
    # The matrix multiplication tensor T is a 3-tensor.
    # T_{i,j,k} corresponds to the term A_{i,j} * B_{j,k} in C_{i,k}.
    # A rank-1 term is u_i * v_j * w_k.
    # The provided interface returns U, V, W.
    # U has shape (n*m, rank). V has shape (m*p, rank). W has shape (p*n, rank).
    # This suggests the decomposition of the tensor T:
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} ??
    # No, that would be 4 indices.
    # Let's look at the schoolbook code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # If we sum over c, for a fixed i,j,k, only one c is non-zero.
    # That c corresponds to the triple (i,j,k).
    # So the term is 1*1*1 = 1 for the correct (i,j,k).
    # This means the tensor T_{i,j,k} is 1 if the indices match the column, 0 otherwise.
    # Wait, the matrix multiplication tensor T_{i,j,k} is 1 for all i,j,k?
    # No, T_{i,j,k} represents the bilinear form A_{i,j} B_{j,k}.
    # The "tensor" usually refers to the structure of the multiplication.
    # The condition for exact decomposition is:
    # For all A, B, C = A @ B, we have C_{i,k} = sum_c (sum_j A_{i,j} U_{i*m+j, c}) (sum_j B_{j,k} V_{j*p+k, c}) W_{k*n+i, c}?
    # No, that's not right either.
    # Let's re-read the standard definition.
    # A rank-1 decomposition of matmul (n, m, p) consists of matrices U, V, W such that:
    # C_{i,k} = sum_c U_{i, c} W_{k, c} (sum_j A_{i,j} V_{j, c})? No.
    #
    # Let's look at the schoolbook again.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # The check usually performed by evaluators for this interface is:
    # Let T be the tensor T_{i,j,k} = 1. (Wait, no. The tensor for matmul is T_{i,j,k} = 1? No.)
    # The tensor T is defined by T_{i,j,k} = 1 if we consider the map A -> A @ B.
    # Actually, the tensor T is such that C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is a trilinear map? No, bilinear in A and B.
    # The tensor T is in V^* x W^* x Z.
    # T(a, b) = a @ b.
    # T_{i,j,k} (where i index of A, j index of B? No, j is shared).
    # Indices: i (row of A), j (col of A / row of B), k (col of B).
    # T_{i,j,k} = 1.
    # Wait, T_{i,j,k} is 1 for all i, j, k?
    # If A = e_i e_j^T and B = e_j e_k^T, then C = e_i e_k^T.
    # So the coefficient of e_i e_k^T in C is 1.
    # This implies the tensor component T_{i,j,k} is 1.
    # So the tensor T is the all-ones tensor?
    # No, that would imply A @ B = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_i e_k^T.
    # Yes! The tensor T for matrix multiplication is indeed the all-ones tensor in the indices (i,j,k).
    #
    # So the check is:
    # Does sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} == 1 for all i,j,k?
    # Let's check the schoolbook.
    # For a specific (i,j,k), only one column c has non-zero entries.
    # In that column, U=1, V=1, W=1. Product = 1.
    # For other columns, at least one is 0.
    # So sum is 1.
    #
    # So the condition is:
    # For all i, j, k: sum_{c=1}^R U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    #
    # We need to find R minimal such that this holds.
    #
    # Let's implement a check for this.
    
    # We can vectorize this check.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # We want to compute T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    
    # This is tricky to vectorize directly without expanding to full tensor size (n*m*p*R).
    # n=4, m=4, p=9. Size = 144 * R.
    # If R=144, size is ~20k. Very fast.
    # If R is smaller, even faster.
    
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R).
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # We need to sum over c (last axis).
    # But the indices must match: i from U, j from U/V, k from V/W, i from W.
    # Wait, W index is k*n+i. So W depends on k and i.
    # U depends on i and j.
    # V depends on j and k.
    
    # Let's iterate over i, j, k and check.
    # Since n, m, p are small, this is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_reshaped[i, j, :]
                v_val = V_reshaped[j, k, :]
                w_val = W_reshaped[k, i, :]
                
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, time_limit=140):
    """
    Attempts to find a decomposition of rank target_rank.
    Uses randomized search with local optimization.
    """
    start_time = time.time()
    
    # We want U (n*m, R), V (m*p, R), W (p*n, R).
    # Initialize with random matrices.
    # Since we need integer/half-integer entries, we can search over integers or halves.
    # But continuous optimization might find a solution that is close to integer.
    
    best_score = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Use a smooth objective function for optimization.
    # Objective: sum_{i,j,k} (1 - sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c])^2
    # We can minimize this.
    
    # We'll use a simple gradient-free method or random restarts with gradient descent.
    # Given the small dimensions, we can try to solve it directly.
    
    # Let's try a few random initializations and use Nelder-Mead or similar.
    # SciPy is available.
    
    import scipy.optimize as opt
    
    def objective(params):
        # params is a flat vector of size (n*m*R + m*p*R + p*n*R)
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    s = np.dot(U[i,j,:], V[j,k,:]*W[k,i,:])
                    err += (1.0 - s)**2
        return err
    
    # To make it faster, we can precompute some parts or use vectorization.
    # But for small R, the loop is okay.
    
    # We will perform a few random restarts.
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize params
        # Scale matters. Let's try random values in [-1, 1].
        total_size = n*m*target_rank + m*p*target_rank + p*n*target_rank
        x0 = np.random.randn(total_size) * 0.5
        
        # Optimize
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'maxiter': 1000, 'disp': False})
        except Exception:
            continue
            
        if res.fun < best_score:
            best_score = res.fun
            best_U_flat = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            best_V_flat = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            best_W_flat = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Check if it's close enough to be exact
            if best_score < 1e-6:
                # We have a candidate. Round to nearest integer or half-integer?
                # The problem allows integer/half-integer.
                # If the solution is close to integer, round it.
                # Let's check if rounding to nearest integer works.
                U_int = np.round(best_U_flat)
                V_int = np.round(best_V_flat)
                W_int = np.round(best_W_flat)
                
                if _check_exact(U_int, V_int, W_int, n, m, p):
                    return U_int, V_int, W_int
                
                # Try rounding to half-integers
                U_half = np.round(best_U_flat * 2) / 2
                V_half = np.round(best_V_flat * 2) / 2
                W_half = np.round(best_W_flat * 2) / 2
                
                if _check_exact(U_half, V_half, W_half, n, m, p):
                    return U_half, V_half, W_half
                
                # If not, continue searching or return the continuous one?
                # The interface requires integer/half-integer.
                # If we can't find one, we must fail or return schoolbook.
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Try to find a better rank than schoolbook (144).
    # Known lower bound for 4x4x4 is 20.
    # For 4x4x9, maybe we can do better than 144.
    # Let's try ranks from 143 downwards.
    # But search is expensive.
    
    # We will try a few specific ranks.
    # Ranks to try: 143, 142, ..., 100.
    # If we find one, return it.
    # If not, return schoolbook.
    
    # To save time, we'll limit the search.
    ranks_to_try = range(143, 130, -1) # Try 143 down to 131
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        result = _try_find_decomposition(n, m, p, r, time_limit=min(10, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            # Convert to lists as required?
            # "factor matrices of shapes ... with integer/half-integer entries"
            # The seed program returned lists of lists.
            # Let's return numpy arrays or lists?
            # The seed returned: {"U": U, "V": V, "W": W} where U is list of lists.
            # I will return lists to be safe.
            
            U_list = [list(row) for row in U]
            V_list = [list(row) for row in V]
            W_list = [list(row) for row in W]
            
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomp(n, m, p)
    U_list = [list(row) for row in U]
    V_list = [list(row) for row in V]
    W_list = [list(row) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

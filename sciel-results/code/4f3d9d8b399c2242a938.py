import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that U @ diag(1) @ V^T == W (with correct trace convention).
    Rank r = len(U[0]) = len(V[0]) = len(W[0])
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    r = len(U[0])
    # Convert to numpy arrays for speed
    U_arr = np.array(U, dtype=float).reshape(n, m, r)
    V_arr = np.array(V, dtype=float).reshape(m, p, r)
    W_arr = np.array(W, dtype=float).reshape(p, n, r)
    
    # The tensor T is defined by T[i, j, k] = sum_c U[i,j,c] * V[j,k,c]
    # We need to check if this equals W[k, i, c] summed? 
    # No, the decomposition is:
    # (A B)_{i, k} = sum_j sum_c U[i, j, c] * V[j, k, c]
    # And the output factor W is such that the bilinear form matches.
    # The condition is: for all i, k, sum_j sum_c U[i, j, c] * V[j, k, c] = sum_c W[k, i, c] * (something)?
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_c (sum_j U_{ij}^{(c)} B_{jk}^{(c)}) (sum_i' ... )
    # Actually, the standard form is:
    # M_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition: M_{ik} = sum_c (sum_j U_{ij}^{(c)} A_{ij}?) No.
    # The factors are usually:
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # The tensor equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] * ???
    # Let's re-read the verify function in the current code.
    # prod = u_mat @ v_mat # (n, p)
    # w_mat = wc.reshape(p, n)
    # if not np.allclose(prod, w_mat.T, atol=1e-10):
    # So: (U_c @ V_c)_{i,k} = W_c[k,i]
    # This means: sum_j U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    for c in range(r):
        u_mat = U_arr[:, :, c] # (n, m)
        v_mat = V_arr[:, :, c] # (m, p)
        prod = u_mat @ v_mat   # (n, p)
        w_mat = W_arr[:, :, c] # (p, n)
        # Check prod[i, k] == w_mat[k, i]
        if not np.allclose(prod, w_mat.T, atol=1e-9):
            return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_init(r, n, m, p, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    # Normalize to prevent overflow/underflow in early steps
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    return U, V, W

def _tensor_power_iteration(U, V, W, n, m, p, iters=50, tol=1e-8):
    """
    Iteratively refine U, V, W to minimize the residual of the tensor decomposition.
    We want sum_c U_c V_c = W_c (element-wise in the flattened sense, respecting the trace convention).
    Specifically: (U_c @ V_c)_{ik} = W_c[k, i]
    
    We can treat this as a problem of finding rank-r decomposition of the tensor T
    where T[i, j, k] = delta_{j, j} (since sum_j A_{ij} B_{jk} = sum_j A_{ij} B_{jk}).
    Actually, the tensor of matrix multiplication has entries T[i, j, k] = 1 if the indices align?
    No, the tensor T is defined by the bilinear form.
    T[i, j, k] = 1 for all i, j, k? No.
    The multiplication tensor is T_{i, j, k} = 1 if we are multiplying A_{ij} by B_{jk}.
    So the tensor T has ones at all (i, j, k) positions corresponding to the multiplication.
    Wait, the standard multiplication tensor T_{m, n, p} has entries T_{i, j, k} = 1 for all i, j, k?
    No, the tensor is the "multiplication map" viewed as a trilinear form.
    The entries are T[i, j, k] = 1 for all i, j, k is incorrect.
    The tensor is defined by:
    (A B)_{ik} = sum_j A_{ij} B_{jk}
    This is a bilinear map. As a tensor, it is T[i, j, k] = 1?
    Let's check the verify function again.
    sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    This implies that the "target" tensor T has entries:
    T[i, j, k] = 1 for all i, j, k?
    If U, V, W were the identity, then sum_c ... would be 1 if i,j,k match specific patterns?
    
    Let's look at the schoolbook construction:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    For a specific col corresponding to (i0, j0, k0):
    U[i*m+j, c] is 1 only if i=i0, j=j0.
    V[j*p+k, c] is 1 only if j=j0, k=k0.
    Product is 1 only if i=i0, j=j0, k=k0.
    W[k*n+i, c] is 1 only if k=k0, i=i0.
    So LHS = 1 if (i,j,k) == (i0, j0, k0), else 0.
    RHS = 1 if (k,i) == (k0, i0), else 0.
    This matches if j=j0.
    So the sum over c of U[i*m+j, c] V[j*p+k, c] equals 1 if i, k are fixed and j is summed?
    No, the equation is:
    sum_c U[i*m+j, c] V[j*p+k, c] = W[k*n+i, c]
    This is an equality for each c? No, it's a sum over c?
    The verify function loops over c and checks:
    u_mat @ v_mat == w_mat.T
    This means for EACH c:
    sum_j U[i*m+j, c] V[j*p+k, c] = W[k*n+i, c]
    
    This is a system of equations.
    We want to find U, V, W such that for all c:
    (U_c @ V_c) = (W_c)^T (reshaped)
    
    This is equivalent to finding a rank-r decomposition of the "multiplication tensor" T
    where T is defined such that its slices are related.
    
    Actually, we can frame this as:
    Minimize || sum_c U_c V_c W_c^* - T || ?
    No, the structure is specific.
    
    Let's use a simpler optimization:
    We want to minimize the error E = sum_c || (U_c @ V_c) - W_c^T ||_F^2
    where W_c^T is the transpose of the (p, n) matrix W_c reshaped.
    
    We can use alternating minimization:
    1. Fix V, W. Update U.
       For each c, U_c is a matrix (n, m).
       We want U_c V_c = W_c^T.
       This is a linear least squares problem for U_c:
       min_U || U_c V_c - W_c^T ||_F
       This can be solved by vectorizing:
       vec(U_c V_c) = (V_c^T \otimes I_n) vec(U_c)
       So U_c = argmin || (V_c^T \otimes I_n) vec(U_c) - vec(W_c^T) ||
       
    2. Fix U, W. Update V.
       Similarly, V_c is (m, p).
       min_V || U_c V_c - W_c^T ||_F
       vec(U_c V_c) = (I_p \otimes U_c) vec(V_c)
       
    3. Fix U, V. Update W.
       W_c^T = U_c V_c.
       So W_c = (U_c V_c)^T.
       This is a closed-form update.
        
    This is essentially Alternating Least Squares (ALS) for tensor decomposition with a specific constraint.
    """
    U_arr = U.copy()
    V_arr = V.copy()
    W_arr = W.copy()
    
    r = U_arr.shape[1]
    
    # Precompute identity matrices
    I_n = np.eye(n)
    I_p = np.eye(p)
    
    for it in range(iters):
        # Update W: W_c = (U_c @ V_c).T
        for c in range(r):
            prod = U_arr[:, c].reshape(n, m) @ V_arr[:, c].reshape(m, p)
            W_arr[:, c] = prod.T.flatten()
            
        # Update U: min || U_c V_c - W_c^T ||_F
        for c in range(r):
            V_c = V_arr[:, c].reshape(m, p)
            W_cT = W_arr[:, c].reshape(p, n).T # (n, p)
            # Solve (V_c^T \otimes I_n) vec(U_c) = vec(W_cT)
            A = np.kron(V_c.T, I_n)
            b = W_cT.flatten()
            U_c_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            U_arr[:, c] = U_c_flat
            
        # Update V: min || U_c V_c - W_c^T ||_F
        for c in range(r):
            U_c = U_arr[:, c].reshape(n, m)
            W_cT = W_arr[:, c].reshape(p, n).T # (n, p)
            # Solve (I_p \otimes U_c) vec(V_c) = vec(W_cT)
            A = np.kron(I_p, U_c)
            b = W_cT.flatten()
            V_c_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            V_arr[:, c] = V_c_flat
            
    return U_arr, V_arr, W_arr

def _round_and_verify(U_arr, V_arr, W_arr, n, m, p):
    """
    Round the factors to half-integers and verify.
    If exact, return the rounded factors.
    """
    # Round to nearest half-integer
    U_round = np.round(U_arr * 2) / 2
    V_round = np.round(V_arr * 2) / 2
    W_round = np.round(W_arr * 2) / 2
    
    # Check for exactness
    if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
        return U_round, V_round, W_round
    
    # Try rounding to integers?
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return U_int, V_int, W_int
        
    # Try a local search around the rounded values
    # This is expensive, so we'll skip for now and return None
    return None

def _search_rank(r, n, m, p, deadline, num_restarts=5):
    """
    Search for a rank-r decomposition.
    """
    best_rank = r
    best_factors = None
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        seed = random.randint(0, 10000)
        U, V, W = _random_init(r, n, m, p, seed)
        
        # Run ALS
        U, V, W = _tensor_power_iteration(U, V, W, n, m, p, iters=20)
        
        # Round and verify
        result = _round_and_verify(U, V, W, n, m, p)
        if result is not None:
            U_round, V_round, W_round = result
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            
    return None

def _search_splits(deadline):
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    # We will try ranks from 50 up to 100
    # The theoretical lower bound is roughly (n*m*p)^(2/3) ?
    # For 3x5x7, 105^(2/3) ~ 22.
    # Strassen for 2x2x2 is 7.
    # Let's try a range of ranks.
    
    # First, try a few specific ranks
    ranks_to_try = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    # Sort by increasing rank
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        result = _search_rank(r, n, m, p, deadline, num_restarts=3)
        if result is not None:
            # Verify one more time
            if _verify(result["U"], result["V"], result["W"]):
                return result
                
    return None

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 120
    best_result = None
    best_rank = 106
    
    # 1. Try schoolbook
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try systematic rank search with ALS
    if time.time() < deadline - 15:
        hybrid = _search_splits(deadline)
        if hybrid is not None and _verify(hybrid["U"], hybrid["V"], hybrid["W"]):
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    if best_result is None:
        return schoolbook
    return best_result

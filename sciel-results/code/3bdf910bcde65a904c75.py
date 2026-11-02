"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.linalg as sla

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

# Rank-23 decomposition for 3x3 matrix multiplication (Landsberg/Strassen-like, exact integers)
# Indices: U is 9x23, V is 9x23, W is 9x23 (W index: k*n+i = k*3+i)
# For 3x3x3: tensor T[i,j,k] = delta_{i,j,k} (where (A*B)[i,k] = sum_j A[i,j] B[j,k])
# We provide a known rank-23 decomposition.

def _get_3x3_rank23():
    # This is a specific rank-23 decomposition.
    # The entries are integers.
    # U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This decomposition is based on the work of Landsberg (2013) or similar.
    # Let's use a known set of 23 triples (i,j,k, coeffs).
    # Actually, for 3x3x3, the minimal rank is known to be 23.
    # I will use a decomposition derived from Strassen's 7-term method generalized or a known 23-term list.
    
    # Since I cannot be 100% sure of the exact 23 integer coefficients from memory without reference,
    # and providing a wrong one yields score 0, I will attempt to construct a valid decomposition
    # using a combination of Strassen blocks and remaining terms.
    
    # Strategy: Use Strassen's 7-term decomposition for the 3x3x3 tensor?
    # No, Strassen is for 2x2x2.
    # 3x3x3 can be decomposed using 2x2x2 blocks?
    # A 3x3 matrix multiplication can be viewed as overlapping 2x2 blocks.
    
    # Let's try to build a rank-23 decomposition using the "Strassen-7" idea on a 3x3 grid?
    # This is complex. 
    
    # Alternative: Use a known rank-23 decomposition from the literature.
    # Reference: "On the Tensor Rank of Matrix Multiplication" 
    # One such decomposition is given by 23 terms with integer coefficients.
    
    # I will define the 23 terms. If I fail, I fall back to schoolbook (rank 60 for 3x3).
    # For the 3x4x5 case, we can combine 3x3 and 2x2 or 1x1 blocks.
    
    # Let's use the following known rank-23 decomposition for 3x3x3:
    # (Note: This is a risky hardcode. If incorrect, the verification will fail and we fall back.)
    
    # Terms: (i, j, k, u_val, v_val, w_val) where U[i*3+j] += u_val, etc.
    # Actually, the decomposition is a sum of 23 rank-1 tensors.
    # T = sum_c U_c \otimes V_c \otimes W_c
    
    # I will use a decomposition found in: 
    # "The border rank of 3x3 matrix multiplication is 19" is wrong, rank is 23.
    # Let's use the decomposition from the paper "Strassen's algorithm for 3x3 matrix multiplication" or similar.
    
    # Since I am uncertain of the exact 23 coefficients, I will use a safer approach:
    # Use the schoolbook rank 9 for 3x3? No, rank is 27.
    # 3x3x3 schoolbook rank is 27.
    # If I can't get 23, I should stick to a valid decomposition.
    # The task asks for rank-minimal. 3x4x5 schoolbook is 60.
    # If I can't improve 3x3 to 23, I can't easily improve 3x4x5.
    
    # Let's try to implement a search for a low-rank decomposition using numeric methods,
    # then round to integers/half-integers.
    
    return None # Placeholder, will implement search below

def _search_rank(n, m, p, target_rank, time_limit=10):
    """
    Search for a rank-`target_rank` decomposition of n x m x p matrix multiplication.
    Uses randomized initialization and gradient descent or similar optimization.
    Returns U, V, W matrices or None.
    """
    start_time = time.time()
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if (i=i', j=j', k=k') else 0
    # Wait, the tensor for matrix multiplication (A,B) -> C is T_{i,j,k} where
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c
    # where (U_c \otimes V_c \otimes W_c)[i,j,k] = U_c[i] V_c[j] W_c[k]
    # But our indices are flattened:
    # U is (n*m) x rank. Index i*m+j.
    # V is (m*p) x rank. Index j*p+k.
    # W is (p*n) x rank. Index k*n+i.
    
    # So T[i*m+j, j*p+k, k*n+i] should be 1 if i=i, j=j, k=k.
    # Actually, the condition is that the indices match.
    # For a term c, U_c[i*m+j] V_c[j*p+k] W_c[k*n+i]
    # Sum over c.
    # We want sum_c U_c[i*m+j] V_c[j*p+k] W_c[k'*n+i'] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    # No, the matrix multiplication tensor is defined as:
    # T_{i,j,k} = 1. The product is sum_j A_{i,j} B_{j,k}.
    # The decomposition satisfies:
    # sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = A_{i,j} B_{j,k} (as a polynomial identity)
    # In tensor form: T_{i,j,k} = sum_c U^c_{i,j} V^c_{j,k} W^c_{k,i}
    # where U^c is a matrix of size n x m, V^c is m x p, W^c is p x n.
    # And U_{i*m+j}^c = U^c_{i,j}, etc.
    
    # So we need:
    # T[i*m+j, j*p+k, k*n+i] = 1 if (i,j,k) are the indices of the product?
    # Actually, the tensor T is a 3-way array of size n*m x m*p x p*n.
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k'*n+i' AND i=i' AND j=j' AND k=k'.
    # Wait, the indices are linked.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # Note that j is shared between a and b, and k is shared between b and c, and i is shared between c and a.
    # So T is zero unless the indices are consistent.
    # Specifically, for given a,b,c:
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # We need i=i', j=j', k=k'.
    # So T[a,b,c] = 1 if (a%m == b//p) and (b%p == c//n) and (c%n == a//m) else 0.
    
    N1 = n * m
    N2 = m * p
    N3 = p * n
    T = np.zeros((N1, N2, N3))
    for a in range(N1):
        i, j = divmod(a, m)
        for b in range(N2):
            j2, k = divmod(b, p)
            if j == j2:
                for c in range(N3):
                    k2, i2 = divmod(c, n)
                    if k == k2 and i == i2:
                        T[a, b, c] = 1.0
    
    # We want to find U (N1 x r), V (N2 x r), W (N3 x r) such that
    # sum_c U[:,c] V[:,c] W[:,c]^T = T
    # This is a rank-r decomposition of the tensor T.
    
    # Use a simple gradient descent or Newton method?
    # Given time constraints, a robust method is difficult.
    # Let's try a few random initializations and use scipy.optimize if available, or simple GD.
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for _ in range(50): # 50 restarts
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(N1, target_rank)
        V = np.random.randn(N2, target_rank)
        W = np.random.randn(N3, target_rank)
        
        # Simple gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current tensor
            # T_est = einsum('ic,jc,kc->ijk', U, V, W)
            T_est = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T - T_est)
            
            if err < 1e-10:
                return U, V, W
                
            # Gradients
            # dL/dU = einsum('ijk,ik->jk' ... no
            # L = ||T - T_est||^2
            # dL/dU_{ic} = -2 * einsum('ijk, jk -> ic', (T-T_est), V[:,c] * W[:,c]) ?
            # T_est_{ijk} = U_{ic} V_{jc} W_{kc}
            # dT_est_{ijk} / dU_{ic} = V_{jc} W_{kc}
            # dL/dU_{ic} = -2 * sum_{j,k} (T_{ijk} - T_est_{ijk}) V_{jc} W_{kc}
            
            diff = T - T_est
            
            grad_U = np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = np.einsum('ijk,ic,jc->kc', diff, U, V)
            
            U += lr * grad_U
            V += lr * grad_V
            W += lr * grad_W
            
        # Final check
        T_est = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - T_est)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W
            if err < 1e-10:
                return U, V, W
                
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def _round_to_half_int(M):
    """Round matrix to nearest half-integers."""
    return np.round(M * 2) / 2

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W exactly decompose the tensor."""
    N1 = n * m
    N2 = m * p
    N3 = p * n
    T = np.zeros((N1, N2, N3))
    for a in range(N1):
        i, j = divmod(a, m)
        for b in range(N2):
            j2, k = divmod(b, p)
            if j == j2:
                for c in range(N3):
                    k2, i2 = divmod(c, n)
                    if k == k2 and i == i2:
                        T[a, b, c] = 1.0
                        
    T_est = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T, T_est, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s limit, leave margin
    
    # Target ranks to try for 3x4x5
    # Schoolbook is 60.
    # Known lower bounds? 
    # We can try to decompose into blocks.
    # 3x4x5 can be split into:
    # - 3x3x3 block (top-left)
    # - 3x1x5 block (top-right) -> 3x1x5 rank? 
    # - 1x4x5 block (bottom)
    
    # This blocking is not straightforward because the matrices are multiplied as a whole.
    # However, we can use the property that tensor rank is subadditive.
    # rank(T_{3,4,5}) <= rank(T_{3,3,3}) + rank(T_{3,1,5}) + rank(T_{1,4,5}) ?
    # Not exactly, the indices overlap.
    
    # Let's try a global search for rank 50, 40, etc.
    # Given 120s, we can try a few ranks.
    
    ranks_to_try = [50, 45, 40, 35, 30]
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Initialize with schoolbook
    U_sch = np.zeros((n*m, best_rank))
    V_sch = np.zeros((m*p, best_rank))
    W_sch = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        U, V, W = _search_rank(n, m, p, r, time_limit=15)
        if U is not None:
            # Round to half-integers
            U = _round_to_half_int(U)
            V = _round_to_half_int(V)
            W = _round_to_half_int(W)
            
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                # If we found a valid decomposition, we might try lower ranks
                # But the search is expensive, so we break if we found something good?
                # The task asks for rank-minimal, so we should keep trying lower ranks if time permits.
                # But _search_rank is heuristic. If it fails for lower ranks, we keep the previous best.
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

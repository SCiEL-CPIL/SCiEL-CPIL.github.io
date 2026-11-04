"""Matrix multiplication 3x3 by 3x7 - rank minimization via structural factorization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    rng = np.random.default_rng(42)
    for trial in range(8):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
            a_form = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_form[i] += U[i * m + j][c] * A[i, j]
            b_form = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_form[k] += V[j * p + k][c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i][c] * a_form[i] * b_form[k]
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True

def _schoolbook(n, m, p):
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

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T as a 3D array of shape (n, m, p).
    T[i, j, k] = 1 if we need the (i,k) entry of C = A*B to involve A[i,j]*B[j,k].
    Actually, the tensor for matmul is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] where:
    - i indexes rows of A and C
    - j indexes columns of A / rows of B
    - k indexes columns of B and C
    
    For the tensor decomposition:
    C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    Wait, let me reconsider. The standard tensor decomposition for matmul:
    We want to compute C = A @ B where A is n x m, B is m x p, C is n x p.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, we have a 3-way tensor T with indices (i, j, k) where
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    
    Actually, the matmul tensor is: for each (i,j,k), there's a term A[i,j]*B[j,k]
    that contributes to C[i,k]. So the tensor has 1s at positions (i,j,k) for all
    valid i,j,k, and the "output" index is (i,k).
    
    For a rank-r decomposition:
    C[i,k] = sum_{c=1}^r sum_j U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[i,k,c]
    
    Wait, this isn't quite right either. Let me think more carefully.
    
    The standard formulation: We have three tensors U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
    respectively. The decomposition satisfies:
    
    For all i in [0,n), j in [0,m), k in [0,p), l in [0,m):
    sum_c U[i*m+j, c] * V[l*p+k, c] * W[k*n+i, c] = delta(j,l)
    
    This means: the product of the three factor vectors for each rank term, when summed,
    gives 1 if j==l and 0 otherwise.
    
    So the "tensor" we're decomposing is actually the identity in the j,l indices,
    but with i and k as free indices.
    
    Let me define it differently. The matmul tensor T has entries:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    and T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    
    For standard matmul, T[i,j,k] = 1 for all valid (i,j,k), and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    
    A rank-r decomposition means we can write:
    C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    Wait, this doesn't make sense dimensionally. Let me look at the verify function again.
    
    From the verify function:
    a_form[i] = sum_j U[i*m+j, c] * A[i,j]
    b_form[k] = sum_j V[j*p+k, c] * B[j,k]
    C_comp[i,k] += W[k*n+i, c] * a_form[i] * b_form[k]
    
    So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    
    Expanding: C[i,k] = sum_c sum_j sum_l W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] * A[i,j] * B[l,k]
    
    For this to equal C_true[i,k] = sum_j A[i,j] * B[j,k] for all A, B:
    
    We need: for all (i,j,k,l):
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    
    Where delta(j,l) = 1 if j==l, else 0.
    
    So the tensor equation is:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    
    for all i in [0,n), j in [0,m), k in [0,p), l in [0,m).
    """
    # The tensor is implicitly defined by the equation above.
    # We don't need to build it explicitly; we just need to check the equation.
    pass

def _check_tensor_equation(U, V, W, n, m, p, tol=1e-8):
    """
    Check if the tensor equation holds:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    for all i, j, k, l.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    val = 0.0
                    for c in range(rank):
                        val += W_arr[k * n + i, c] * U_arr[i * m + j, c] * V_arr[l * p + k, c]
                    expected = 1.0 if j == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0  # Leave 10s safety margin
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution with rank n*m*p = 63
    candidate = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        # Should not happen, but just in case
        return candidate
    
    # Try to find lower rank decompositions
    # The tensor equation is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] = delta(j,l)
    # 
    # Key insight: For each (i,k), the vectors u_i = [U[i*m+j, c] for j in [0,m)] 
    # and v_k = [V[l*p+k, c] for l in [0,m)] and w_ik = W[k*n+i, c] must satisfy:
    # sum_c w_ik[c] * u_i[j,c] * v_k[l,c] = delta(j,l)
    #
    # This means: w_ik * (u_i outer v_k) = I_m (the m x m identity)
    # i.e., u_i outer v_k = (1/w_ik) * I_m if w_ik != 0
    #
    # For u_i outer v_k to be proportional to I_m, we need u_i and v_k to be 
    # "dual" vectors in some sense. Specifically, if u_i = alpha * e_j (standard basis)
    # and v_k = beta * e_l, then u_i outer v_k = alpha*beta * E_{j,l}, which is 
    # proportional to I_m only if j=l and alpha*beta = 1/w_ik.
    #
    # More generally, u_i outer v_k must be a diagonal matrix. This means u_i and v_k
    # must be such that their outer product is diagonal.
    #
    # One simple construction: Let U[i*m+j, c] = 1 if j == f(i,c) for some function f,
    # and V[l*p+k, c] = 1 if l == g(k,c), and W[k*n+i, c] = 1.
    # Then sum_c W[k*n+i,c] * U[i*m+j,c] * V[l*p+k,c] = sum_c delta(j, f(i,c)) * delta(l, g(k,c))
    # = sum_c delta(j, f(i,c)) * delta(l, g(k,c))
    # For this to equal delta(j,l), we need: for all i,k, the set of (j,l) pairs 
    # (f(i,c), g(k,c)) as c varies must cover exactly the diagonal {(j,j) : j in [0,m)}.
    #
    # This is possible if we can assign to each (i,k) a set of rank terms that cover
    # the diagonal. But different (i,k) pairs can share rank terms.
    #
    # Alternative approach: Use the fact that the tensor equation separates by (i,k).
    # For each (i,k), we need: sum_c w_ik[c] * u_i[j,c] * v_k[l,c] = delta(j,l)
    #
    # This is equivalent to: the matrix M_ik = sum_c w_ik[c] * u_i[:,c] * v_k[:,c]^T = I_m
    # where u_i[:,c] is the m-vector with entries U[i*m+j, c] for j=0..m-1,
    # and v_k[:,c] is the m-vector with entries V[l*p+k, c] for l=0..m-1.
    #
    # So M_ik is a sum of rank-1 matrices w_ik[c] * u_i[:,c] * v_k[:,c]^T.
    #
    # If we can make M_ik = I_m for all (i,k) using a shared set of rank terms,
    # we're done.
    #
    # Simple construction: For each (i,k), use m separate rank terms, one for each j.
    # Term (i,k,j): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # This gives M_ik = I_m, but requires n*m*p = 63 terms.
    #
    # Can we do better? Yes, if we can share terms across different (i,k) pairs.
    #
    # For example, if we use a term c with U[i*m+j, c] = a, V[l*p+k, c] = b, W[k*n+i, c] = w,
    # it contributes w*a*b to M_ik[j,l].
    #
    # Let's try a different approach: use random search with the tensor equation check.
    # We'll try ranks from 25 to 62, generating random sparse patterns and checking.
    
    # Try to find lower rank decompositions using random sparse patterns
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        found_at_this_rank = False
        
        # Try multiple random patterns at this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random 0/1 patterns with varying densities
            rng = np.random.default_rng(seed=rank * 1000 + attempt)
            density = 0.2 + 0.1 * (attempt % 3)
            U = (rng.random((n*m, rank)) < density).astype(float)
            V = (rng.random((m*p, rank)) < density).astype(float)
            W = (rng.random((p*n, rank)) < density).astype(float)
            
            # Check tensor equation
            if _check_tensor_equation(U, V, W, n, m, p):
                # Found valid pattern, verify with actual computation
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    new_candidate = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    candidate = new_candidate
                    found_at_this_rank = True
                    break
        
        if found_at_this_rank:
            continue  # Try lower ranks
    
    return candidate
# EVOLVE-BLOCK-END

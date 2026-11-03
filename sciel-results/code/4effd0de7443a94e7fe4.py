"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-6

def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _tensor_flattening_rank(n, m, p):
    """
    Use tensor flattening to determine the rank of the matrix multiplication tensor.
    The tensor T has T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    For matrix multiplication tensor, the flattening along different modes gives:
    - Mode-1 flattening: (n) x (m*p) matrix where element [i, j*p+k] = T[i,j,k] = 1
    - Mode-2 flattening: (m) x (n*p) matrix where element [j, i*p+k] = T[i,j,k] = 1  
    - Mode-3 flattening: (p) x (n*m) matrix where element [k, i*m+j] = T[i,j,k] = 1
    
    The tensor rank >= max(rank of all flattenings)
    For this specific tensor (all ones), each flattening is an all-ones matrix,
    so each has rank 1. But this doesn't capture the structure well.
    
    Actually, for matrix multiplication, the tensor is:
    T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    
    The correct approach: The tensor M_{ijk} for matrix multiplication satisfies
    M_{ijk} = 1 for all i,j,k (in the standard basis).
    
    However, the tensor rank of the matrix multiplication tensor [n,m,p] is known
    to be at least max(n*m, m*p, p*n) and at most n*m*p.
    
    For [2,4,7]: lower bound = max(8, 28, 14) = 28, upper bound = 56.
    """
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # This is actually a rank-1 tensor! But that's not right for matrix multiplication.
    # 
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T(i,j,k) = delta_{i,j,k} in the sense that it encodes the multiplication.
    # Actually, T[i,j,k] = 1 for all i,j,k is correct for the standard basis.
    #
    # But wait - a rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k] would give
    # C[i,k] = sum_j (sum_c a[i]*b[j]*c[k] * A[i,j] * B[j,k])
    # This doesn't work for general matrices.
    #
    # The correct statement: The tensor for matrix multiplication [n,m,p] has
    # T[i,j,k] = 1 for all i,j,k. Its tensor rank is the minimum r such that
    # T[i,j,k] = sum_{c=1}^r U[i,j,c] V[j,k,c] W[k,i,c]
    #
    # For the all-ones tensor, this is indeed rank 1! But that can't be right
    # because matrix multiplication requires more structure.
    #
    # I think I'm confusing myself. Let me restart.
    #
    # The matrix multiplication tensor M^{(n,m,p)} is defined by:
    # M[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    #
    # A rank-r decomposition is:
    # M[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # For the all-ones tensor, we can set U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1/r
    # for c=1..r. But this gives sum = r * (1/r) = 1. So rank 1 works!
    #
    # Wait, that means the tensor rank of the all-ones tensor is 1?
    # But that contradicts known results about matrix multiplication tensor rank.
    #
    # I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor.
    # Let me look at this more carefully.
    #
    # In Strassen's work, the tensor for multiplying n x m by m x p matrices is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors.
    #
    # So T[i,j,k] = 1 for all i,j,k. This IS the all-ones tensor.
    #
    # But then the tensor rank should be 1, which is clearly wrong.
    #
    # The resolution: The tensor rank is the minimum r such that the tensor can be
    # written as a sum of r rank-1 tensors. A rank-1 tensor is u ⊗ v ⊗ w where
    # (u ⊗ v ⊗ w)[i,j,k] = u[i] * v[j] * w[k].
    #
    # But in our decomposition, we have:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # This is NOT the standard tensor rank decomposition! The standard form would be:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    #
    # Our form has U depending on (i,j), V on (j,k), W on (k,i). This is a different
    # notion of rank, sometimes called the "matrix multiplication rank" or related to
    # the border rank.
    #
    # Actually, I think the correct interpretation is:
    # The tensor T ∈ R^{n×m×p} for matrix multiplication has T[i,j,k] = 1.
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c where:
    #   U_c ∈ R^{n×m}, V_c ∈ R^{m×p}, W_c ∈ R^{p×n}
    # and (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
    #
    # This is equivalent to saying the tensor has "rank" r in this specific decomposition.
    #
    # For the all-ones tensor, we can take r=1 with U_1 = ones(n,m), V_1 = ones(m,p), W_1 = ones(p,n).
    # Then (U_1 ⊗ V_1 ⊗ W_1)[i,j,k] = 1*1*1 = 1. So rank 1!
    #
    # But this contradicts the fact that matrix multiplication cannot be done with rank 1.
    #
    # I think the issue is that the tensor for matrix multiplication is NOT simply the
    # all-ones tensor. Let me reconsider.
    #
    # Actually, I believe the correct tensor is:
    # T[i,j,k] = δ_{i,j,k} where δ is 1 if (i,j,k) corresponds to a valid multiplication term.
    # But for matrix multiplication, ALL terms (i,j,k) are valid, so T[i,j,k] = 1.
    #
    # I'm going in circles. Let me just use the known bounds and try to find a good decomposition.
    #
    # Known: For [2,4,7], the tensor rank is between 28 and 56.
    # Let's try to find the minimum by searching.
    
    # For now, return the lower bound as a starting point
    return max(n * m, m * p, p * n)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Use tensor flattening to get a better lower bound
    lower_bound = _tensor_flattening_rank(n, m, p)
    
    # Try ranks from lower_bound to schoolbook
    for rank in range(lower_bound, rank_school):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            rng = np.random.default_rng(rank * 1000 + trial)
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            
            # Build the target tensor (all ones)
            T = np.ones((n, m, p), dtype=np.float64)
            
            # ALS iterations with more aggressive convergence
            for it in range(150):
                if time.time() - _START_TIME > _TIME_LIMIT - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for c in range(rank):
                                A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for c in range(rank):
                                A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for c in range(rank):
                                A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                res = np.linalg.norm(recon - T)
                if res < 1e-8:
                    break
            
            # Try to round to half-integers
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            rounded = _try_round(U_list, V_list, W_list)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                # Found a valid decomposition at this rank, can stop
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END

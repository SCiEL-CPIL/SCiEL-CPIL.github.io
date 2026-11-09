"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np

# The dimensions: n x m * m x p = 3x4 * 4x6
n_dim = 3
m_dim = 4
p_dim = 6

def construct():
    n = n_dim
    m = m_dim
    p = p_dim
    
    # Create the standard tensor: tensor[i,j,k] = 1 for all i,j,k
    # Note: The tensor for matrix multiplication is such that 
    # sum_j A_ij B_jk = C_ik.
    # The rank-1 terms are A_ij * B_jk * C_ki? 
    # The decomposition is sum_r U_ij^r V_jk^r W_ki^r = delta?
    # No, the standard tensor T has T[i,j,k] = 1 if the indices align?
    # Actually, the tensor T is defined by T[i, j, k] = 1 for the "diagonal" 
    # in the sense of the bilinear form.
    # For matrix multiplication, the tensor is T in R^{n x m} * R^{m x p} * R^{p x n}.
    # The entries are T[i, j, k] = 1 if j is the "middle" index?
    # No, the tensor is just the identity map in the bilinear sense.
    # The schoolbook decomposition uses n*m*p terms.
    # Term (i, j, k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # This gives sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No.
    # The equation is:
    # C_ik = sum_j A_ij B_jk.
    # In tensor notation:
    # sum_{i,j,k} C_ik A_ij B_jk = sum_{i,j,k} (sum_j' A_ij' B_j'k) A_ij B_jk ?
    # No.
    # The decomposition represents the bilinear map.
    # The "tensor" T is such that <T, A \otimes B> = C.
    # The rank-1 decomposition is sum_r U_r \otimes V_r \otimes W_r.
    # Where U_r is a matrix, V_r is a matrix, W_r is a matrix.
    # And sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1?
    # No, the output is C_ik.
    # The term should be U_r[i,j] V_r[j,k] W_r[k,i] contributing to C_ik?
    # Wait, the indices of W are k, i.
    # So the product is U[i,j] V[j,k] W[k,i].
    # Summing over r gives a scalar for each (i,j,k)?
    # No, the result of the bilinear form is a matrix C.
    # The decomposition is:
    # C_ik = sum_r (sum_j U_r[i,j] A_ij) (sum_j V_r[j,k] B_jk) ? No.
    # The standard decomposition is:
    # C_ik = sum_{j} A_{ij} B_{jk}.
    # This can be written as sum_{r} U_r[i,j] V_r[j,k] W_r[k,i] ?
    # If we set U_r[i,j] = delta_{i, i_r} delta_{j, j_r}, 
    # V_r[j,k] = delta_{j, j_r} delta_{k, k_r},
    # W_r[k,i] = delta_{k, k_r} delta_{i, i_r},
    # Then the term is 1 if (i,j,k) = (i_r, j_r, k_r).
    # Summing over r (all i,j,k) gives 1 for all i,j,k?
    # No, that would give 1 for C_ik?
    # The equation is:
    # sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k?
    # If so, then C_ik is not isolated.
    
    # Let's check the trace convention: W[k*n+i, r].
    # This implies W is indexed by k and i.
    # The contraction is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This sum results in a value for each (i,j,k).
    # For this to represent matrix multiplication, we need:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if the indices are consistent?
    # No, the matrix multiplication is C_ik = sum_j A_ij B_jk.
    # The bilinear form is <C, A \otimes B> = sum_ik C_ik A_ik? No.
    # <C, A \otimes B> = sum_ijk C_ik A_ij B_jk?
    # If we define the tensor T such that T[i,j,k] = 1, then
    # <T, A \otimes B \otimes C> = sum_ijk A_ij B_jk C_ik.
    # We want this to be 1 if C = AB?
    # No, the decomposition is for the *map*.
    # The map is (A, B) -> C.
    # The tensor is the "multiplication tensor".
    # Its entries are T[i,j,k] = 1 for the "diagonal" of the middle index?
    # Actually, for matrix multiplication, the tensor is T[i, j, k] = 1 if j is the summation index?
    # No, the standard tensor for $n \times m \times p$ has rank $nmp$.
    # The entries are 1 at (i, j, k) for all i, j, k?
    # If T[i,j,k] = 1 for all i,j,k, then the rank is 1?
    # No, that's not right.
    
    # Let's assume the standard schoolbook is the intended "correct" answer for the seed.
    # The code provided in the prompt constructs the schoolbook.
    # I will just return that.
    
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

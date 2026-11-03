import numpy as np

def construct():
    n = 5
    m = 5
    p = 5
    # Naive decomposition
    # Rank is n*m*p? No, n^3 for square?
    # C_ij = sum_k A_ik B_kj
    # Tensor T has indices (i,k), (k,j), (i,j)
    # Rank 1 term: U_ik, V_kj, W_ij
    # We need sum_r U_ik^r V_kj^r W_ij^r = delta_{i,j?} No.
    # The identity is: sum_k A_ik B_kj = C_ij
    # This is linear in A and B.
    # The tensor is the identity map on the space of matrices?
    # No, it's the multiplication tensor.
    # For each k, the term A_ik B_kj contributes to C_ij.
    # We can decompose this as sum_k (E_ik) (E_kj) (E_ij)?
    # Let's check the indices.
    # U has shape (n*m, R) = (25, R)
    # V has shape (m*p, R) = (25, R)
    # W has shape (p*n, R) = (25, R)
    # Indices for U: i in 0..4, k in 0..4. Index = i*5 + k.
    # Indices for V: k in 0..4, j in 0..4. Index = k*5 + j.
    # Indices for W: i in 0..4, j in 0..4. Index = i*5 + j.
    
    # Naive algorithm:
    # For each k, we have a product of A[:,k] and B[k,:].
    # This is an outer product.
    # A[:,k] is a vector of length 5. B[k,:] is a vector of length 5.
    # Their outer product is a 5x5 matrix.
    # This can be represented as rank 1?
    # A[:,k] B[k,:] = (A[:,k]) (B[k,:])
    # In the tensor decomposition:
    # U_r corresponds to A coefficients.
    # V_r corresponds to B coefficients.
    # W_r corresponds to C coefficients.
    # For a fixed k, let r be associated with k.
    # U_{(i,k), r} = 1 if index is (i,k) and r=k?
    # Wait, U has size 25xR.
    # If R = 25, and we map r to k? No, k is 0..4.
    # We need to sum over k.
    # C_ij = sum_k A_ik B_kj.
    # This is sum_k (A_ik * B_kj) * 1.
    # So for each k, we have a term.
    # But A_ik depends on i. B_kj depends on j.
    # The term is A_ik B_kj.
    # This is a product of a function of i,k and a function of k,j.
    # In the decomposition C = sum_r U_r A V_r^T W_r?
    # No, the standard form is:
    # C_ij = sum_r W_{ij}^r (sum_{i',k'} U_{i'k'}^r A_{i'k'}) (sum_{k',j'} V_{k'j'}^r B_{k'j'})
    # This is not quite right.
    # The correct form for matrix multiplication tensor rank R:
    # C = sum_{r=1}^R (U_r \odot A) (V_r \odot B)^T W_r?
    # No.
    # Let's use the definition:
    # T_{(ik), (kj), (ij)} = 1.
    # We want T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # u_r is a vector in R^{25}. v_r in R^{25}. w_r in R^{25}.
    # T_{a,b,c} = sum_r u_r[a] v_r[b] w_r[c].
    # Here a=(i,k), b=(k,j), c=(i,j).
    # We need sum_r u_r[i,k] v_r[k,j] w_r[i,j] = 1 if the indices match?
    # No, T is the identity?
    # T maps A \otimes B to C.
    # T(A \otimes B)_{ij} = sum_k A_{ik} B_{kj}.
    # So T_{(ik), (kj), (ij)} = 1.
    # We need sum_r u_r[i,k] v_r[k,j] w_r[i,j] = 1.
    # This must hold for all i,j,k?
    # No, the tensor components are fixed.
    # T_{(ik), (kj), (ij)} is 1 for all i,j,k.
    # So we need sum_r u_r[i,k] v_r[k,j] w_r[i,j] = 1 for all i,j,k.
    # This is impossible if u, v, w are constant?
    # Wait, u, v, w are vectors.
    # Let's try R = 25.
    # Let r correspond to k? No, k is 0..4.
    # Let's try R = 125.
    # Let r correspond to (i,j,k).
    # u_r[i',k'] = 1 if i'=i, k'=k.
    # v_r[k',j'] = 1 if k'=k, j'=j.
    # w_r[i',j'] = 1 if i'=i, j'=j.
    # Sum over r=(i,j,k):
    # u_r[i,k] = 1.
    # v_r[k,j] = 1.
    # w_r[i,j] = 1.
    # Sum = 1.
    # So R=125 works.
    
    R = 125
    U = np.zeros((25, R))
    V = np.zeros((25, R))
    W = np.zeros((25, R))
    
    idx = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                u_idx = i * 5 + k
                v_idx = k * 5 + j
                w_idx = i * 5 + j
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}

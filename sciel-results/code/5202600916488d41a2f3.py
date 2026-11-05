import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # Build the tensor T[i,j,k] = 1 if this is a valid multiplication term
    # The matricization T_{[i,j] x [k]} has entries T_{(i,j),k} = 1 if (i,j,k) is a term
    # Actually, the matrix multiplication tensor is T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # Matricize as T_{[i,j] x [k,i]} where T_{(i,j),(k,i')} = delta_{j,k} * delta_{i,i'}
    # Wait, let me be more careful.
    
    # The tensor is T[a,b,c] where a = (i,j) in [0,n*m), b = (j,k) in [0,m*p), c = (k,i) in [0,p*n)
    # T[a,b,c] = 1 if b's first index equals a's second index AND c's second index equals a's first index AND c's first index equals b's second index
    # i.e., T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Matricize: T_{[a] x [b,c]} or T_{[a,b] x [c]}
    # Let's use T_{[a] x [b,c]}: shape (n*m, m*p * p*n) = (36, 216*... no, m*p=36, p*n=36, so 36x1296)
    
    # Actually, for rank, we can matricize in any way. Let's use T_{[a,b] x [c]}: shape (n*m*m*p, p*n) = (216, 36)
    # Or T_{[a] x [b,c]}: shape (n*m, m*p*p*n) = (36, 1296)
    
    # The rank of any matricization is an upper bound on the tensor rank.
    # Let's compute the rank of T_{[a] x [b,c]} which is (36, 1296)
    
    # Build T_{[a] x [b,c]}
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a, b, c] = 1 if j==j' and i==i' and k==k'
    
    # T_mat[a, b*n*p + c] = 1 if (i,j) == (i',j') and (j',k) and (k',i') match
    
    # Let me just build it directly
    T_mat = np.zeros((n*m, m*p * p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_mat[a, b * p * n + c] = 1
    
    # Compute SVD
    try:
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
        # The rank is the number of non-zero singular values
        tol = 1e-10
        rank_found = np.sum(s > tol)
        
        if rank_found < rank_school and time.time() < deadline:
            # We have a decomposition of rank rank_found
            # U_svd[:, :r], s[:r], Vt[:r, :]
            r = rank_found
            U_decomp = U_svd[:, :r]
            S_decomp = s[:r]
            Vt_decomp = Vt[:r, :]
            
            # The decomposition is T = sum_r U[:,r] * S[r] * Vt[r,:]
            # So T[a, bc] = sum_r U[a,r] * S[r] * Vt[r, bc]
            # We need U[a, r'] = U[a,r] * S[r], V[bc, r'] = Vt[r, bc]
            # Then T = U' * V' where U' = U * diag(S), V' = Vt.T
            # But we need 3-way decomposition, not 2-way.
            
            # Actually, the matricization rank gives us a 2-factor decomposition of the matricized tensor.
            # For the 3-way tensor, we need T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # The SVD gives T[a, bc] = sum_r (U_svd[a,r] * s[r]) * Vt[r, bc]
            # We can set U[a,r] = U_svd[a,r] * s[r], and V[b,r]*W[c,r] = Vt[r, b*n*p+c]
            # But we need to factor Vt[r, bc] into V[b,r] * W[c,r] for each r.
            # This is not generally possible.
            
            pass  # This approach doesn't directly give a 3-way decomposition
    
    except:
        pass
    
    # For now, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

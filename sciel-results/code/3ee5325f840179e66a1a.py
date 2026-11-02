import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Note: The prompt says W[k*n+i, r].
        # Standard tensor decomposition: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # But the interface specifies shapes:
        # U: (n*m, rank) -> Index i*m+j
        # V: (m*p, rank) -> Index j*p+k
        # W: (p*n, rank) -> Index k*n+i
        
        # Let's map the indices carefully.
        # T is defined over i in [0,n), j in [0,m), k in [0,p).
        # The value is 1.
        # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
        # No, standard matrix multiplication tensor is:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor T_{i, j, k} such that C_{i, k} = sum_j A_{i, j} T_{i, j, k} B_{j, k}
        # Here T_{i, j, k} = 1 if we are doing standard multiplication?
        # Wait. The tensor for matrix multiplication $A \times B$ is defined such that
        # $(A \times B)_{ik} = \sum_j A_{ij} B_{jk}$.
        # This is a bilinear map. The tensor is $T_{ij}^k$?
        # The prompt says "Matrix multiplication 2,3,3".
        # "output factor follows the official trace convention W[k*n+i,r]"
        
        # Let's stick to the prompt's definition of U, V, W shapes.
        # U has shape (n*m, rank). Index $u = i \cdot m + j$.
        # V has shape (m*p, rank). Index $v = j \cdot p + k$.
        # W has shape (p*n, rank). Index $w = k \cdot n + i$.
        
        # The decomposition condition is usually:
        # $\sum_r U[u, r] V[v, r] W[w, r]$ ... this doesn't make sense as a single scalar for all i,j,k.
        # It must be that for each $i, k$, the product $C_{ik}$ is computed.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # In tensor decomposition terms, the multiplication tensor $\mathcal{M}$ has entries $\mathcal{M}_{i, j, k} = \delta_{ik}$? No.
        # The tensor $\mathcal{M}$ is defined such that $\langle \mathcal{M}, A \otimes B \rangle = \text{vec}(A B)$.
        # $\mathcal{M}_{i, j, k} = 1$ if the index in the output $k$ matches the index in B, and $i$ matches A?
        # Actually, the standard tensor for $n \times m$ by $m \times p$ multiplication has rank $nmp$ for schoolbook.
        # The entries are 1.
        # Specifically, $T[i, j, k] = 1$ for all $i,j,k$?
        # Let's check the schoolbook code in the prompt:
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This implies the term is $1 \cdot 1 \cdot 1 = 1$.
        # And the sum over $j$ for fixed $i, k$ gives $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So the tensor $T$ has $T[i, j, k] = 1$ for all $i, j, k$.
        
        # Let's verify the reconstruction.
        # We need to compute $C_{ik} = \sum_r (\sum_j A_{ij} U[i*m+j, r]) \dots$? No.
        # The factorization is:
        # $C_{ik} = \sum_r (\sum_j A_{ij} U_{i,j,r}) \dots$?
        # The standard form is:
        # $C_{ik} = \sum_r \left( \sum_j A_{ij} U_{i,j,r} \right) \left( \sum_k B_{jk} V_{j,k,r} \right) W_{k,i,r}$?
        # No, the factors are applied to A and B to produce intermediate terms $M_r$, and then combined to form C.
        # $M_r = \sum_{i,j} A_{ij} U_{i,j,r}$
        # $N_r = \sum_{j,k} B_{jk} V_{j,k,r}$
        # $C_{ik} = \sum_r M_r N_r W_{k,i,r}$?
        # Let's check the schoolbook case.
        # $U_{i,j,r} = 1$ if $r$ corresponds to $(i,j)$.
        # $M_r = A_{i_r, j_r}$.
        # $V_{j,k,r} = 1$ if $r$ corresponds to $(j_r, k)$. Note $j_r$ is fixed for a given $r$ in schoolbook?
        # In schoolbook, $r$ is a triple $(i,j,k)$.
        # $U_{i,j,r} = 1$ if $i=i_r, j=j_r$.
        # $V_{j,k,r} = 1$ if $j=j_r, k=k_r$.
        # $W_{k,i,r} = 1$ if $k=k_r, i=i_r$.
        # $M_r = A_{i_r, j_r}$.
        # $N_r = B_{j_r, k_r}$.
        # $C_{ik} = \sum_r A_{i_r, j_r} B_{j_r, k_r} W_{k,i,r}$.
        # Since $W_{k,i,r} = 1$ only if $k=k_r, i=i_r$, the sum becomes $A_{i,j} B_{j,k}$ when $i=i_r, k=k_r$.
        # So yes, $T[i,j,k] = 1$ is the correct tensor to match against if we view the decomposition as:
        # $\sum_r U_r \otimes V_r \otimes W_r = T$ where $T[i,j,k]=1$.
        # And the reconstruction is $C_{ik} = \sum_r (\sum_{i',j'} A_{i'j'} U_{i'j',r}) (\sum_{j',k'} B_{j'k'} V_{j'k',r}) W_{ki,r}$.
        
        # So the error function should be:
        # Compute the tensor $T_{calc}[i, j, k] = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$.
        # Compare $T_{calc}$ to $T_{target}$ (which is all ones).
        
        U_flat = U.reshape(n*m, -1)
        V_flat = V.reshape(m*p, -1)
        W_flat = W.reshape(p*n, -1)
        
        # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can use einsum or loops.
        # T_calc = np.einsum('ir,jr,kr->ijk', 
        #                    U_flat.reshape(n, m, -1), 
        #                    V_flat.reshape(m, p, -1), 
        #                    W_flat.reshape(p, n, -1))
        
        # Wait, W index is k*n+i. So W_flat[k*n+i, r].
        # Reshape W_flat to (p, n, R).
        # U_flat to (n, m, R).
        # V_flat to (m, p, R).
        
        U_3d = U_flat.reshape(n, m, -1)
        V_3d = V_flat.reshape(m, p, -1)
        W_3d = W_flat.reshape(p, n, -1)
        
        T_calc = np.einsum('imr,jpr,knr->ijk', U_3d, V_3d, W_3d)
        
        # Check exactness
        return np.allclose(T_calc, T, atol=1e-9)

    def solve_rank(R, time_limit):
        start_time = time.time()
        best_error = float('inf')
        best_factors = None
        
        # Try multiple restarts
        while time.time() - start_time < time_limit:
            # Initialize
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # ALS
            for iter in range(100):
                # Update U
                # Fix V, W. Solve for U.
                # T_calc[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # For fixed i,j, we want U[i*m+j, :] to minimize sum_k (T[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
                # This is a linear least squares problem for each (i,j).
                # Let A_{i,j} be the vector of U[i*m+j, :].
                # The term is sum_k A_{i,j} @ (V[j*p+k, :] * W[k*n+i, :])
                # Let M_{i,j,k, r} = V[j*p+k, r] * W[k*n+i, r]
                # We want A_{i,j} @ M_{i,j,k, :} = T[i,j,k] for all k.
                # This is A_{i,j} @ M_{i,j} = T[i,j, :]
                # A_{i,j} is (1, R). M_{i,j} is (R, p). T[i,j,:] is (p,).
                
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ M = T[i,j,:]
                        # A is (1, R).
                        # M.T @ A.T = T[i,j,:].T
                        # A.T = np.linalg.lstsq(M.T, T[i,j,:].T, rcond=None)[0]
                        # A = A.T.T
                        try:
                            A_sol = np.linalg.lstsq(M.T, T[i,j,:].T, rcond=None)[0]
                            U[i*m+j, :] = A_sol
                        except:
                            pass

                # Update V
                # Fix U, W. Solve for V.
                # For fixed j,k, V[j*p+k, :]
                # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let B_{j,k} be V[j*p+k, :].
                # T[i,j,k] = sum_i B_{j,k} @ (U[i*m+j, :] * W[k*n+i, :])
                # B_{j,k} @ N_{j,k} = T[:,j,k]
                # N_{j,k, i, r} = U[i*m+j, r] * W[k*n+i, r]
                # N_{j,k} is (n, R).
                
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            B_sol = np.linalg.lstsq(N.T, T[:,j,k].T, rcond=None)[0]
                            V[j*p+k, :] = B_sol
                        except:
                            pass

                # Update W
                # Fix U, V. Solve for W.
                # For fixed k,i, W[k*n+i, :]
                # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let C_{k,i} be W[k*n+i, :].
                # T[i,j,k] = sum_j C_{k,i} @ (U[i*m+j, :] * V[j*p+k, :])
                # C_{k,i} @ P_{k,i} = T[:,j,k] summed over j?
                # T[i,j,k] for fixed k,i is a vector over j.
                # C_{k,i} @ P_{k,i} = T[:, :, k][:, i].T?
                # P_{k,i, j, r} = U[i*m+j, r] * V[j*p+k, r]
                # P_{k,i} is (m, R).
                
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # T[:, j, k] for fixed i is T[i, :, k]
                        try:
                            C_sol = np.linalg.lstsq(P.T, T[i,:,k].T, rcond=None)[0]
                            W[k*n+i, :] = C_sol
                        except:
                            pass
                
                # Check error
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                T_calc = np.einsum('imr,jpr,knr->ijk', U_3d, V_3d, W_3d)
                error = np.linalg.norm(T_calc - T)
                
                if error < best_error:
                    best_error = error
                    best_factors = (U.copy(), V.copy(), W.copy())
                    
                if error < 1e-9:
                    break

        if best_factors is None:
            return None
            
        U, V, W = best_factors
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
        return None

    # Try ranks 15, 16, 17
    for R in [15, 16, 17]:
        time_limit = 30.0
        result = solve_rank(R, time_limit)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to Rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

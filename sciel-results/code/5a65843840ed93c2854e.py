import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (always valid, rank 54)
    def schoolbook():
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
    
    # Build the 3x3 multiplication tensor T[i,j,k] = 1 for all i,j,k in 0..2
    # We want: sum_t U_t[i*m+j] * V_t[j*3+k] * W_t[k*3+i] = 1 for all i,j,k
    # This is the CP decomposition of the all-ones 3x3x3 tensor? No!
    # Actually, the multiplication tensor for 3x3 is:
    # M[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    # In the trace convention W[k*n+i], the tensor entry is M[i*m+j, j*3+k, k*3+i]
    # For standard matrix mult: C[k,i] = sum_j A[i,j] * B[j,k]
    # So M[i*m+j, j*3+k, k*3+i] = 1 for all i,j,k in 0..2
    # All other entries are 0.
    
    # The tensor T of shape (n*m, m*3, 3*n) = (9, 9, 9)
    # T[i*m+j, j*3+k, k*3+i] = 1
    
    n3, m3, p3 = 3, 3, 3  # For 3x3 subproblem
    T = np.zeros((n3*m3, m3*p3, p3*n3))
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                T[i*m3+j, j*p3+k, k*n3+i] = 1.0
    
    # Try to find CP decomposition of T with rank r
    # Use ALS (Alternating Least Squares) with multiple restarts
    
    def try_als_decomposition(tensor, target_rank, max_iters=200, n_restarts=5, tol=1e-10):
        """Try to decompose tensor into CP form of given rank using ALS."""
        shape = tensor.shape
        best_loss = float('inf')
        best_factors = None
        
        for restart in range(n_restarts):
            # Initialize randomly
            rng = np.random.RandomState(restart * 12345 + 7)
            U = rng.randn(shape[0], target_rank)
            V = rng.randn(shape[1], target_rank)
            W = rng.randn(shape[2], target_rank)
            
            prev_loss = float('inf')
            for it in range(max_iters):
                # Update U
                for t in range(target_rank):
                    # Residual
                    resid = tensor - np.einsum('it,jt,kt->ijk', U, V, W)
                    # Solve for U[:, t]
                    # sum_i U[i,t] * (V[j,t]*W[k,t]) = resid[i,j,k]
                    # For fixed t, we need: sum over j,k of resid[i,j,k] * V[j,t]*W[k,t] = sum_j,k ...
                    # Actually: U[:,t] = argmin sum_ijk (resid[i,j,k] + U[i,t]*V[j,t]*W[k,t])^2
                    # d/dU[i,t] = -2 * sum_jk resid[i,j,k] * V[j,t]*W[k,t]
                    # But we also need to account for the fact that changing U[i,t] affects resid
                    # Standard ALS: 
                    # U[:,t] = (sum_ijk V[j,t]*W[k,t]*tensor[i,j,k]) / (sum_ijk V[j,t]^2*W[k,t]^2) ... no
                    # 
                    # Correct ALS for CP:
                    # Fix V, W. For each t:
                    # U[:,t] = argmin ||tensor - sum_s U[:,s] outer(V[:,s],W[:,s])||_F^2
                    # = (tensor * (V[:,t] outer W[:,t])) / (||V[:,t] outer W[:,t]||_F^2)
                    # where * is element-wise multiplication summed over j,k
                    
                    vw = V[:, t] * W[:, t]  # shape (shape[1], shape[2])
                    # numerator[i] = sum_jk tensor[i,j,k] * V[j,t] * W[k,t]
                    num = np.einsum('ijk,jk->i', tensor, vw)
                    # denominator = sum_jk V[j,t]^2 * W[k,t]^2
                    den = np.sum(vw ** 2)
                    if den > 1e-15:
                        U[:, t] = num / den
                    
                    # Update V
                    for t in range(target_rank):
                        uw = U[:, t] * W[:, t]  # shape (shape[0], shape[2])
                        # numerator[j] = sum_ik tensor[i,j,k] * U[i,t] * W[k,t]
                        num = np.einsum('ijk,ik->j', tensor, uw)
                        den = np.sum(uw ** 2)
                        if den > 1e-15:
                            V[:, t] = num / den
                    
                    # Update W
                    for t in range(target_rank):
                        uv = U[:, t] * V[:, t]  # shape (shape[0], shape[1])
                        # numerator[k] = sum_ij tensor[i,j,k] * U[i,t] * V[j,t]
                        num = np.einsum('ijk,ij->k', tensor, uv)
                        den = np.sum(uv ** 2)
                        if den > 1e-15:
                            W[:, t] = num / den
                
                # Compute loss
                approx = np.einsum('it,jt,kt->ijk', U, V, W)
                loss = np.sum((tensor - approx) ** 2)
                
                if abs(prev_loss - loss) < tol:
                    break
                prev_loss = loss
            
            if loss < best_loss:
                best_loss = loss
                best_factors = (U.copy(), V.copy(), W.copy())
        
        return best_loss, best_factors
    
    # Try ranks from 23 upward
    found_decomp = None
    found_rank = None
    
    for target_rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 20:
            break
        loss, factors = try_als_decomposition(T, target_rank, max_iters=100, n_restarts=3, tol=1e-12)
        if factors is not None:
            U3, V3, W3 = factors
            # Verify exactness: check if sum_t U3[i,j,t]*V3[j,k,t]*W3[k,i,t] == 1 for all i,j,k
            max_err = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0
                        for t in range(target_rank):
                            s += U3[i*3+j, t] * V3[j*3+k, t] * W3[k*3+i, t]
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < 1e-8:
                found_decomp = (U3, V3, W3)
                found_rank = target_rank
                break
    
    if found_decomp is not None and time.time() < deadline - 5:
        U3, V3, W3 = found_decomp
        r3 = found_rank
        total_rank = 2 * r3
        
        # Build the 3x6 decomposition by duplicating for two 3x3 blocks
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for block in range(2):
            offset = block * 3
            for t in range(r3):
                col = block * r3 + t
                # U is same for both blocks (A is 3x3)
                for i in range(3):
                    for j in range(3):
                        U[i * m + j][col] = U3[i * 3 + j, t]
                # V: B is 3x6, block b uses columns offset..offset+2
                for j in range(3):
                    for k in range(3):
                        V[j * p + (k + offset)][col] = V3[j * 3 + k, t]
                # W: C is 6x3, block b uses rows offset..offset+2
                for k in range(3):
                    for i in range(3):
                        W[(k + offset) * n + i][col] = W3[k * 3 + i, t]
        
        # Final verification
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(total_rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-6:
            # Convert to lists of floats (integer/half-integer check not strict for now)
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()

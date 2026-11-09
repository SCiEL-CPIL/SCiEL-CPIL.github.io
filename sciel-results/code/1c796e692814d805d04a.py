import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Helper to check if a decomposition is valid and return the rounded matrices if possible
    def check_and_round(U, V, W, tol=1e-7):
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (1.0 - s)**2
        if err < tol:
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            # Verify exactness with rounded values
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(U_r.shape[1]):
                            s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        if abs(1.0 - s) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_r, V_r, W_r
            else:
                # If rounding fails, we cannot guarantee exactness with these floats
                # Unless we assume the evaluator accepts floats close to integers?
                # The contract says "integer/half-integer entries".
                # So we should probably not return non-rounded floats.
                return None
        return None

    def gradient_descent(rank, steps=2000, lr=0.01, seed=0):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Precompute indices for speed? Dimensions are small, loops are fine.
        for step in range(steps):
            # Compute S[i, j, k]
            S = np.zeros((n, m, p))
            for r in range(rank):
                # U[:, r] is (n*m, 1) -> reshape to (n, m)
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                # S[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
                # This is a tensor contraction.
                # For small sizes, explicit loops or np.einsum
                # np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
                S += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            
            # Compute gradients
            # dE/dU[i,j,r] = -2 * (1 - S[i,j,k]) * V[j,k,r] * W[k,i,r]  (sum over k)
            # dE/dV[j,k,r] = -2 * (1 - S[i,j,k]) * U[i,j,r] * W[k,i,r]  (sum over i)
            # dE/dW[k,i,r] = -2 * (1 - S[i,j,k]) * U[i,j,r] * V[j,k,r]  (sum over j)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Residual R[i,j,k] = 1 - S[i,j,k]
            Res = 1.0 - S
            
            for r in range(rank):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                
                # dU[i,j] = -2 * sum_k Res[i,j,k] * Vr[j,k] * Wr[k,i]
                # dU = -2 * np.einsum('ijk,jk,ki->ij', Res, Vr, Wr)
                dU_r = -2.0 * np.einsum('ijk,jk,ki->ij', Res, Vr, Wr)
                
                # dV[j,k] = -2 * sum_i Res[i,j,k] * Ur[i,j] * Wr[k,i]
                # dV = -2 * np.einsum('ijk,ij,ki->jk', Res, Ur, Wr)
                dV_r = -2.0 * np.einsum('ijk,ij,ki->jk', Res, Ur, Wr)
                
                # dW[k,i] = -2 * sum_j Res[i,j,k] * Ur[i,j] * Vr[j,k]
                # dW = -2 * np.einsum('ijk,ij,jk->ki', Res, Ur, Vr)
                dW_r = -2.0 * np.einsum('ijk,ij,jk->ki', Res, Ur, Vr)
                
                dU[:, r] = dU_r.reshape(-1)
                dV[:, r] = dV_r.reshape(-1)
                dW[:, r] = dW_r.reshape(-1)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Optional: Normalize to prevent drift?
            # For bilinear forms, scaling U up, V down, W down can happen.
            # We can periodically re-normalize columns.
            if step % 100 == 0:
                for r in range(rank):
                    normU = np.linalg.norm(U[:, r])
                    normV = np.linalg.norm(V[:, r])
                    normW = np.linalg.norm(W[:, r])
                    if normU > 1e-9 and normV > 1e-9 and normW > 1e-9:
                        # Scale such that product of norms is 1? Or just keep them bounded.
                        # Let's try to keep norms around 1.
                        scale = (normU * normV * normW)**(1/3)
                        if scale > 1e-9:
                            U[:, r] /= scale
                            V[:, r] /= scale
                            W[:, r] /= scale
            
        # Final check
        return check_and_round(U, V, W)

    best_result = None
    best_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Try ranks from likely optimal up to schoolbook
    # Known bounds for 4x4x6: Lower 59. Upper 60-64?
    # Let's try 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
    # If we find a valid one, we return it.
    # We should try lower ranks first.
    
    # If time is short, we might not finish high ranks.
    # But schoolbook is always valid.
    
    ranks_to_try = list(range(60, 76))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            res = gradient_descent(R, steps=3000, lr=0.005, seed=seed)
            if res is not None:
                U, V, W = res
                # Convert to lists of lists of floats (or ints if integer)
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Clean up -0.0 and floats that are integers
                def clean_matrix(mat):
                    cleaned = []
                    for row in mat:
                        new_row = []
                        for val in row:
                            if abs(val - round(val)) < 1e-9:
                                new_row.append(int(round(val)))
                            elif abs(val*2 - round(val*2)) < 1e-9:
                                new_row.append(val) # Keep as float if half-integer
                            else:
                                new_row.append(val)
                        cleaned.append(new_row)
                    return cleaned
                
                U_list = clean_matrix(U_list)
                V_list = clean_matrix(V_list)
                W_list = clean_matrix(W_list)
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R
                found = True
                break
        
        if found:
            # If we found a low rank, we might want to try even lower?
            # But the loop goes 60, 61... so we stop at the first valid one.
            # Wait, the loop iterates R from 60 upwards.
            # So if 60 is valid, we return 60.
            # If 60 fails, we try 61.
            # This is correct.
            break

    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END

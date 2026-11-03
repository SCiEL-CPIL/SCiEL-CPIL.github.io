import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # The decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # (This is the standard matrix multiplication tensor where all entries are 1
    #  because T[a,b,c] = 1 iff the indices match in the appropriate way)
    
    # Actually, let me re-read the task more carefully.
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 iff i=i', j=j', k=k'
    
    # So the tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # T[a,b,c] = 0 otherwise
    
    # This is the standard matrix multiplication tensor.
    # The decomposition is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For the schoolbook algorithm:
    # For each (i,j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r indexes the (i,j,k) triple
    
    # Let's build the tensor properly
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # First, try to construct a Strassen-based decomposition
    # For 2x4 * 4x4:
    # A is 2x4, B is 4x4, C = A*B is 2x4
    # 
    # We can view this as:
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    #
    # Using Strassen for each 2x2 multiplication:
    # C1 = A1*B11 + A2*B21: this requires computing two 2x2 products
    # C2 = A1*B12 + A2*B22: this requires computing two 2x2 products
    #
    # But we can be smarter. Let's use the fact that we're multiplying
    # a 2x4 by a 4x4.
    
    # Actually, let me think about this differently.
    # The rank of matrix multiplication tensor for n x m * m x p is at most n*m*p (schoolbook)
    # For 2x4 * 4x4, schoolbook gives rank 32.
    # 
    # Can we do better? 
    # One approach: use Strassen's algorithm on 2x2 blocks.
    # 
    # A (2x4) = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B (4x4) = [[B11, B12], [B21, B22]]
    # C (2x4) = [C_left | C_right]
    # 
    # C_left = A_left * B11 + A_right * B21
    # C_right = A_left * B12 + A_right * B22
    # 
    # Each of these is a sum of two 2x2 matrix multiplications.
    # Using Strassen, each 2x2 multiplication takes 7 scalar products.
    # So C_left needs 14 products, C_right needs 14 products, total 28.
    # But we can share some products!
    
    # Let's try a direct construction.
    # 
    # For C_left = A_left * B11 + A_right * B21:
    # Let M1 = A_left * B11 (7 Strassen products)
    # Let M2 = A_right * B21 (7 Strassen products)
    # C_left = M1 + M2
    # 
    # For C_right = A_left * B12 + A_right * B22:
    # Let M3 = A_left * B12 (7 Strassen products)
    # Let M4 = A_right * B22 (7 Strassen products)
    # C_right = M3 + M4
    # 
    # Total: 28 products. But can we do better?
    
    # Alternative: think of it as multiplying two 2x4 matrices.
    # Actually, let me just try to find a rank-28 decomposition using the
    # alternating minimization approach, starting from the Strassen initialization.
    
    # First, let me construct the Strassen decomposition explicitly.
    # 
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our problem:
    # C_left = A_left * B11 + A_right * B21
    # 
    # Let's define:
    # A_left = [[a00, a01], [a10, a11]]
    # A_right = [[a02, a03], [a12, a13]]
    # B11 = [[b00, b01], [b10, b11]]
    # B21 = [[b20, b21], [b30, b31]]
    # 
    # C_left = A_left * B11 + A_right * B21
    # 
    # This is equivalent to computing:
    # [A_left | A_right] * [[B11], [B21]]
    # 
    # Where [A_left | A_right] is 2x4 and [[B11], [B21]] is 4x2.
    # 
    # Hmm, this is getting complicated. Let me just use the alternating
    # minimization approach with a good initialization.
    
    # Let's try rank 24, 26, 28, 30, 32
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for R in [24, 26, 28, 30, 32]:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + restart * 1000)
            
            # Initialize with random values, but scale appropriately
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Normalize columns to have unit norm
            for r in range(R):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Alternating minimization
            for iteration in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U given V, W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Let A[r, (b,c)] = V[b,r] * W[c,r]
                # Then T[a, :, :].flatten() = U[a, :] @ A
                # U[a, :] = T[a, :, :].flatten() @ A^+
                
                A = np.zeros((R, m*p * p*n))
                for r in range(R):
                    for b in range(m*p):
                        for c in range(p*n):
                            A[r, b * p*n + c] = V[b, r] * W[c, r]
                
                T_flat = T.reshape(n*m, m*p * p*n)
                try:
                    ATA = A @ A.T
                    ATA += 1e-12 * np.eye(R)
                    U_new = T_flat @ A.T @ np.linalg.inv(ATA)
                except np.linalg.LinAlgError:
                    break
                
                # Solve for V given U, W
                B_mat = np.zeros((R, n*m * p*n))
                for r in range(R):
                    for a in range(n*m):
                        for c in range(p*n):
                            B_mat[r, a * p*n + c] = U[a, r] * W[c, r]
                
                T_flat2 = T.reshape(m*p, n*m * p*n)
                try:
                    BTA = B_mat @ B_mat.T
                    BTA += 1e-12 * np.eye(R)
                    V_new = T_flat2 @ B_mat.T @ np.linalg.inv(BTA)
                except np.linalg.LinAlgError:
                    break
                
                # Solve for W given U, V
                C_mat = np.zeros((R, n*m * m*p))
                for r in range(R):
                    for a in range(n*m):
                        for b in range(m*p):
                            C_mat[r, a * m*p + b] = U[a, r] * V[b, r]
                
                T_flat3 = T.reshape(p*n, n*m * m*p)
                try:
                    CTA = C_mat @ C_mat.T
                    CTA += 1e-12 * np.eye(R)
                    W_new = T_flat3 @ C_mat.T @ np.linalg.inv(CTA)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                
                if err < 1e-10:
                    best_result = (R, U.copy(), V.copy(), W.copy())
                    break
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

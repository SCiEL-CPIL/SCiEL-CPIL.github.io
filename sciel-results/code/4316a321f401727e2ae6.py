"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # p = 9 = 3*3, so we can use block-circulant structure
    # The idea: view the 3x9 multiplication as 3x3 matrix multiplication
    # over a 3-dimensional algebra (group algebra of Z_3)
    
    # For 2x3 times 3x3, schoolbook rank is 18
    # Using the group algebra Z_3 = {e, g, g^2} with g^3 = e
    # We can construct a rank-18 decomposition by tensoring
    # the standard 2x3x3 decomposition with the algebra basis
    
    # Actually, let's use a simpler approach:
    # For 2x3 x 3x9, we can think of the 9 columns as 3 blocks of 3
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # Write k = 3*q + r where q in {0,1,2}, r in {0,1,2}
    # Then C[i, 3q+r] = sum_j A[i,j] * B[j, 3q+r]
    
    # We can use the standard 2x3x3 decomposition for each block q
    # The standard 2x3x3 matrix multiplication tensor has rank 18
    # (schoolbook) but we might do better
    
    # Let's try to construct a rank-18 decomposition using block structure
    
    # First, let's verify what rank is achievable
    # For 2x3 x 3x3, the known minimum rank is 18 (schoolbook)
    # Actually, for 2x3 x 3x3, the rank is at most 18 (schoolbook)
    # and at least max(2*3, 3*3, 2*3) = 9
    # The actual minimum is not known to be less than 18 for these small dimensions
    
    # Let's just use the schoolbook decomposition for now, but try to find
    # a better one using the block structure
    
    # Schoolbook rank for 2x3 x 3x9 is 54
    # But we can do better by exploiting the structure
    
    # Approach: Use the fact that 9 = 3*3 and the multiplication is linear
    # C[i, 3q+r] = sum_j A[i,j] * B[j, 3q+r]
    
    # We can use the standard 2x3x3 decomposition for each of the 3 blocks
    # This gives rank 3 * 18 = 54, which is the same as schoolbook
    
    # To do better, we need to share structure across blocks
    
    # Let's try a different approach: use the group algebra Z_3
    # Define omega = exp(2*pi*i/3)
    # The group algebra has basis {1, omega, omega^2}
    # Multiplication: omega * omega = omega^2, omega^2 * omega = 1, etc.
    
    # Actually, let's just try to find a lower-rank decomposition using ALS
    # with a better initialization based on the block structure
    
    start_time = time.time()
    time_limit = 120  # Total time limit
    
    # Build the tensor
    tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a rank-r decomposition using ALS
    def try_rank(r, num_restarts=3, max_iter=200, time_budget=30):
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        restart_start = time.time()
        
        for restart in range(num_restarts):
            if time.time() - restart_start > time_budget:
                break
            
            # Initialize with block structure in mind
            # For the 2x3 x 3x9 case, use a structured initialization
            if r <= 54:
                # Use a random initialization with some structure
                np.random.seed(restart * 100 + r)
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
                
                # Add some structure: initialize based on the known non-zero pattern
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Assign some weight to the first few columns
                            for d in range(min(r, 3)):
                                U[a, d] += 1.0 / (d + 1)
                                V[b, d] += 1.0 / (d + 1)
                                W[c, d] += 1.0 / (d + 1)
            else:
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            
            for it in range(max_iter):
                if time.time() - restart_start > time_budget:
                    break
                
                # Update U
                design = np.zeros((m * p * p * n, r))
                for d in range(r):
                    outer = np.outer(V[:, d], W[:, d])
                    design[:, d] = outer.flatten()
                
                for a in range(n * m):
                    target = tensor[a, :, :].flatten()
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Update V
                design = np.zeros((n * m * p * n, r))
                for d in range(r):
                    outer = np.outer(U[:, d], W[:, d])
                    design[:, d] = outer.flatten()
                for b in range(m * p):
                    target = tensor[:, b, :].flatten()
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Update W
                design = np.zeros((n * m * m * p, r))
                for d in range(r):
                    outer = np.outer(U[:, d], V[:, d])
                    design[:, d] = outer.flatten()
                for c in range(p * n):
                    target = tensor[:, :, c].flatten()
                    try:
                        W[c, :], _, _, _ = np.linalg.lstsq(design, target, rcond=None)
                    except:
                        pass
                
                # Check error
                if it % 10 == 0 or it == max_iter - 1:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = 0.0
                                for d in range(r):
                                    val += U[a, d] * V[b, d] * W[c, d]
                                err += (val - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-10:
                        break
        
        return best_err, best_U, best_V, best_W
    
    # Try ranks from 18 upward (since 9 is a lower bound, and 18 is a natural candidate)
    # But first, let's try to construct a rank-18 decomposition analytically
    
    # For 2x3 x 3x9, we can use the following structure:
    # Write the 3x9 matrix B as three 3x3 blocks: B = [B_0 | B_1 | B_2]
    # Then C = A * B = [A*B_0 | A*B_1 | A*B_2]
    # Each A*B_q is a 2x3 matrix, computed by 2x3 x 3x3 multiplication
    
    # The 2x3 x 3x3 multiplication tensor has rank at most 18 (schoolbook)
    # If we can find a rank-18 decomposition for 2x3 x 3x3, then we can
    # use it three times (once for each block) to get a rank-54 decomposition
    # for 2x3 x 3x9. But that's not better than schoolbook.
    
    # To do better, we need to share the decomposition across blocks.
    # This is where the group algebra approach comes in.
    
    # Let's try ranks from 18 to 54 and see what we can find
    for r in range(18, 55):
        if time.time() - start_time > time_limit - 10:
            break
        
        err, U, V, W = try_rank(r, num_restarts=2, max_iter=100, time_budget=15)
        
        if err < 1e-8:
            # Found a valid decomposition, round to nearest 0.5
            U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
            V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
            W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
            
            # Verify
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rounded[i*m+j][d] * V_rounded[j*p+k][d] * W_rounded[k*n+i][d] for d in range(r))
                        if abs(val - 1.0) > 1e-6:
                            verified = False
                            break
                    if not verified:
                        break
                if not verified:
                    break
            
            if verified:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    # If we didn't find a better decomposition, return schoolbook
    rank = n * m * p  # 54
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
# EVOLVE-BLOCK-END

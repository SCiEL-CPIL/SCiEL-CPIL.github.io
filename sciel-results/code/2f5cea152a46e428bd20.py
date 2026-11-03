"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 130s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    def verify(U, V, W, r):
        """Verify that U, V, W form a valid rank-r decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank20_embedding():
        """
        Embed 2x3x6 into 6x6x6 and use a rank-20 Strassen-like decomposition.
        
        Strategy: 
        - A is 2x3, B is 3x6, C = A*B is 2x6
        - Embed A into 6x6: place in rows 0-1, cols 0-2
        - Embed B into 6x6: place in rows 0-2, cols 0-5
        - The top-left 2x6 of the 6x6 product gives C
        
        Use a known rank-20 decomposition for 6x6x6 matrix multiplication.
        """
        # For 6x6 matrices partitioned into 2x2 blocks of size 3x3:
        # A = [[A11, A12], [A21, A22]] where each is 3x3
        # B = [[B11, B12], [B21, B22]] where each is 3x3
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen for 2x2 blocks of 3x3:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # But we need rank-20 for the full 6x6x6, not just the block structure.
        # The above gives 7 * (rank of 3x3) = 7*27 = 189, which is too high.
        
        # Instead, let's try to directly construct a rank-20 solution.
        # The key insight: we only need the top-left 2x6 of the 6x6 product.
        # This means we can potentially use fewer terms.
        
        # Let's try a direct algebraic approach for rank 20.
        # We need: sum_{c=1}^{20} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # Try using the structure: C[i,k] = sum_j A[i,j] * B[j,k]
        # We can write this as: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        
        # A rank-1 decomposition would be: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This equals 1 only when (i,j,k) is a valid triple.
        
        # Let's try a structured approach with rank 20.
        # Use the fact that 2*3*6 = 36, and we want to reduce to 20.
        
        # One approach: use intermediate sums.
        # Define S1 = A[:,0] + A[:,1]  (2x1 vector)
        # Define S2 = B[0,:] + B[1,:]  (1x6 vector)  
        # etc.
        
        # Actually, let me try a more systematic search for rank 20.
        
        np.random.seed(42)
        r = 20
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for iteration in range(50):
            if time.time() > DEADLINE:
                return None
            
            # Solve for V given U, W
            A_mat = np.zeros((n * m * p, m * p * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (j * p + k) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * W[k * n + i, c]
            
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V = v_flat.reshape(m * p, r)
            
            # Solve for U given V, W
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (i * m + j) * r + c
                            A_mat[e, u_idx] = V[j * p + k, c] * W[k * n + i, c]
            
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U = u_flat.reshape(n * m, r)
            
            # Solve for W given U, V
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (k * n + i) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = w_flat.reshape(p * n, r)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - 1.0) ** 2
            
            if error < 1e-9:
                # Try to round to half-integers
                U_rounded = np.array([[round_to_half(val) for val in row] for row in U])
                V_rounded = np.array([[round_to_half(val) for val in row] for row in V])
                W_rounded = np.array([[round_to_half(val) for val in row] for row in W])
                
                # Verify rounded solution
                if verify_np(U_rounded, V_rounded, W_rounded, r):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None
    
    # Try rank 20 first (the target from the embedding strategy)
    if time.time() < DEADLINE:
        res = try_rank20_embedding()
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Try ranks from 21 to 36 as fallback
    for r in range(21, 37):
        if time.time() > DEADLINE:
            break
            
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            
            # Solve for V given U, W
            A_mat = np.zeros((n * m * p, m * p * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (j * p + k) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * W[k * n + i, c]
            
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V = v_flat.reshape(m * p, r)
            
            # Solve for U given V, W
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (i * m + j) * r + c
                            A_mat[e, u_idx] = V[j * p + k, c] * W[k * n + i, c]
            
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U = u_flat.reshape(n * m, r)
            
            # Solve for W given U, V
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            u_idx = (k * n + i) * r + c
                            A_mat[e, u_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = w_flat.reshape(p * n, r)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - 1.0) ** 2
            
            if error < 1e-9:
                U_rounded = np.array([[round_to_half(val) for val in row] for row in U])
                V_rounded = np.array([[round_to_half(val) for val in row] for row in V])
                W_rounded = np.array([[round_to_half(val) for val in row] for row in W])
                
                if verify_np(U_rounded, V_rounded, W_rounded, r):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END

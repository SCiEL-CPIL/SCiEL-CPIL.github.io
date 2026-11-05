import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to use Laderman's rank-23 decomposition for 3x3x3
    # and split the 3x3x6 into two 3x3x3 multiplications
    
    # Laderman's algorithm terms for 3x3x3
    # Each term is (u, v, w) where u, v, w are 3x3 matrices (flattened to 9-vectors)
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
    
    # Here are the 23 terms from Laderman's algorithm
    # Indices are 0-based: A[i,j], B[j,k], C[i,k]
    # u corresponds to A coefficients, v to B coefficients, w to C coefficients
    
    laderman_terms = [
        # Term 1: A[0,0] * B[0,0] -> C[0,0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2: A[1,1] * B[1,1] -> C[1,1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 3: A[2,2] * B[2,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Term 4: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0]+C[0,1]+C[1,0]+C[1,1]
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        # Term 5: (A[1,1]+A[2,2]) * (B[1,1]+B[2,2]) -> C[1,1]+C[1,2]+C[2,1]+C[2,2]
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,1, 0,1,1]),
        # Term 6: (A[0,0]+A[2,2]) * (B[0,0]+B[2,2]) -> C[0,0]+C[0,2]+C[2,0]+C[2,2]
        ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,1, 0,0,0, 1,0,1]),
        # Term 7: (A[0,1]+A[1,1]) * (B[0,0]+B[1,0]) -> C[0,0]+C[0,1]+C[1,0]+C[1,1]
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        # Term 8: (A[0,0]+A[0,1]) * (B[1,0]+B[0,0]) -> C[0,0]+C[0,1]+C[1,0]+C[1,1]
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
        # Term 9: (A[1,0]+A[1,1]) * (B[0,1]+B[1,1]) -> C[0,1]+C[1,0]+C[1,1]+C[1,2]
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 1,1,1, 0,0,0]),
        # Term 10: (A[1,1]+A[1,0]) * (B[1,1]+B[0,1]) -> C[0,1]+C[1,0]+C[1,1]+C[1,2]
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 1,1,1, 0,0,0]),
        # Term 11: (A[2,1]+A[2,2]) * (B[0,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,1]+C[2,2]
        ([0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        # Term 12: (A[2,2]+A[2,1]) * (B[2,2]+B[0,2]) -> C[0,2]+C[1,2]+C[2,1]+C[2,2]
        ([0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        # Term 13: (A[2,0]+A[2,2]) * (B[1,2]+B[2,2]) -> C[1,2]+C[2,0]+C[2,1]+C[2,2]
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 1,1,1]),
        # Term 14: (A[2,2]+A[2,0]) * (B[2,2]+B[1,2]) -> C[1,2]+C[2,0]+C[2,1]+C[2,2]
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 1,1,1]),
        # Term 15: (A[0,0]+A[0,2]) * (B[2,0]+B[0,0]) -> C[0,0]+C[0,2]+C[2,0]+C[2,2]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [1,0,1, 0,0,0, 1,0,1]),
        # Term 16: (A[0,2]+A[0,0]) * (B[0,0]+B[2,0]) -> C[0,0]+C[0,2]+C[2,0]+C[2,2]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [1,0,1, 0,0,0, 1,0,1]),
        # Term 17: (A[0,2]+A[2,2]) * (B[2,0]+B[0,2]) -> C[0,2]+C[2,0]+C[2,1]+C[2,2]
        ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,1,1]),
        # Term 18: (A[2,0]+A[0,0]) * (B[0,2]+B[2,0]) -> C[0,2]+C[2,0]+C[2,1]+C[2,2]
        ([1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,1,0]),
        # Term 19: (A[0,1]+A[2,1]) * (B[1,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,1]+C[2,2]
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        # Term 20: (A[2,1]+A[0,1]) * (B[2,2]+B[1,2]) -> C[0,2]+C[1,2]+C[2,1]+C[2,2]
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,1], [0,0,1, 0,0,1, 0,1,1]),
        # Term 21: (A[1,0]+A[2,0]) * (B[0,2]+B[1,2]) -> C[0,2]+C[1,0]+C[1,2]+C[2,0]
        ([0,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,1, 1,0,1, 1,0,0]),
        # Term 22: (A[2,0]+A[1,0]) * (B[1,2]+B[0,2]) -> C[0,2]+C[1,0]+C[1,2]+C[2,0]
        ([0,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,1, 1,0,1, 1,0,0]),
        # Term 23: (A[0,1]+A[1,0]) * (B[1,0]+B[0,1]) -> C[0,1]+C[1,0]+C[1,1]+C[0,0]
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
    ]
    
    # Verify Laderman's decomposition for 3x3x3
    def verify_laderman(terms):
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
        # Wait, the indices need to be consistent.
        # U is (n*m, rank) -> index (i, j) where i is row of A, j is col of A
        # V is (m*p, rank) -> index (j, k) where j is row of B, k is col of B
        # W is (p*n, rank) -> index (k, i) where k is row of C (col of B), i is col of C (row of A)
        # So W[k*n+i, r]
        
        n3, m3, p3 = 3, 3, 3
        rank3 = len(terms)
        
        # Build expected tensor
        T_expected = np.zeros((n3 * m3 * p3,))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T_expected[i * m3 * p3 + j * p3 + k] = 1.0
        
        # Build tensor from decomposition
        T_decomp = np.zeros((n3 * m3 * p3,))
        for r, (u, v, w) in enumerate(terms):
            u_arr = np.array(u).reshape(3, 3)
            v_arr = np.array(v).reshape(3, 3)
            w_arr = np.array(w).reshape(3, 3)
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        T_decomp[i * m3 * p3 + j * p3 + k] += u_arr[i, j] * v_arr[j, k] * w_arr[i, k]
        
        return np.allclose(T_decomp, T_expected, atol=1e-10)
    
    # Check if our Laderman terms are correct
    laderman_valid = verify_laderman(laderman_terms)
    
    if laderman_valid and time.time() < deadline:
        # Use Laderman's decomposition for two 3x3x3 blocks
        # B is 3x6, split into B1 (3x3) and B2 (3x3)
        # C = A * B = [A*B1 | A*B2]
        
        rank_total = 2 * 23  # 46
        
        U = np.zeros((n * m, rank_total))  # (9, 46)
        V = np.zeros((m * p, rank_total))  # (18, 46)
        W = np.zeros((p * n, rank_total))  # (18, 46)
        
        for block in range(2):
            offset = block * 23
            for r, (u, v, w) in enumerate(laderman_terms):
                col = offset + r
                u_arr = np.array(u).reshape(3, 3)
                v_arr = np.array(v).reshape(3, 3)
                w_arr = np.array(w).reshape(3, 3)
                
                # Fill U: (i, j) -> u[i, j]
                for i in range(3):
                    for j in range(3):
                        U[i * 3 + j, col] = u_arr[i, j]
                
                # Fill V: (j, k) where k is 0..2 for block 0, 3..5 for block 1
                for j in range(3):
                    for k in range(3):
                        V[j * 6 + (block * 3 + k), col] = v_arr[j, k]
                
                # Fill W: (k, i) where k is 0..2 for block 0, 3..5 for block 1
                for k in range(3):
                    for i in range(3):
                        W[(block * 3 + k) * 3 + i, col] = w_arr[i, k]
        
        # Verify the full decomposition
        def verify_full(U, V, W, n, m, p):
            # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            # Actually, the tensor is T[i,j,k] = 1 for all i,j,k in the appropriate ranges
            # The decomposition should satisfy:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # In tensor form: T[i,j,k] = 1 (the matrix multiplication tensor)
            # And T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
            
            rank = U.shape[1]
            T_expected = np.ones((n, m, p))
            
            T_decomp = np.zeros((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            
            return np.allclose(T_decomp, T_expected, atol=1e-10)
        
        full_valid = verify_full(U, V, W, n, m, p)
        
        if full_valid:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

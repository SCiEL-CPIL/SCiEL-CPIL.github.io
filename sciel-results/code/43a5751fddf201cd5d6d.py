import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = 3, 4, 8

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    def get_strassen_222():
        # Strassen's algorithm for 2x2x2
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = A @ B
        #
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.array([
            [1, 0, 0, 0, 0, 0, 0], # a11
            [0, 0, 1, 0, 1, 0, 0], # a12
            [0, 0, 0, 0, 0, -1, 0], # a21
            [1, 1, 0, 1, 0, 0, 0]  # a22
        ])
        
        V = np.array([
            [1, 1, 0, 0, 0, 1, 0], # b11
            [0, 0, 1, 0, 0, 1, 0], # b12
            [1, 0, -1, 1, 0, 0, 1], # b21
            [1, 0, 0, -1, 1, 0, 1]  # b22
        ])
        
        W = np.array([
            [1, 0, 0, 0, 0, 0, 0], # C11
            [0, 0, 1, 0, 1, 0, 0], # C12
            [0, 1, 0, 1, 0, 0, 0], # C21
            [1, -1, 1, 0, 0, 1, 0] # C22
        ])
        
        # Note: The W matrix above is transposed relative to some conventions.
        # Let's verify the convention: W[k*n+i, r]
        # C[i, k] = sum_r W[k*n+i, r] * ...
        # For 2x2x2, n=2, p=2.
        # C[0,0] is index 0*2+0 = 0 in W (row 0)
        # C[0,1] is index 1*2+0 = 2 in W (row 2)
        # C[1,0] is index 0*2+1 = 1 in W (row 1)
        # C[1,1] is index 1*2+1 = 3 in W (row 3)
        
        # My W array above:
        # Row 0: [1, 0, 0, 0, 0, 0, 0] -> C11 coeff for M1
        # Row 1: [0, 0, 1, 0, 1, 0, 0] -> C21 coeff for M3, M5 ?? Wait.
        
        # Let's re-derive W carefully.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W rows correspond to C indices (k*n+i):
        # C[0,0] -> k=0, i=0 -> row 0. Coeffs: M1(1), M2(0), M3(0), M4(1), M5(-1), M6(0), M7(1)
        # C[0,1] -> k=1, i=0 -> row 2. Coeffs: M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0)
        # C[1,0] -> k=0, i=1 -> row 1. Coeffs: M1(0), M2(1), M3(0), M4(1), M5(0), M6(0), M7(0)
        # C[1,1] -> k=1, i=1 -> row 3. Coeffs: M1(1), M2(-1), M3(1), M4(0), M5(0), M6(1), M7(0)
        
        W_correct = np.array([
            [1, 0, 0, 1, -1, 0, 1], # C[0,0]
            [0, 1, 0, 1, 0, 0, 0],  # C[1,0]
            [0, 0, 1, 0, 1, 0, 0],  # C[0,1]
            [1, -1, 1, 0, 0, 1, 0]  # C[1,1]
        ])
        
        return U, V, W_correct

    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        rank = U_np.shape[1]
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Decomp: C[i,k] = sum_c W[k*n+i, c] * ( sum_j U[i*m+j, c] A[i,j] V[j*p+k, c] )
            
            # Efficient check using einsum
            # T_ijk = sum_c U_ic V_jk W_ki
            # T should be identity tensor for matmul: T_ijk = 1 if i,j,k correspond to A_ij B_jk -> C_ik
            # Actually, the tensor T for matmul is such that C = T(A,B).
            # T[i, j, k] = 1.
            
            # Let's just compute the result
            M = U_3d * A[:, :, np.newaxis] # (n, m, r)
            N = V_3d * B[:, :, np.newaxis] # (m, p, r)
            
            # Sum over j: S[i, k, c] = sum_j M[i, j, c] N[j, k, c]
            S = np.einsum('ijc,jkc->ikc', M, N)
            
            # C_dec[i, k] = sum_c S[i, k, c] W[k, i, c]
            # W_3d is (p, n, r) -> W[k, i, c]
            C_dec = np.einsum('ikc,kic->ik', S, W_3d)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def combine_decomps(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2, n, m, p):
        # We are splitting the problem into two blocks.
        # Block 1: n1 x m1 x p1
        # Block 2: n2 x m2 x p2
        # Total: n x m x p
        # We assume the split is along the n dimension (rows of A, rows of C)
        # A = [A1; A2] where A1 is n1xm, A2 is n2xm
        # B is m x p
        # C = [C1; C2] where C1 = A1 @ B (n1 x p), C2 = A2 @ B (n2 x p)
        
        # U, V, W shapes:
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r_total = r1 + r2
        
        U_new = np.zeros((n * m, r_total))
        V_new = np.zeros((m * p, r_total))
        W_new = np.zeros((p * n, r_total))
        
        # Fill Block 1
        # U1 corresponds to A1 (n1 x m)
        # Indices in U_new for block 1: i in [0, n1), j in [0, m)
        # U1 indices: i in [0, n1), j in [0, m)
        # Note: m1 should be m, p1 should be p?
        # In this split, the m and p dimensions are shared.
        # So m1 = m, p1 = p. n1 + n2 = n.
        
        for i in range(n1):
            for j in range(m):
                U_new[i * m + j, :r1] = U1[i * m + j, :]
                
        for j in range(m):
            for k in range(p):
                V_new[j * p + k, :r1] = V1[j * p + k, :]
                
        # W1 corresponds to C1 (n1 x p)
        # W1 indices: k in [0, p), i in [0, n1)
        # W_new indices: k in [0, p), i in [0, n)
        # For block 1, i is in [0, n1)
        for k in range(p):
            for i in range(n1):
                W_new[k * n + i, :r1] = W1[k * n1 + i, :]
                
        # Fill Block 2
        # U2 corresponds to A2 (n2 x m)
        # Indices in U2: i in [0, n2), j in [0, m)
        # Indices in U_new: i in [n1, n), j in [0, m)
        for i in range(n2):
            for j in range(m):
                U_new[(n1 + i) * m + j, r1:] = U2[i * m + j, :]
                
        for j in range(m):
            for k in range(p):
                V_new[j * p + k, r1:] = V2[j * p + k, :]
                
        # W2 corresponds to C2 (n2 x p)
        # W2 indices: k in [0, p), i in [0, n2)
        # W_new indices: k in [0, p), i in [n1, n)
        for k in range(p):
            for i in range(n2):
                W_new[k * n + (n1 + i), r1:] = W2[k * n2 + i, :]
                
        return U_new, V_new, W_new

    def solve_block(n, m, p):
        # Base cases
        if n == 1:
            return get_schoolbook(1, m, p)
        if m == 1:
            return get_schoolbook(n, 1, p)
        if p == 1:
            return get_schoolbook(n, m, 1)
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
            
        # Recursive split
        # We can split n, m, or p.
        # Let's try splitting n first.
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = solve_block(n1, m, p)
            U2, V2, W2 = solve_block(n2, m, p)
            return combine_decomps(n1, m, p, U1, V1, W1, n2, m, p, U2, V2, W2, n, m, p)
            
        # If n is 1, split m?
        # If we split m, A is n x m. A = [A1, A2]
        # C = A1 @ B1 + A2 @ B2
        # This is more complex to combine because V and W need to be split too.
        # For now, let's stick to splitting n.
        # If n=1, we use schoolbook.
        
        return get_schoolbook(n, m, p)

    # Strategy:
    # 1. Try a recursive split of 3x4x8 into 1x4x8 and 2x4x8.
    # 2. 1x4x8 is schoolbook (rank 32).
    # 3. 2x4x8 can be split into 2x2x8 and 2x2x8? Or 2x4x4 and 2x4x4?
    #    Let's see what solve_block(2, 4, 8) does.
    #    It splits n=2 into 1 and 1.
    #    solve_block(1, 4, 8) -> schoolbook rank 32.
    #    solve_block(1, 4, 8) -> schoolbook rank 32.
    #    Total rank for 2x4x8 = 64.
    #    Total rank for 3x4x8 = 32 (1x4x8) + 64 (2x4x8) = 96.
    #    This is not better than schoolbook!
    
    # We need a smarter split or better base cases.
    # Strassen 2x2x2 is rank 7 vs 8.
    # 2x4x8:
    #   Split into 2x2x8 and 2x2x8?
    #   2x2x8: Split into 2x2x4 and 2x2x4?
    #   2x2x4: Split into 2x2x2 and 2x2x2?
    #     2x2x2: Rank 7.
    #     2x2x2: Rank 7.
    #     Total 2x2x4: Rank 14.
    #   Total 2x2x8: Rank 28.
    #   Total 2x4x8: Rank 56.
    #   Total 3x4x8: 32 (1x4x8) + 56 (2x4x8) = 88.
    #   This is better than 96!
    
    # Let's implement this specific recursive structure.
    def solve_block_optimized(n, m, p):
        if n == 1:
            return get_schoolbook(1, m, p)
        if m == 1:
            return get_schoolbook(n, 1, p)
        if p == 1:
            return get_schoolbook(n, m, 1)
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
            
        # Split n
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = solve_block_optimized(n1, m, p)
            U2, V2, W2 = solve_block_optimized(n2, m, p)
            return combine_decomps(n1, m, p, U1, V1, W1, n2, m, p, U2, V2, W2, n, m, p)
            
        # If n == 1, we can't split n.
        # We can split m or p.
        # Let's split p if p > 1.
        # A is 1 x m. B is m x p.
        # C = A @ B.
        # Split B into B1 (m x p1) and B2 (m x p2).
        # C1 = A @ B1, C2 = A @ B2.
        # This splits the p dimension.
        
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            # Subproblem 1: 1 x m x p1
            U1, V1, W1 = solve_block_optimized(1, m, p1)
            # Subproblem 2: 1 x m x p2
            U2, V2, W2 = solve_block_optimized(1, m, p2)
            
            # Combine
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            r_total = r1 + r2
            
            U_new = np.zeros((1 * m, r_total))
            V_new = np.zeros((m * p, r_total))
            W_new = np.zeros((p * 1, r_total))
            
            # U is just A.
            for j in range(m):
                U_new[j, :r1] = U1[j, :]
                U_new[j, r1:] = U2[j, :]
                
            # V splits B.
            for j in range(m):
                for k in range(p1):
                    V_new[j * p + k, :r1] = V1[j * p1 + k, :]
                for k in range(p2):
                    V_new[j * p + (p1 + k), r1:] = V2[j * p2 + k, :]
                    
            # W splits C.
            # W shape (p, 1, r).
            for k in range(p1):
                W_new[k, :r1] = W1[k, :]
            for k in range(p2):
                W_new[p1 + k, r1:] = W2[k, :]
                
            return U_new, V_new, W_new
            
        # If p == 1, split m?
        # A is 1 x m. B is m x 1.
        # C is 1 x 1.
        # C = sum_j A_j B_j.
        # This is just a dot product. Rank 1 is minimal?
        # Schoolbook rank is 1*m*1 = m.
        # We can do rank 1:
        # U = [1, 1, ..., 1]
        # V = [1, 1, ..., 1]
        # W = [1]
        # C = 1 * (sum A_j * 1) * (sum B_j * 1) ? No.
        # C = sum_c W_c (sum_j U_j A_j V_j)
        # If U_j=1, V_j=1, W_1=1.
        # C = 1 * sum_j A_j B_j. Correct.
        
        if m > 1:
            U = np.ones((1 * m, 1))
            V = np.ones((m * 1, 1))
            W = np.ones((1 * 1, 1))
            return U, V, W
            
        return get_schoolbook(n, m, p)

    U, V, W = solve_block_optimized(n, m, p)
    
    # Verify
    if verify_decomp(U, V, W, n, m, p):
        # Clean to integers
        U_clean = np.round(U).astype(int)
        V_clean = np.round(V).astype(int)
        W_clean = np.round(W).astype(int)
        
        # Final check
        if verify_decomp(U_clean, V_clean, W_clean, n, m, p):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
            
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = get_schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# Run construct to ensure it works
if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found rank: {rank}")

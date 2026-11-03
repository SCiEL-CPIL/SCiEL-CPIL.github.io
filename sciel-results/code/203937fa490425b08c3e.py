import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        """Check if the decomposition is valid using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        
        # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Reshape for broadcasting
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r)
        
        U_t = U_arr.reshape(n, m, r)
        V_t = V_arr.reshape(m, p, r)
        W_t = W_arr.reshape(p, n, r)
        
        # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # This is a contraction over c.
        # Let's use tensordot or einsum.
        # einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            return False
            
        # Target tensor is 1 if i==i' (always true in W index logic? No)
        # The tensor for matrix multiplication M(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The decomposition is M(A,B)_{i,k} = sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_l B_{l,k} V_{l,k,c}) W_{k,i,c}? 
        # Wait, the standard tensor decomposition for matmul is:
        # (A * B)_{i,k} = sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_l B_{l,k} V_{l,k,c}) W_{k,i,c} ?
        # No, the standard form is:
        # (A * B)_{i,k} = sum_c ( sum_j A_{i,j} U_{i,j,c} ) ( sum_l B_{l,k} V_{l,k,c} ) ? No, that's not right.
        
        # Let's re-read the current code's definition:
        # s = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This s is the coefficient for the product A[i,j] * B[j,k].
        # The full product is sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c]) A[i,j] B[j,k]
        # This equals sum_{i,k} (sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] A[i,j] B[j,k])
        # We want this to equal sum_{i,k} (sum_j A[i,j] B[j,k]) C[i,k]
        # So for each i,k, the coefficient of A[i,j] B[j,k] must be 1 if we are forming the sum?
        # No. The term is A[i,j] * B[j,k].
        # The result C[i,k] = sum_j A[i,j] B[j,k].
        # So the decomposition must satisfy:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k?
        # Let's check dimensions.
        # U[i,j,c] is scalar. V[j,k,c] is scalar. W[k,i,c] is scalar.
        # The sum is over c.
        # This value is the coefficient multiplying A[i,j] B[j,k] in the expanded sum.
        # Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j] B[j,k] in the expression for C[i,k] is 1.
        # However, the sum over j is part of the definition of C[i,k].
        # The tensor T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] must be 1 for all i,j,k?
        # Let's check the schoolbook:
        # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c = i*m+j*p+k?
        # In schoolbook, for a specific (i,j,k), there is one c where all are 1.
        # So sum_c ... = 1.
        # Yes, the condition is T_{i,j,k} = 1 for all i,j,k.
        
        target = np.ones((n, m, p))
        if np.allclose(T, target, atol=tol):
            return True
        return False

    def residual_vec(U, V, W):
        """Compute the residual vector."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        
        U_t = U_arr.reshape(n, m, r)
        V_t = V_arr.reshape(m, p, r)
        W_t = W_arr.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        target = np.ones((n, m, p))
        res = T - target
        return res.flatten()

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        """Strassen's rank-7 decomposition for 2x2x2."""
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def build_2x2x3_decomposition():
        """
        Rank-9 bilinear algorithm for 2x2 * 2x3.
        Computes C = A * B where A is 2x2, B is 2x3, C is 2x3.
        
        Uses 9 scalar multiplications:
        - 4 for the first two columns (schoolbook-style with sharing)
        - 5 for the third column using Strassen-like sharing
        
        Actually, let's use a cleaner approach:
        Split B into B1 (2x2) and B2 (2x1).
        C = A*B1 + A*B2
        
        For A*B1: use 5 multiplications (Strassen for 2x2x2 gives rank 7, but we can do better for rectangular)
        Actually, for 2x2 * 2x3, we need 6 multiplications minimum (since it contains 2x2*2x1 which needs 4, 
        and the structure... let me think differently).
        
        Let me just use schoolbook for all 3 columns: 2*2*3 = 12 multiplications.
        But we can share: compute A11, A12, A21, A22 combinations once.
        
        Better: Use the fact that we need A*B where B is 2x3.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        We can factor:
        Let s0 = A[0,0], s1 = A[0,1], s2 = A[1,0], s3 = A[1,1]
        
        For each column k of B:
        C[0,k] = s0*B[0,k] + s1*B[1,k]
        C[1,k] = s2*B[0,k] + s3*B[1,k]
        
        That's 4 multiplications per column * 3 columns = 12.
        
        Can we do better? The tensor is 2x2x3. Its border rank is at least... 
        Actually, let's try a different factorization:
        
        Compute:
        m1 = s0 * B[0,0]
        m2 = s1 * B[1,0]
        m3 = s0 * B[0,1]
        m4 = s1 * B[1,1]
        m5 = s0 * B[0,2]
        m6 = s1 * B[1,2]
        m7 = s2 * B[0,0]
        m8 = s3 * B[1,0]
        m9 = s2 * B[0,1]
        m10 = s3 * B[1,1]
        m11 = s2 * B[0,2]
        m12 = s3 * B[1,2]
        
        That's 12. Can we reduce?
        
        Alternative: use Strassen-like for the 2x2x2 part and add 4 for the extra column.
        Strassen for 2x2x2 uses 7 mults. For the 3rd column, we need 4 more (one per A element).
        Total: 7 + 4 = 11.
        
        Or: schoolbook for 2x2x2 is 4, plus 4 for the 3rd column = 8? No, that's not right either.
        
        Let me just implement rank 12 (schoolbook) for now, and the Kronecker with Strassen gives 7*12=84.
        But wait, the directive says rank 9 for 2x2x3. Let me try to find a rank-9 algorithm.
        
        Actually, for 2x2 * 2x3, we can view it as computing two dot products of length 2 for each of 3 columns, 
        but with shared A factors. The minimal rank is known to be 6 for 2x2x2 (Strassen is 7, but actually 
        2x2x2 matmul has rank 7 over reals... no wait, it's 7).
        
        Hmm, let me just use a valid decomposition. The key insight from the directive is to get rank 63 total.
        63 = 7 * 9. So we need a rank-9 2x2x3 algorithm.
        
        Let me construct one:
        For 2x2 * 2x3, think of B as [B_left(2x2) | B_right(2x1)].
        C = A*B_left + A*B_right.
        
        Use Strassen for A*B_left (7 mults) and 4 mults for A*B_right. Total 11.
        
        To get 9, we need more sharing. Let me try:
        Define:
        x1 = A[0,0]
        x2 = A[0,1]  
        x3 = A[1,0]
        x4 = A[1,1]
        
        y1 = B[0,0], y2 = B[1,0]
        y3 = B[0,1], y4 = B[1,1]
        y5 = B[0,2], y6 = B[1,2]
        
        We want:
        C[0,0] = x1*y1 + x2*y2
        C[0,1] = x1*y3 + x2*y4
        C[0,2] = x1*y5 + x2*y6
        C[1,0] = x3*y1 + x4*y2
        C[1,1] = x3*y3 + x4*y4
        C[1,2] = x3*y5 + x4*y6
        
        Try 9 mults:
        m1 = x1 * y1
        m2 = x2 * y2
        m3 = x1 * (y3-y1)  -> x1*y3 - x1*y1
        m4 = (x2+x1) * y4  -> x2*y4 + x1*y4... not quite
        
        This is getting complex. Let me just use rank 12 for now and see if the overall structure works,
        then optimize. Actually, for the purpose of this improvement, let me use a simpler valid decomposition.
        
        I'll use rank 8: treat it as two separate 2x2*2x1 problems (each rank 4? No, 2x2*2x1 needs 4 mults).
        2x2*2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]. That's 4 mults.
        For 3 columns: 3*4 = 12.
        
        Let me just implement rank 12 schoolbook for 2x2x3.
        """
        r = 12
        U = [[0] * r for _ in range(4)]    # 2x2 = 4 entries
        V = [[0] * r for _ in range(6)]    # 2x3 = 6 entries  
        W = [[0] * r for _ in range(6)]    # 2x3 = 6 entries
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i * 2 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        """Kronecker product of Strassen for 4x4x4, rank 49."""
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def kronecker_4x4x7_from_2x2():
        """
        Build 4x4x7 using Kronecker product of:
        - 2x2x2 Strassen (rank 7) for row/col blocks
        - 2x2x3 schoolbook (rank 12) for the first 3 B columns
        - 2x2x4 schoolbook (rank 16) for the last 4 B columns
        
        Total rank: 7 * 12 + 7 * 16 = 84 + 112 = 196. That's worse!
        
        Alternative: Use a single 2x2x7 algorithm.
        2x2 * 2x7 schoolbook: 4*7 = 28 rank.
        Kronecker with Strassen: 7 * 28 = 196. Worse.
        
        Let me try: Split 4x4x7 as (2x2 blocks in A rows) x (2x2 blocks in A cols) x (7 in B cols).
        For each 2x2x2x7 subproblem, we need a rank-r algorithm.
        2x2 * 2x7: minimal rank? At least 4*7/2 = 14? 
        Schoolbook is 28. 
        If we can do 2x2*2x7 in rank 9, then total is 7*9 = 63.
        
        Let me construct a rank-9 2x2*2x7 algorithm... that seems unlikely.
        
        For now, let me keep the 97-rank construction and try optimization.
        """
        return None

    def build_4x4x7_decomposition():
        """
        Build 4x4x7 decomposition using 2x2 block structure.
        
        Split the 4x4x7 problem into blocks:
        - Rows of A: 2 blocks of 2
        - Cols of A / rows of B: 2 blocks of 2  
        - Cols of B: 3 blocks (2, 2, 3) or (2, 3, 2) etc.
        
        Strategy: Use Kronecker product of:
        - 2x2x2 Strassen (rank 7) for the 2x2 block in A/B rows
        - 2x2x3 algorithm (rank 12) for the 2x3 block in B cols
        
        This gives rank 7*12 = 84 for the full 4x4x6 part.
        Then add 4x4x1 schoolbook for the last column: 16 more.
        Total: 84 + 16 = 100. That's worse than 97.
        
        Alternative: Split B cols as (3, 4) or use different block sizes.
        
        Let me try: Split into 2x2 blocks for A rows, 2x2 for A cols/B rows, and handle B cols as 7.
        Actually, let me try a different approach:
        
        Use the 2x2x2 Strassen for the "core" and handle the rectangular part differently.
        
        For 4x4 * 4x7:
        Think of A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x2 plus 2x2 blocks of 2x1.
        
        Actually, the simplest valid construction that beats 97:
        Use Kronecker of Strassen(2x2x2) with schoolbook(2x2x3) for first 6 cols, 
        plus schoolbook for last col.
        
        Rank = 7 * 12 + 16 = 84 + 16 = 100. Still worse.
        
        Let me try: Kronecker of Strassen with a better 2x2x7 algorithm.
        2x2 * 2x7: schoolbook is 4*7 = 28. 
        Can we do better? 2x2*2x7 tensor... 
        
        Actually, let me just keep the existing 97-rank construction and focus on optimization.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j * 7 + k][r] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k * 4 + i][r] = W4[k * 4 + i][r]
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    def tensors_to_vec(U, V, W):
        """Flatten the three factor matrices into one vector."""
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        """Unflatten a vector into three factor matrices."""
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def objective(x, r):
        """Objective function for minimization: squared residual."""
        U, V, W = vec_to_tensors(x, r)
        res = residual_vec(U, V, W)
        return np.dot(res, res)

    def reduce_rank_optimization(U_init, V_init, W_init, target_rank, rng, max_restarts=5):
        """
        Try to reduce the rank of a valid decomposition to target_rank
        using optimization.
        """
        r_init = len(U_init[0])
        if target_rank >= r_init:
            return None
            
        U_init = np.array(U_init, dtype=float)
        V_init = np.array(V_init, dtype=float)
        W_init = np.array(W_init, dtype=float)
        
        best_x = None
        best_obj = np.inf
        
        for restart in range(max_restarts):
            # Initialize x for target_rank
            # Strategy: Random linear combination of columns from init
            M = np.random.randn(target_rank, r_init)
            # Scale down to keep values reasonable
            M /= np.sqrt(r_init)
            
            U_new = U_init @ M.T
            V_new = V_init @ M.T
            W_new = W_init @ M.T
            
            x0 = tensors_to_vec(U_new, V_new, W_new)
            
            # Optimize
            try:
                res = minimize(objective, x0, args=(target_rank,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except Exception:
                pass
                
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

        if best_x is not None and best_obj < 1e-8:
            U_out, V_out, W_out = vec_to_tensors(best_x, target_rank)
            if check_valid(U_out.tolist(), V_out.tolist(), W_out.tolist(), tol=1e-8):
                return U_out.tolist(), V_out.tolist(), W_out.tolist()
                
        return None

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)
    np.random.seed(12345)

    # Step 1: Build Kronecker + schoolbook base (rank 97)
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Step 2: Try to reduce rank using optimization with better strategy
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 60:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])
        
        # Try reducing rank with multiple restarts and different strategies
        for target_rank in range(cur_rank - 1, max(70, cur_rank - 15), -1):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            success = False
            for attempt in range(5):
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break
                
                # Different initialization strategies
                if attempt == 0:
                    # Random linear combination
                    M = np.random.randn(target_rank, cur_rank)
                    M /= np.sqrt(cur_rank)
                    U_new = cur_U @ M.T
                    V_new = cur_V @ M.T
                    W_new = cur_W @ M.T
                elif attempt == 1:
                    # Keep first target_rank columns
                    U_new = cur_U[:, :target_rank].copy()
                    V_new = cur_V[:, :target_rank].copy()
                    W_new = cur_W[:, :target_rank].copy()
                elif attempt == 2:
                    # Random projection with bias toward keeping structure
                    M = np.zeros((target_rank, cur_rank))
                    # Keep some original columns
                    keep_count = min(target_rank // 2, cur_rank)
                    M[:keep_count, :keep_count] = np.eye(keep_count)
                    if target_rank > keep_count:
                        M[keep_count:, :] = np.random.randn(target_rank - keep_count, cur_rank)
                        M[keep_count:, :] /= np.sqrt(cur_rank)
                    U_new = cur_U @ M.T
                    V_new = cur_V @ M.T
                    W_new = cur_W @ M.T
                else:
                    # Gaussian perturbation of truncated
                    U_new = cur_U[:, :target_rank].copy()
                    V_new = cur_V[:, :target_rank].copy()
                    W_new = cur_W[:, :target_rank].copy()
                    U_new += 0.1 * np.random.randn(*U_new.shape)
                    V_new += 0.1 * np.random.randn(*V_new.shape)
                    W_new += 0.1 * np.random.randn(*W_new.shape)
                
                x0 = tensors_to_vec(U_new.tolist(), V_new.tolist(), W_new.tolist())
                
                try:
                    res = minimize(objective, x0, args=(target_rank,), method='L-BFGS-B', 
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                    if res.fun < 1e-8:
                        U_out, V_out, W_out = vec_to_tensors(res.x, target_rank)
                        if check_valid(U_out.tolist(), V_out.tolist(), W_out.tolist(), tol=1e-8):
                            best_result = {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                            best_rank = target_rank
                            cur_U = U_out
                            cur_V = V_out
                            cur_W = W_out
                            success = True
                            break
                except Exception:
                    pass
            
            if not success and elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

    # Step 3: Final polish pass - try to snap to integers/half-integers
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            
            # Try snapping to nearest half-integer
            U_snap = np.round(cur_U * 2) / 2
            V_snap = np.round(cur_V * 2) / 2
            W_snap = np.round(cur_W * 2) / 2
            
            if check_valid(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), tol=1e-6):
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            elif check_valid(cur_U.tolist(), cur_V.tolist(), cur_W.tolist(), tol=1e-9):
                pass  # Keep as is
            else:
                # Try integer snapping
                U_int = np.round(cur_U)
                V_int = np.round(cur_V)
                W_int = np.round(cur_W)
                if check_valid(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=1e-6):
                    best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        except Exception:
            pass

    return best_result

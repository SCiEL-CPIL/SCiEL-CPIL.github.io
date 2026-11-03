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

    def build_4x4x7_decomposition():
        """Build 4x4x7 decomposition from Kronecker 4x4x4 + schoolbook for extra columns."""
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

    # Step 2: Try to reduce rank using optimization
    # We can try to reduce from 97 down to, say, 80
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 60:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])
        
        # Try reducing by 1, 2, ... steps
        # Optimization is expensive, so we try a few steps
        for target_rank in range(cur_rank - 1, max(50, cur_rank - 10), -1):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            res = reduce_rank_optimization(cur_U, cur_V, cur_W, target_rank, rng, max_restarts=3)
            if res is not None:
                U_new, V_new, W_new = res
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                cur_U = np.array(U_new, dtype=float)
                cur_V = np.array(V_new, dtype=float)
                cur_W = np.array(W_new, dtype=float)
                # If we succeeded, we can try to reduce further
                # But we need to be careful with time.
                # Let's allow up to 5 successful reductions in this phase
                if cur_rank - target_rank >= 5:
                    break

    # Step 3: Final polish pass
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            # Just re-run check to be sure
            if check_valid(cur_U.tolist(), cur_V.tolist(), cur_W.tolist(), tol=1e-9):
                 # Optional: snap to integers if close?
                 # The task allows half-integers.
                 pass
        except Exception:
            pass

    return best_result

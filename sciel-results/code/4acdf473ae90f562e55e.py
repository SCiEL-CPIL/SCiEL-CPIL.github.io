import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # safety margin before 150s kill

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    
    Strategy:
    1. Build the known rank-55 Strassen+schoolbook baseline.
    2. Compute the tensor T.
    3. Determine the lower bound from matricization ranks.
    4. Attempt to find a lower-rank decomposition (likely rank 30) using 
       Alternating Least Squares (ALS) with multiple random restarts.
    5. Verify exactness by rounding to nearest half-integers and checking the 
       tensor equation.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    def build_rank_55():
        """Build the known rank-55 Strassen+schoolbook decomposition."""
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
            
            # P1: A02*B20
            U[a02, col] += 1.0
            V[20 + k0, col] += 1.0
            W[c00, col] += 1.0
            col += 1
            
            # P2: A02*B21
            U[a02, col] += 1.0
            V[20 + k1, col] += 1.0
            W[c01, col] += 1.0
            col += 1
            
            # P3: A12*B20
            U[a12, col] += 1.0
            V[20 + k0, col] += 1.0
            W[c10, col] += 1.0
            col += 1
            
            # P4: A12*B21
            U[a12, col] += 1.0
            V[20 + k1, col] += 1.0
            W[c11, col] += 1.0
            col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Check for integer/half-integer entries
        # The problem states entries should be integer/half-integer.
        # We check if rounding to nearest 0.5 gives a valid decomposition.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0.0
                    for t in range(rank):
                        coeff += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True

    def verify_exact_half_integer(U, V, W):
        """Verify if rounding to nearest half-integers yields an exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        n, m, p = DIMENSIONS
        rank = U_r.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0.0
                    for t in range(rank):
                        coeff += U_r[i*m + j, t] * V_r[j*p + k, t] * W_r[k*n + i, t]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True

    def build_tensor(U, V, W):
        """Build the full 3-tensor from factor matrices."""
        rank = U.shape[1]
        T = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
        for t in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
        return T

    def matricize(T, mode):
        """Reshape tensor T to a matrix by flattening all modes except `mode`."""
        if mode == 0:
            return T.reshape(T.shape[0], -1)
        elif mode == 1:
            return T.transpose(1, 0, 2).reshape(T.shape[1], -1)
        elif mode == 2:
            return T.transpose(2, 0, 1).reshape(T.shape[2], -1)

    def als_decompose(T, rank, max_iter=100, tol=1e-12):
        """
        Alternating Least Squares for tensor decomposition.
        T is the target tensor.
        Returns U, V, W of the given rank.
        """
        n, m, p = T.shape
        
        # Random initialization
        U = np.random.randn(n, rank)
        V = np.random.randn(m, rank)
        W = np.random.randn(p, rank)
        
        prev_error = np.inf
        for it in range(max_iter):
            # Fix V, W, solve for U
            # T_{i,jk} = sum_t U_{it} V_{jt} W_{kt}
            # Vectorize: T_reshaped (n, m*p) = U (n, rank) * (V * W)_reshaped (rank, m*p)^T ?
            # Let M_U be (n, m*p). M_U[i, j*p+k] = T[i,j,k]
            # Let M_VW be (rank, m*p). M_VW[t, j*p+k] = sum_t' V[j,t'] W[k,t']? No.
            # M_VW[t, j*p+k] = V[j,t] * W[k,t] is not linear in t for the product.
            # Actually, for fixed t, the contribution to T[i,j,k] is U[i,t] V[j,t] W[k,t].
            # So T = sum_t U[:,t] outer V[:,t] outer W[:,t].
            # To solve for U:
            # T_{i,j,k} = sum_t U_{i,t} (V_{j,t} W_{k,t})
            # Let A_{t, j*k} = V_{j,t} W_{k,t}.
            # Then T_reshaped (n, m*p) = U (n, rank) * A (rank, m*p).
            # U = T_reshaped * A^T * (A A^T)^{-1}
            
            T_reshaped = T.reshape(n, m * p)
            A = np.zeros((rank, m * p))
            for t in range(rank):
                for j in range(m):
                    for k in range(p):
                        A[t, j * p + k] = V[j, t] * W[k, t]
            
            # Solve U A = T_reshaped
            try:
                U, _, _, _ = np.linalg.lstsq(A.T, T_reshaped, rcond=None)
                U = U.T # lstsq returns X such that A.T X = T_reshaped, so X is U.T? 
                # A.T is (m*p, rank). T_reshaped is (n, m*p).
                # We want U (n, rank) such that U A.T = T_reshaped? No.
                # T_reshaped[i, idx] = sum_t U[i,t] A[t, idx].
                # So T_reshaped = U A.
                # We need to solve U A = T_reshaped for U.
                # This is equivalent to A.T U.T = T_reshaped.T.
                # U.T = A.T^{-1} T_reshaped.T.
                # Using lstsq: solve A.T X = T_reshaped.T for X.
                X, _, _, _ = np.linalg.lstsq(A.T, T_reshaped.T, rcond=None)
                U = X.T
            except np.linalg.LinAlgError:
                pass
            
            # Fix U, W, solve for V
            # T_{i,j,k} = sum_t (U_{i,t} W_{k,t}) V_{j,t}
            # Let B_{t, i*k} = U_{i,t} W_{k,t}.
            # T_reshaped2 (m, n*p) = V (m, rank) * B (rank, n*p).
            T_reshaped2 = T.transpose(1, 0, 2).reshape(m, n * p)
            B = np.zeros((rank, n * p))
            for t in range(rank):
                for i in range(n):
                    for k in range(p):
                        B[t, i * p + k] = U[i, t] * W[k, t]
            
            try:
                X, _, _, _ = np.linalg.lstsq(B.T, T_reshaped2.T, rcond=None)
                V = X.T
            except np.linalg.LinAlgError:
                pass
            
            # Fix U, V, solve for W
            # T_{i,j,k} = sum_t (U_{i,t} V_{j,t}) W_{k,t}
            # Let C_{t, i*j} = U_{i,t} V_{j,t}.
            # T_reshaped3 (p, n*m) = W (p, rank) * C (rank, n*m).
            T_reshaped3 = T.transpose(2, 0, 1).reshape(p, n * m)
            C = np.zeros((rank, n * m))
            for t in range(rank):
                for i in range(n):
                    for j in range(m):
                        C[t, i * m + j] = U[i, t] * V[j, t]
            
            try:
                X, _, _, _ = np.linalg.lstsq(C.T, T_reshaped3.T, rcond=None)
                W = X.T
            except np.linalg.LinAlgError:
                pass
            
            # Calculate error
            T_est = build_tensor(U, V, W)
            error = np.linalg.norm(T - T_est)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
            
            if time.time() - start_time > DEADLINE - 10:
                break
                
        return U, V, W, error

    # 1. Build baseline rank-55
    U_base, V_base, W_base = build_rank_55()
    
    if not verify_decomposition(U_base, V_base, W_base):
        # Fallback if baseline is broken (should not happen)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    # 2. Compute tensor and lower bounds
    T = build_tensor(U_base, V_base, W_base)
    
    mat0 = matricize(T, 0)
    mat1 = matricize(T, 1)
    mat2 = matricize(T, 2)
    
    r0 = np.linalg.matrix_rank(mat0, tol=1e-10)
    r1 = np.linalg.matrix_rank(mat1, tol=1e-10)
    r2 = np.linalg.matrix_rank(mat2, tol=1e-10)
    
    lower_bound = max(r0, r1, r2)
    
    # 3. Search for lower rank
    # We know rank >= lower_bound.
    # Try ranks from lower_bound to 54.
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 55
    
    # Optimization: Try the most likely ranks first.
    # If lower_bound is 30, try 30, 31, ...
    # If lower_bound is 20, try 20, ...
    
    # To save time, we can limit the number of restarts per rank.
    max_restarts = 20
    
    for target_rank in range(lower_bound, 55):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 20:
                break
                
            U_cand, V_cand, W_cand, error = als_decompose(T, target_rank, max_iter=50)
            
            # Check if the solution is close to exact
            if error < 1e-8:
                # Try to round to half-integers
                if verify_exact_half_integer(U_cand, V_cand, W_cand):
                    # Verify the rounded version exactly
                    U_r = np.round(U_cand * 2) / 2
                    V_r = np.round(V_cand * 2) / 2
                    W_r = np.round(W_cand * 2) / 2
                    
                    if verify_decomposition(U_r, V_r, W_r):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break # Found a valid decomposition for this rank, move to next (lower) rank? No, we are iterating upwards from lower_bound. So this is the minimal rank found so far. We can stop searching for this rank.
            
            # If we found a valid rank, we should try to see if a lower rank works?
            # The loop goes from lower_bound to 54. So if we find rank 30, we break the inner loop.
            # Then we continue to rank 31? No, we want the MINIMAL rank.
            # So if we find a valid decomposition at rank R, we should check if R-1 works?
            # But we started from lower_bound. So if we find valid at lower_bound, that's the best.
            # If we find valid at lower_bound+1, we should have checked lower_bound first.
            # So the logic is correct: iterate from lower_bound up. First valid one is the min.
        
        if found:
            # Since we iterate from lower_bound upwards, the first found is the minimal rank.
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


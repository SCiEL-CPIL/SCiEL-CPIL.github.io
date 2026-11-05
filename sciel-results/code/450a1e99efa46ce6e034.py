"""Rewritten program: algebraic Strassen-style construction for 3x4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    # Build target tensor T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    def verify(U, V, W, rank):
        """Verify that U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)
    
    # Schoolbook solution (rank 72) as guaranteed fallback
    def schoolbook():
        rank = n * m * p  # 72
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1.0
                    V[j, k, col] = 1.0
                    W[k, i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Strassen-style construction for 3x4x6
    # Decompose A (3x4) into blocks: A11 (3x3), A12 (3x1)
    # Decompose B (4x6) into blocks: B11 (3x3), B12 (3x3), B21 (1x3), B22 (1x3)
    # C = A*B has blocks: C11 (3x3) = A11*B11 + A12*B21
    #                          C12 (3x3) = A11*B12 + A12*B22
    
    # Strategy: Use rank-23 decomposition for each 3x3x3 multiplication
    # C11 = A11*B11 + A12*B21: needs rank(3x3x3) + rank(3x1x3) = 23 + 9 = 32
    # C12 = A11*B12 + A12*B22: needs rank(3x3x3) + rank(3x1x3) = 23 + 9 = 32
    # Total: 64
    
    # First, find a rank-23 decomposition for 3x3x3
    def find_3x3x3_rank23():
        """Find rank-23 decomposition for 3x3x3 matrix multiplication using algebraic structure"""
        n3, m3, p3 = 3, 3, 3
        rank3 = 23
        T3 = np.ones((n3, m3, p3))
        T3_flat = T3.flatten()
        
        # Use known Strassen-like structure for 3x3x3
        # The multiplication tensor for 3x3x3 has rank 23
        # We construct it using the known algebraic decomposition
        
        # Initialize with schoolbook rank 27 and try to reduce
        U = np.zeros((n3, m3, rank3))
        V = np.zeros((m3, p3, rank3))
        W = np.zeros((p3, n3, rank3))
        
        # Start with a structured initialization based on Strassen's ideas
        # For 3x3x3, we can use the following approach:
        # Use 4x4x4 Strassen (rank 7) embedded in 3x3x3 with modifications
        
        # Actually, let's use a direct algebraic construction
        # The 3x3x3 multiplication tensor can be decomposed as:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # We need 23 rank-1 terms
        
        # Use known result: rank of 3x3x3 is 23
        # Construct using block structure
        
        # Initialize with random values and use alternating projection
        np.random.seed(42)
        
        def snap_to_halfint(X):
            return np.round(X * 2.0) / 2.0
        
        def snap_to_int(X):
            return np.round(X)
        
        def alternating_proj_333(U, V, W, max_iters=500, seed=0):
            """Alternating projection for 3x3x3 rank 23"""
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Check current error
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T_rec - T3
                current_err = np.sum(err**2)
                
                if current_err < 1e-10:
                    # Try snapping to half-integers
                    U_s = snap_to_halfint(U)
                    V_s = snap_to_halfint(V)
                    W_s = snap_to_halfint(W)
                    T_rec2 = np.einsum('ijc,jkc,kic->ijk', U_s, V_s, W_s)
                    if np.allclose(T_rec2.flatten(), T3_flat, atol=1e-8):
                        return U_s, V_s, W_s
                    
                    # Try integer snapping
                    U_i = snap_to_int(U)
                    V_i = snap_to_int(V)
                    W_i = snap_to_int(W)
                    T_rec3 = np.einsum('ijc,jkc,kic->ijk', U_i, V_i, W_i)
                    if np.allclose(T_rec3.flatten(), T3_flat, atol=1e-8):
                        return U_i, V_i, W_i
                    
                    return None
                
                # Solve for U
                for i in range(n3):
                    for j in range(m3):
                        A = np.einsum('kc,kc->ck', V[j], W[:, i])
                        b = T3[i, j, :]
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            U[i, j, :] = u_sol
                        except:
                            pass
                
                # Solve for V
                for j in range(m3):
                    for k in range(p3):
                        A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                        b = T3[:, j, k]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            V[j, k, :] = v_sol
                        except:
                            pass
                
                # Solve for W
                for k in range(p3):
                    for i in range(n3):
                        A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                        b = T3[i, :, k]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            W[k, i, :] = w_sol
                        except:
                            pass
                
                # Snap to half-integers
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
            
            return None
        
        # Try multiple random initializations
        for seed in range(100):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(seed)
            U = np.random.randn(n3, m3, rank3) * 0.5
            V = np.random.randn(m3, p3, rank3) * 0.5
            W = np.random.randn(p3, n3, rank3) * 0.5
            U = snap_to_halfint(U)
            V = snap_to_halfint(V)
            W = snap_to_halfint(W)
            
            result = alternating_proj_333(U, V, W, max_iters=100, seed=seed)
            if result:
                U3, V3, W3 = result
                # Verify
                T_rec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                if np.allclose(T_rec.flatten(), T3_flat, atol=1e-8):
                    return U3, V3, W3
        
        return None
    
    # Build rank-64 solution for 3x4x6 from 3x3x3 rank-23 factors
    def build_rank64(U3, V3, W3):
        """Build rank-64 solution for 3x4x6 using 3x3x3 rank-23 factors"""
        rank_total = 64
        U = np.zeros((n, m, rank_total))
        V = np.zeros((m, p, rank_total))
        W = np.zeros((p, n, rank_total))
        
        # Block 1: A_left (3x3) * B_top_left (3x3) -> C_top_left (3x3)
        # Cols 0-22
        for c in range(23):
            U[0:3, 0:3, c] = U3[:, :, c]
            V[0:3, 0:3, c] = V3[:, :, c]
            W[0:3, 0:3, c] = W3[:, :, c]
        
        # Block 2: A_left (3x3) * B_top_right (3x3) -> C_top_right (3x3)
        # Cols 23-45
        for c in range(23):
            U[0:3, 0:3, 23+c] = U3[:, :, c]
            V[0:3, 3:6, 23+c] = V3[:, :, c]
            W[3:6, 0:3, 23+c] = W3[:, :, c]
        
        # Block 3: A_right (3x1) * B_bottom_left (1x3) -> C_bottom_left (3x3)
        # Cols 46-57 (12 cols, but we need 9 for 3x1x3)
        # Actually 3x1x3 has rank 9 (3*1*3)
        col_offset = 46
        for i in range(3):
            for k in range(3):
                c = col_offset + i*3 + k
                U[i, 3, c] = 1.0
                V[3, k, c] = 1.0
                W[k, i, c] = 1.0
        
        # Block 4: A_right (3x1) * B_bottom_right (1x3) -> C_bottom_right (3x3)
        # Cols 58-63 (6 cols, but we need 9 for 3x1x3)
        # Wait, 3x1x3 has rank 9, so we need 9 cols
        # Total: 23 + 23 + 9 + 9 = 64
        col_offset = 55
        for i in range(3):
            for k in range(3):
                c = col_offset + i*3 + k
                U[i, 3, c] = 1.0
                V[3, 3+k, c] = 1.0
                W[3+k, i, c] = 1.0
        
        return U, V, W, rank_total
    
    # Try to find 3x3x3 rank-23 decomposition
    U3 = V3 = W3 = None
    if time.time() < deadline - 20:
        res333 = find_3x3x3_rank23()
        if res333:
            U3, V3, W3 = res333
    
    if U3 is not None:
        U64, V64, W64, rank64 = build_rank64(U3, V3, W3)
        if verify(U64, V64, W64, rank64):
            return {
                "U": U64.reshape(NM, rank64).tolist(),
                "V": V64.reshape(MP, rank64).tolist(),
                "W": W64.reshape(PN, rank64).tolist()
            }
    
    # Fallback: schoolbook rank 72
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    return {
        "U": U_sb.reshape(NM, rank_sb).tolist(),
        "V": V_sb.reshape(MP, rank_sb).tolist(),
        "W": W_sb.reshape(PN, rank_sb).tolist()
    }

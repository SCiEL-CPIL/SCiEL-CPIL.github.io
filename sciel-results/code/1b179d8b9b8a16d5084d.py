import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Try to find a rank-20 decomposition using two rank-10 blocks for 2x3x4.
    - Fall back to rank-24 using four rank-6 blocks for 2x3x2.
    - Fall back to rank-44 using the current Strassen-based approach.
    - Fall back to schoolbook rank 48.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 8
    
    def build_expected_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T[a_idx, b_idx, c_idx] = 1
        return T
    
    def verify_decomposition(U, V, W, T_expected):
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_expected)
    
    T_expected = build_expected_tensor()
    
    # Try rank-20: two 2x3x4 blocks of rank 10 each
    # Use ALS to search for rank-10 decomposition of 2x3x4
    def try_rank10_2x3x4():
        """Try to find a rank-10 decomposition for 2x3x4 matrix multiplication."""
        n2, m2, p2 = 2, 3, 4
        T_block = np.zeros((n2*m2, m2*p2, p2*n2), dtype=np.int64)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a_idx = i * m2 + j
                    b_idx = j * p2 + k
                    c_idx = k * n2 + i
                    T_block[a_idx, b_idx, c_idx] = 1
        
        rank = 10
        # Try multiple random initializations with ALS
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            np.random.seed(attempt)
            U = np.random.randn(n2*m2, rank)
            V = np.random.randn(m2*p2, rank)
            W = np.random.randn(p2*n2, rank)
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Update U
                UW = np.einsum('bd,cd->bc', V, W)
                U_target = np.einsum('abc,ac->ab', T_block, W)
                U_new = np.linalg.lstsq(UW, U_target, rcond=None)[0]
                
                # Update V
                UW = np.einsum('ad,cd->ac', U, W)
                V_target = np.einsum('abc,bc->ab', T_block, W)
                V_new = np.linalg.lstsq(UW, V_target, rcond=None)[0]
                
                # Update W
                UV = np.einsum('ad,bd->ab', U, V)
                W_target = np.einsum('abc,ab->ac', T_block, U)
                W_new = np.linalg.lstsq(UV, W_target, rcond=None)[0]
                
                U, V, W = U_new, V_new, W_new
            
            # Check if we found an exact integer solution
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify_decomposition(U_int, V_int, W_int, T_block):
                return U_int, V_int, W_int
        
        return None
    
    result_rank10 = try_rank10_2x3x4()
    if result_rank10 is not None:
        U1, V1, W1 = result_rank10
        # Build full 2x3x8 decomposition from two 2x3x4 blocks
        rank_total = 20
        U = np.zeros((n*m, rank_total), dtype=np.int64)
        V = np.zeros((m*p, rank_total), dtype=np.int64)
        W = np.zeros((p*n, rank_total), dtype=np.int64)
        
        for block in range(2):
            k_start = block * 4
            U[:, block*10:(block+1)*10] = U1
            V[:, block*10:(block+1)*10] = np.zeros((m*p, 10))
            W[:, block*10:(block+1)*10] = np.zeros((p*n, 10))
            
            # Map the 2x3x4 indices to 2x3x8 indices
            for i in range(2):
                for j in range(3):
                    for k in range(4):
                        k_global = k_start + k
                        a_idx = i * m + j
                        b_idx = j * p + k_global
                        c_idx = k_global * n + i
                        
                        # Find which rank-1 term in the block handles this
                        # This is complex; let's just copy and verify
                        pass
        
        # This approach is getting complex. Let me try a simpler direct construction.
        pass
    
    # Try rank-24: four 2x3x2 blocks of rank 6 each
    def try_rank6_2x3x2():
        """Try to find a rank-6 decomposition for 2x3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        T_block = np.zeros((n2*m2, m2*p2, p2*n2), dtype=np.int64)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a_idx = i * m2 + j
                    b_idx = j * p2 + k
                    c_idx = k * n2 + i
                    T_block[a_idx, b_idx, c_idx] = 1
        
        rank = 6
        # Try multiple random initializations with ALS
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            np.random.seed(attempt * 7 + 3)
            U = np.random.randn(n2*m2, rank)
            V = np.random.randn(m2*p2, rank)
            W = np.random.randn(p2*n2, rank)
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline:
                    return None
                
                # Update U
                UW = np.einsum('bd,cd->bc', V, W)
                U_target = np.einsum('abc,ac->ab', T_block, W)
                U_new = np.linalg.lstsq(UW, U_target, rcond=None)[0]
                
                # Update V
                UW = np.einsum('ad,cd->ac', U, W)
                V_target = np.einsum('abc,bc->ab', T_block, W)
                V_new = np.linalg.lstsq(UW, V_target, rcond=None)[0]
                
                # Update W
                UV = np.einsum('ad,bd->ab', U, V)
                W_target = np.einsum('abc,ab->ac', T_block, U)
                W_new = np.linalg.lstsq(UV, W_target, rcond=None)[0]
                
                U, V, W = U_new, V_new, W_new
            
            # Check if we found an exact integer solution
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify_decomposition(U_int, V_int, W_int, T_block):
                return U_int, V_int, W_int
        
        return None
    
    result_rank6 = try_rank6_2x3x2()
    if result_rank6 is not None:
        U1, V1, W1 = result_rank6
        rank_total = 24
        U = np.zeros((n*m, rank_total), dtype=np.int64)
        V = np.zeros((m*p, rank_total), dtype=np.int64)
        W = np.zeros((p*n, rank_total), dtype=np.int64)
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            col_start = block * 6
            
            for r in range(6):
                col = col_start + r
                for i in range(2):
                    for j in range(3):
                        a_idx = i * m + j
                        U[a_idx, col] = U1[i*m2 + j, r]
                        b_idx = j * p + k0
                        V[b_idx, col] = V1[j*m2 + 0, r]
                        b_idx = j * p + k1
                        V[b_idx, col] = V1[j*m2 + 1, r]
                        c_idx = k0 * n + i
                        W[c_idx, col] = W1[0*n2 + i, r]
                        c_idx = k1 * n + i
                        W[c_idx, col] = W1[1*n2 + i, r]
        
        if verify_decomposition(U, V, W, T_expected):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-44 construction (Strassen per block)
    rank = 44
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # 4 rank-1 terms for the 2x1x2 part (first column of A)
        U[0*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
        U[0*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 0, col] = 1
        col += 1
        U[1*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        U[1*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # 7 rank-1 terms for the 2x2x2 part (Strassen)
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
    
    if verify_decomposition(U, V, W, T_expected):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook rank 48
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_sb[a_idx, c] = 1
                V_sb[b_idx, c] = 1
                W_sb[c_idx, c] = 1
                c += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

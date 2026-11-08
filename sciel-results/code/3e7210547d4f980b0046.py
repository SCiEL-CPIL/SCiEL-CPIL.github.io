import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    def find_224_decomposition(max_time=30):
        """Find a rank-12 decomposition for 2x2x4 matmul."""
        # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3}
        # Find U (4x12), V (8x12), W (8x12) such that
        # sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
        
        rank = 12
        U_size = 4  # i*2+j
        V_size = 8  # j*4+k
        W_size = 8  # k*2+i
        
        # Build the constraint matrix: for each (i,j,k), we have a constraint
        # sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
        
        # Alternating minimization
        np.random.seed(42)
        
        best_error = float('inf')
        best_UVW = None
        
        for attempt in range(50):
            if time.time() > deadline - max_time:
                break
                
            # Initialize
            U = np.random.randint(0, 2, size=(U_size, rank)) * 2 - 1  # -1 or 1
            V = np.random.randint(0, 2, size=(V_size, rank)) * 2 - 1
            W = np.random.randint(0, 2, size=(W_size, rank)) * 2 - 1
            
            for iteration in range(200):
                if time.time() > deadline - max_time:
                    break
                    
                # Fix U, V, solve for W
                # For each (i,j,k): sum_r U[i*2+j,r] * V[j*4+k,r] * W[k*2+i,r] = 1
                # For fixed U,V: for each k,i: sum_j sum_r U[i*2+j,r] * V[j*4+k,r] * W[k*2+i,r] = 1
                # This gives 8 equations (k in 0..3, i in 0..1) for 8 unknowns (W[0..7])
                # Wait, W has 8*12=96 entries, but each equation only involves 4 of them (one per r)
                
                # Actually, for each (i,j,k) constraint:
                # sum_r (U[i*2+j, r] * V[j*4+k, r]) * W[k*2+i, r] = 1
                # Let A_r = U[i*2+j, r] * V[j*4+k, r]
                # Then sum_r A_r * W[k*2+i, r] = 1
                # But W[k*2+i, r] is a single entry, not a vector.
                
                # Hmm, this is not a linear system in the usual sense because each equation
                # involves different entries of W.
                
                # Let me think differently. W is 8x12. Entry W[k*2+i, r] appears in the
                # constraint for (i,j,k) with coefficient U[i*2+j, r] * V[j*4+k, r].
                
                # For a fixed W column r, W[k*2+i, r] appears in constraints for all j:
                # sum_j U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
                # W[k*2+i, r] * sum_j U[i*2+j, r] * V[j*4+k, r] = 1
                # W[k*2+i, r] = 1 / sum_j U[i*2+j, r] * V[j*4+k, r]
                
                # So for each (k, i, r): W[k*2+i, r] = 1 / sum_j U[i*2+j, r] * V[j*4+k, r]
                
                # Similarly: U[i*2+j, r] = 1 / sum_k V[j*4+k, r] * W[k*2+i, r]
                # And: V[j*4+k, r] = 1 / sum_i U[i*2+j, r] * W[k*2+i, r]
                
                # This is a fixed-point iteration!
                
                # Update W
                for k in range(4):
                    for i in range(2):
                        for r in range(rank):
                            s = sum(U[i*2+j, r] * V[j*4+k, r] for j in range(2))
                            if abs(s) > 1e-10:
                                W[k*2+i, r] = 1.0 / s
                            else:
                                W[k*2+i, r] = 0.0
                
                # Update U
                for i in range(2):
                    for j in range(2):
                        for r in range(rank):
                            s = sum(V[j*4+k, r] * W[k*2+i, r] for k in range(4))
                            if abs(s) > 1e-10:
                                U[i*2+j, r] = 1.0 / s
                            else:
                                U[i*2+j, r] = 0.0
                
                # Update V
                for j in range(2):
                    for k in range(4):
                        for r in range(rank):
                            s = sum(U[i*2+j, r] * W[k*2+i, r] for i in range(2))
                            if abs(s) > 1e-10:
                                V[j*4+k, r] = 1.0 / s
                            else:
                                V[j*4+k, r] = 0.0
                
                # Check error
                err = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            val = sum(U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] for r in range(rank))
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_UVW = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-12:
                    break
            
            if best_error < 1e-12:
                break
        
        if best_UVW is None or best_error > 1e-8:
            return None
        
        U, V, W = best_UVW
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded version is exact
        err = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    val = sum(U_rounded[i*2+j, r] * V_rounded[j*4+k, r] * W_rounded[k*2+i, r] for r in range(rank))
                    err += (val - 1.0) ** 2
        
        if err < 1e-10:
            return (U_rounded, V_rounded, W_rounded)
        
        # Try rounding to integers only
        U_rounded = np.round(U)
        V_rounded = np.round(V)
        W_rounded = np.round(W)
        
        err = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    val = sum(U_rounded[i*2+j, r] * V_rounded[j*4+k, r] * W_rounded[k*2+i, r] for r in range(rank))
                    err += (val - 1.0) ** 2
        
        if err < 1e-10:
            return (U_rounded, V_rounded, W_rounded)
        
        return None

    # Try to find rank-12 decomposition for 2x2x4
    decomp_224 = find_224_decomposition(max_time=30)
    
    if decomp_224 is not None:
        U_224, V_224, W_224 = decomp_224
        rank = 48
        U = [[0] * rank for _ in range(n * m)]  # 8 x 48
        V = [[0] * rank for _ in range(m * p)]  # 32 x 48
        W = [[0] * rank for _ in range(p * n)]  # 16 x 48
        
        def add_224_block(base_rank, i_offset, j_offset, k_offset):
            for r in range(12):
                current_rank = base_rank + r
                for li in range(2):
                    for lj in range(2):
                        val = U_224[li*2+lj, r]
                        if val != 0:
                            U[(li + i_offset) * 4 + (lj + j_offset)][current_rank] += val
                for lj in range(2):
                    for lk in range(4):
                        val = V_224[lj*4+lk, r]
                        if val != 0:
                            V[(lj + j_offset) * 8 + (lk + k_offset)][current_rank] += val
                for lk in range(4):
                    for li in range(2):
                        val = W_224[lk*2+li, r]
                        if val != 0:
                            W[(lk + k_offset) * 2 + (li + i_offset)][current_rank] += val
        
        add_224_block(0, 0, 0, 0)
        add_224_block(12, 0, 0, 4)
        add_224_block(24, 0, 2, 0)
        add_224_block(36, 0, 2, 4)
        
        # Verify
        def verify_full():
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        total = 0
                        for r in range(rank):
                            u_idx = i * 4 + j
                            v_idx = j * 8 + k
                            w_idx = k * 2 + i
                            total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                        if abs(total - 1) > 1e-10:
                            return False
            return True
        
        if verify_full():
            return {"U": U, "V": V, "W": W}
    
    # Fallback: Strassen-based rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                U[(li + i_off) * 4 + (lj + j_off)][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                W[(lk + k_off) * 2 + (li + i_off)][cr] += val
    
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(7, 0, 0, 2)
    add_strassen_block(14, 0, 0, 4)
    add_strassen_block(21, 0, 0, 6)
    add_strassen_block(28, 0, 2, 0)
    add_strassen_block(35, 0, 2, 2)
    add_strassen_block(42, 0, 2, 4)
    add_strassen_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}

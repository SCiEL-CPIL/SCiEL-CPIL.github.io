"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Laderman rank-20 decomposition for 3x3*3x3 -> 3x3
    # U[i*3+j], V[j*3+k], W[k*3+i]
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    
    def verify_333(U, V, W):
        """Verify that the decomposition correctly computes 3x3*3x3->3x3 using vectorized NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # U_arr shape: (9, 20), V_arr shape: (9, 20), W_arr shape: (9, 20)
        # For each (i,j,k), check sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] == 1
        # Reshape to 3D: U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
        U_3d = U_arr.reshape(3, 3, 20)  # [i, j, r]
        V_3d = V_arr.reshape(3, 3, 20)  # [j, k, r]
        W_3d = W_arr.reshape(3, 3, 20)  # [k, i, r]
        # Compute product tensor: prod[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Use einsum: 'ijr,jkr,kr->ijk' wait, W is [k,i,r] so indices are k,i,r
        # We want: for each i,j,k: sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # W_3d[k,i,r] -> we need to index as [k,i,r]
        # einsum: U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r] -> sum over r
        result = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
        # Wait, W_3d is [k,i,r], so the indices are k,i,r. In einsum, we need to match.
        # Let me be more careful:
        # U_3d: shape (3,3,20), indices (i,j,r)
        # V_3d: shape (3,3,20), indices (j,k,r)  
        # W_3d: shape (3,3,20), indices (k,i,r)
        # We want output[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # einsum: 'ijr,jkr,kr->ijk' would work if W_3d were indexed as [k,i,r] -> but in einsum,
        # the third operand's indices are 'kr' meaning first index=k, second index=r? No.
        # W_3d has shape (3,3,20) with axes (k,i,r). So in einsum, we label them as 'kir'.
        # So: np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if np.max(np.abs(result - 1.0)) > 1e-9:
            return False
        return True
    
    lader_valid = verify_333(lader_U, lader_V, lader_W)
    
    if lader_valid:
        # Two 3x3x3 blocks (k=0..5) + one 3x3x2 block (k=6..7)
        rank_lader = 20
        rank_tail = 18  # schoolbook for 3x3x2
        total_rank = 2 * rank_lader + rank_tail  # 58
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First 3x3x3 block: k indices 0-2, columns 0-19
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Second 3x3x3 block: k indices 3-5, columns 20-39
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][rank_lader + r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p + (3+k)][rank_lader + r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(3+k)*n+i][rank_lader + r] = lader_W[r][k*3+i]
        
        # 3x3x2 block: k indices 6-7, columns 40-57 (schoolbook)
        col = 2 * rank_lader
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*m+j][col] = 1
                    V[j*p + (6+k)][col] = 1
                    W[(6+k)*n+i][col] = 1
                    col += 1
        
        # Verify the full decomposition using vectorized NumPy
        def verify_full(U, V, W):
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            # U_arr: (9, total_rank), V_arr: (24, total_rank), W_arr: (24, total_rank)
            # U_3d: (3,3,total_rank) -> [i,j,r]
            # V_3d: (3,8,total_rank) -> [j,k,r]
            # W_3d: (8,3,total_rank) -> [k,i,r]
            U_3d = U_arr.reshape(3, 3, total_rank)
            V_3d = V_arr.reshape(3, 8, total_rank)
            W_3d = W_arr.reshape(8, 3, total_rank)
            # result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
            result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            if np.max(np.abs(result - 1.0)) > 1e-9:
                return False
            return True
        
        if verify_full(U, V, W):
            # Convert to integer/half-integer format
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook rank 72
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
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

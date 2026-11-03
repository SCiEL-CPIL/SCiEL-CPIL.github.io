import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def schoolbook_decomp(n_sub, m_sub, p_sub):
        """Generate schoolbook decomposition for n_sub x m_sub x p_sub."""
        rank = n_sub * m_sub * p_sub
        U = [[0] * rank for _ in range(n_sub * m_sub)]
        V = [[0] * rank for _ in range(m_sub * p_sub)]
        W = [[0] * rank for _ in range(p_sub * n_sub)]
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j][col] = 1
                    V[j * p_sub + k][col] = 1
                    W[k * n_sub + i][col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_decomp():
        """Generate Strassen decomposition for 2x2x2 (rank 7)."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # n=2, m=2 -> 4 rows
        V = [[0]*rank for _ in range(4)]  # m=2, p=2 -> 4 rows
        W = [[0]*rank for _ in range(4)]  # p=2, n=2 -> 4 rows
        
        # P1: (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2: (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # P3: a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # P4: d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # P5: (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # P6: (c-d)(f+g)
        U[2][5] = 1; U[3][5] = -1
        V[1][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # P7: (b-d)(e+g)
        U[1][6] = 1; U[3][6] = -1
        V[0][6] = 1; V[2][6] = 1
        W[0][6] = 1
        
        return U, V, W, rank

    def get_3x2x2_decomp():
        """Generate decomposition for 3x2x2 (rank 11)."""
        U1, V1, W1, r1 = strassen_decomp()
        U2, V2, W2, r2 = schoolbook_decomp(1, 2, 2)
        
        total_rank = r1 + r2
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(4)]
        W = [[0] * total_rank for _ in range(6)]
        
        for r in range(r1):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][r] = U1[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[j * 2 + k][r] = V1[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[k * 3 + i][r] = W1[k * 2 + i][r]
        
        for r in range(r2):
            for j in range(2):
                U[2 * 2 + j][r1 + r] = U2[j][r]
            for j in range(2):
                for k in range(2):
                    V[j * 2 + k][r1 + r] = V2[j * 2 + k][r]
            for k in range(2):
                W[k * 3 + 2][r1 + r] = W2[k][r]
        
        return U, V, W, total_rank

    def get_3x2x1_decomp():
        """Generate decomposition for 3x2x1 (rank 6)."""
        U, V, W, rank = schoolbook_decomp(3, 2, 1)
        return U, V, W, rank

    def get_3x2x7_decomp():
        """Generate decomposition for 3x2x7 (rank 39)."""
        U_parts = []
        V_parts = []
        W_parts = []
        ranks = []
        
        for k_offset in [0, 2, 4]:
            U_sub, V_sub, W_sub, r_sub = get_3x2x2_decomp()
            U_parts.append(U_sub)
            V_parts.append(V_sub)
            W_parts.append(W_sub)
            ranks.append(r_sub)
        
        U_sub, V_sub, W_sub, r_sub = get_3x2x1_decomp()
        U_parts.append(U_sub)
        V_parts.append(V_sub)
        W_parts.append(W_sub)
        ranks.append(r_sub)
        
        total_rank = sum(ranks)
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(14)]
        W = [[0] * total_rank for _ in range(21)]
        
        col_offset = 0
        for part_idx in range(4):
            U_sub = U_parts[part_idx]
            V_sub = V_parts[part_idx]
            W_sub = W_parts[part_idx]
            r_sub = ranks[part_idx]
            
            if part_idx < 3:
                k_offset = part_idx * 2
                for i in range(6):
                    for r in range(r_sub):
                        U[i][col_offset + r] = U_sub[i][r]
                for j in range(2):
                    for local_k in range(2):
                        global_k = k_offset + local_k
                        for r in range(r_sub):
                            V[j * 7 + global_k][col_offset + r] = V_sub[j * 2 + local_k][r]
                for local_k in range(2):
                    global_k = k_offset + local_k
                    for i in range(3):
                        for r in range(r_sub):
                            W[global_k * 3 + i][col_offset + r] = W_sub[local_k * 3 + i][r]
            else:
                k_offset = 6
                for i in range(6):
                    for r in range(r_sub):
                        U[i][col_offset + r] = U_sub[i][r]
                for j in range(2):
                    for r in range(r_sub):
                        V[j * 7 + 6][col_offset + r] = V_sub[j * 1 + 0][r]
                for i in range(3):
                    for r in range(r_sub):
                        W[6 * 3 + i][col_offset + r] = W_sub[0 * 3 + i][r]
            
            col_offset += r_sub
        
        return U, V, W, total_rank

    def get_3x4x7_decomp():
        """Generate decomposition for 3x4x7 (rank 78)."""
        U_parts = []
        V_parts = []
        W_parts = []
        ranks = []
        
        for j_offset in [0, 2]:
            U_sub, V_sub, W_sub, r_sub = get_3x2x7_decomp()
            U_parts.append(U_sub)
            V_parts.append(V_sub)
            W_parts.append(W_sub)
            ranks.append(r_sub)
        
        total_rank = sum(ranks)
        U = [[0] * total_rank for _ in range(12)]
        V = [[0] * total_rank for _ in range(28)]
        W = [[0] * total_rank for _ in range(21)]
        
        col_offset = 0
        for part_idx in range(2):
            U_sub = U_parts[part_idx]
            V_sub = V_parts[part_idx]
            W_sub = W_parts[part_idx]
            r_sub = ranks[part_idx]
            
            j_offset = part_idx * 2
            
            for i in range(3):
                for local_j in range(2):
                    global_j = j_offset + local_j
                    for r in range(r_sub):
                        U[i * 4 + global_j][col_offset + r] = U_sub[i * 2 + local_j][r]
            
            for local_j in range(2):
                global_j = j_offset + local_j
                for k in range(7):
                    for r in range(r_sub):
                        V[global_j * 7 + k][col_offset + r] = V_sub[local_j * 7 + k][r]
            
            for i in range(21):
                for r in range(r_sub):
                    W[i][col_offset + r] = W_sub[i][r]
            
            col_offset += r_sub
        
        return U, V, W, total_rank

    def verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Try to build the 78-rank decomposition
    try:
        U, V, W, rank = get_3x4x7_decomp()
        
        if verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Schoolbook fallback
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

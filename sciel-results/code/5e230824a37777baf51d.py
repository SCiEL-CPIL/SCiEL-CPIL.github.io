"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 core: rank 7
    # Tensors: U[0:4, :], V[0:4, :], W[0:4, :]
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Build Strassen factor matrices for 2x2x2
    def strassen_2x2x2():
        sU = [[0]*7 for _ in range(4)]
        sV = [[0]*7 for _ in range(4)]
        sW = [[0]*7 for _ in range(4)]
        
        # Each term c has contributions to U (A side), V (B side), W (C side)
        # U index: i*2+j, V index: j*2+k, W index: k*2+i
        
        # M1 = (A00+A11)*(B00+B11)
        # A side: A00 at (0,0) and A11 at (1,1)
        # B side: B00 at (0,0) and B11 at (1,1)
        # C side: contributes to C00 (+1), C11 (+1)
        sU[0*2+0][0] = 1; sU[1*2+1][0] = 1
        sV[0*2+0][0] = 1; sV[1*2+1][0] = 1
        sW[0*2+0][0] = 1; sW[1*2+1][0] = 1  # C00, C11
        
        # M2 = (A10+A11)*B00
        sU[1*2+0][1] = 1; sU[1*2+1][1] = 1
        sV[0*2+0][1] = 1
        sW[0*2+1][1] = 1  # C10
        
        # M3 = A00*(B01-B11)
        sU[0*2+0][2] = 1
        sV[0*2+1][2] = 1; sV[1*2+1][2] = -1
        sW[1*2+0][2] = 1  # C01
        
        # M4 = A11*(B10-B00)
        sU[1*2+1][3] = 1
        sV[1*2+0][3] = 1; sV[0*2+0][3] = -1
        sW[0*2+0][3] = 1; sW[0*2+1][3] = 1  # C00, C10
        
        # M5 = (A00+A01)*B11
        sU[0*2+0][4] = 1; sU[0*2+1][4] = 1
        sV[1*2+1][4] = 1
        sW[1*2+0][4] = -1  # C01 (negative because C00 = M1+M4-M5+M7)
        
        # M6 = (A10-A00)*(B00+B10)
        sU[1*2+0][5] = 1; sU[0*2+0][5] = -1
        sV[0*2+0][5] = 1; sV[1*2+0][5] = 1
        sW[1*2+1][5] = 1  # C11
        
        # M7 = (A01-A11)*(B01+B11)
        sU[0*2+1][6] = 1; sU[1*2+1][6] = -1
        sV[0*2+1][6] = 1; sV[1*2+1][6] = 1
        sW[0*2+0][6] = 1; sW[1*2+1][6] = 1  # C00, C11
        
        return sU, sV, sW
    
    sU, sV, sW = strassen_2x2x2()
    
    # Schoolbook factor for ni x mj x pk
    def schoolbook_2d(ni, mj, pk):
        rank = ni * mj * pk
        U = [[0]*rank for _ in range(ni*mj)]
        V = [[0]*rank for _ in range(mj*pk)]
        W = [[0]*rank for _ in range(pk*ni)]
        col = 0
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[i*mj+j][col] = 1
                    V[j*pk+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W
    
    # Try different partitions
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    def try_partition(part_n, part_m, part_p):
        nonlocal best_rank, best_result
        if time.time() > deadline - 5:
            return
        
        total_rank = 0
        # Count total rank first
        for ni in part_n:
            for mj in part_m:
                for pk in part_p:
                    if ni == 2 and mj == 2 and pk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * mj * pk
        
        if total_rank >= best_rank:
            return
        
        # Build the full decomposition
        U_rows = []
        V_rows = []
        W_rows = []
        
        # Track offsets
        n_offset = 0
        m_offset = 0
        p_offset = 0
        
        for ni in part_n:
            m_offset = 0
            for mj in part_m:
                p_offset = 0
                for pk in part_p:
                    if ni == 2 and mj == 2 and pk == 2:
                        # Use Strassen
                        block_U = sU
                        block_V = sV
                        block_W = sW
                        block_rank = 7
                    else:
                        block_U, block_V, block_W = schoolbook_2d(ni, mj, pk)
                        block_rank = ni * mj * pk
                    
                    # Embed into full matrices
                    for r in range(ni * mj):
                        U_rows.append([0] * total_rank)
                    for r in range(mj * pk):
                        V_rows.append([0] * total_rank)
                    for r in range(pk * ni):
                        W_rows.append([0] * total_rank)
                    
                    col_start = sum(
                        (7 if (a==2 and b==2 and c==2) else a*b*c)
                        for a in part_n for b in part_m for c in part_p
                        if (a, b, c) == (ni, mj, pk) or 
                           (list(part_n).index(a) < list(part_n).index(ni)) or
                           (list(part_n).index(a) == list(part_n).index(ni) and 
                            list(part_m).index(b) < list(part_m).index(mj)) or
                           (list(part_n).index(a) == list(part_n).index(ni) and 
                            list(part_m).index(b) == list(part_m).index(mj) and 
                            list(part_p).index(c) < list(part_p).index(pk))
                    )
                    
                    # Simpler: just track column offset
                    p_offset += pk
                m_offset += mj
            n_offset += ni
        
        # This approach is getting complex. Let me use a cleaner method.
        return total_rank, None
    
    # Cleaner approach: build directly
    def build_decomposition(part_n, part_m, part_p):
        if time.time() > deadline - 5:
            return None
        
        total_rank = 0
        for ni in part_n:
            for mj in part_m:
                for pk in part_p:
                    if ni == 2 and mj == 2 and pk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * mj * pk
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ni_idx, ni in enumerate(part_n):
            for mj_idx, mj in enumerate(part_m):
                for pk_idx, pk in enumerate(part_p):
                    if ni == 2 and mj == 2 and pk == 2:
                        block_U = sU
                        block_V = sV
                        block_W = sW
                        block_rank = 7
                    else:
                        block_U, block_V, block_W = schoolbook_2d(ni, mj, pk)
                        block_rank = ni * mj * pk
                    
                    # Row offsets in full matrices
                    u_row_start = sum(part_n[:ni_idx]) * m + sum(part_m[:mj_idx])
                    v_row_start = sum(part_m[:mj_idx]) * p + sum(part_p[:pk_idx])
                    w_row_start = sum(part_p[:pk_idx]) * n + sum(part_n[:ni_idx])
                    
                    for r in range(block_rank):
                        for i in range(len(block_U)):
                            if block_U[i][r] != 0:
                                U[u_row_start + i][col + r] = block_U[i][r]
                        for j in range(len(block_V)):
                            if block_V[j][r] != 0:
                                V[v_row_start + j][col + r] = block_V[j][r]
                        for k in range(len(block_W)):
                            if block_W[k][r] != 0:
                                W[w_row_start + k][col + r] = block_W[k][r]
                    
                    col += block_rank
        
        return {"U": U, "V": V, "W": W}
    
    # Try all possible partitions of 3, 5, 7 into parts of size 1 or 2
    def get_partitions(total, max_part=2):
        if total == 0:
            yield []
            return
        for first in range(1, min(max_part, total) + 1):
            for rest in get_partitions(total - first, max_part):
                yield [first] + rest
    
    best_rank = n * m * p
    best_result = None
    
    for part_n in get_partitions(n):
        for part_m in get_partitions(m):
            for part_p in get_partitions(p):
                if time.time() > deadline - 10:
                    break
                
                total_rank = 0
                for ni in part_n:
                    for mj in part_m:
                        for pk in part_p:
                            if ni == 2 and mj == 2 and pk == 2:
                                total_rank += 7
                            else:
                                total_rank += ni * mj * pk
                
                if total_rank < best_rank:
                    result = build_decomposition(part_n, part_m, part_p)
                    if result is not None:
                        best_rank = total_rank
                        best_result = result
                    if time.time() > deadline - 10:
                        break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END


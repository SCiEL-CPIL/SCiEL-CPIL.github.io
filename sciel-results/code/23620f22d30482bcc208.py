def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    sU, sV, sW = _strassen_2x2()
    
    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0-3 of B and C)
    # 2x2 block structure, schoolbook outer (4 terms), Strassen inner (7)
    # Total: 4 * 7 = 28
    
    # Part 2: 4x4 * 4x2 -> 4x2 (columns 4-5 of B and C)
    # 2x2 block structure, schoolbook outer (4 terms), Strassen inner (7)
    # Total: 4 * 7 = 28
    
    total_rank = 56
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Part 1: cols 0-3
    col_offset = 0
    for ai in range(2):
        for bj in range(2):
            for t in range(7):
                c = col_offset + (ai*2 + bj) * 7 + t
                
                # U: A[i,j] where i=2*ai+ai2, j=2*bj+bj2
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff = sU[ai2*2 + bj2, t]
                        if coeff != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, c] += coeff
                
                # V: B[j,k] where j=2*bj+bj2, k=2*bj+ck2... wait
                # B is 4x4, blocks: B[bj][ck] where bj,ck in {0,1}
                # For this term, we need B[bj][bj] block (same block index for both)
                # Actually no: C_block[ai][bj] = sum_bj' A_block[ai][bj'] * B_block[bj'][bj]
                # Wait, I'm confusing myself.
                
                # Let me redo. For 4x4 * 4x4:
                # C_block[ai][cj] = sum_bj A_block[ai][bj] * B_block[bj][cj]
                # Each outer term is (ai, bj, cj) but we're doing schoolbook so
                # each term fixes ai, bj, cj and does A_block[ai][bj] * B_block[bj][cj]
                
                # I made an error. Let me restart the loop structure.

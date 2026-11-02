"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition 12 columns into 3 groups of 4.
    # For each group of 4 columns (2x2x4), use rank 11:
    #   4 naive products for column 0 of the group
    #   7 Strassen products for columns 1,2,3 of the group
    # Total rank = 3 * 11 = 33.
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(3):
        col_offset = block_idx * 4
        base = block_idx * 11
        
        # --- Naive 4 products for column col_offset (local col 0) ---
        # C[i, col_offset] = A[i,0]*B[0,col_offset] + A[i,1]*B[1,col_offset]
        # r=0: A00 * B0,local0
        U[0*2+0][base+0] = 1
        V[0*12+col_offset][base+0] = 1
        W[col_offset*2+0][base+0] += 1
        
        # r=1: A10 * B0,local0
        U[1*2+0][base+1] = 1
        V[0*12+col_offset][base+1] = 1
        W[col_offset*2+1][base+1] += 1
        
        # r=2: A01 * B1,local0
        U[0*2+1][base+2] = 1
        V[1*12+col_offset][base+2] = 1
        W[col_offset*2+0][base+2] += 1
        
        # r=3: A11 * B1,local0
        U[1*2+1][base+3] = 1
        V[1*12+col_offset][base+3] = 1
        W[col_offset*2+1][base+3] += 1
        
        # --- Strassen 7 products for columns col_offset+1, col_offset+2, col_offset+3 ---
        # These are local columns 1,2,3.
        # We treat this as a 2x2x3 multiplication.
        # Let's use a rank-7 decomposition for 2x2x3.
        # 
        # For 2x2x3: A is 2x2, B is 2x3, C is 2x3.
        # C[0,0] = A00*B00 + A01*B10
        # C[1,0] = A10*B00 + A11*B10
        # C[0,1] = A00*B01 + A01*B11
        # C[1,1] = A10*B01 + A11*B11
        # C[0,2] = A00*B02 + A01*B12
        # C[1,2] = A10*B02 + A11*B12
        #
        # Local cols: lc0 = col_offset+1, lc1 = col_offset+2, lc2 = col_offset+3
        #
        # m1 = A00*(B01 - B11)  [using local col 1 and 2 of B]
        # Wait, I need a proper 2x2x3 rank-7 decomposition.
        #
        # Let me use the following approach:
        # Apply Strassen to cols (lc0, lc1) as a 2x2x2, then handle lc2.
        # But that gives 7+4=11 for the 3 cols, total 4+11=15 per block. Too much.
        #
        # Alternative: find a direct rank-7 for 2x2x3.
        #
        # Actually, let me just use 2 Strassen blocks (7+7=14) for the 4 cols,
        # total 2*7*3 = 42. That's the same as before!
        #
        # Let me reconsider. For 2x2x3, the minimum rank is 7 (same as 2x2x2
        # since the extra column doesn't help reduce rank below 7... actually it
        # might be higher).
        #
        # Let me just use naive for all 4 columns: 4*4 = 16 per block, total 48. Worse.
        #
        # Best option: 2 Strassen blocks per group of 4 = 14 per block, total 42.
        # That's the same as the original!
        #
        # Let me try: 1 Strassen (cols 0,1) + naive (cols 2,3) = 7+4 = 11 per block.
        # Total = 33.
        
        # Strassen for local cols 1,2 (global cols col_offset+1, col_offset+2)
        lc0 = col_offset + 1
        lc1 = col_offset + 2
        lc2 = col_offset + 3
        
        # m1 = A00*(B0,lc1 - B1,lc1) ... let me use standard Strassen
        # for the 2x2 submatrix formed by local cols lc0, lc1.
        
        # m1: A00 * (B0,lc1 - B1,lc1)
        U[0*2+0][base+4] = 1
        V[0*12+lc1][base+4] = 1
        V[1*12+lc1][base+4] = -1
        W[lc0*2+0][base+4] += -1  # C[0,lc0] gets -A00*B1,lc1... 
        # Hmm, this is getting confused. Let me restart with a clean Strassen for 2x2x2.
        
        # Standard Strassen for 2x2x2 (A is 2x2, B is 2x2, C is 2x2):
        # m1 = A00*(B01-B11)
        # m2 = (A10+A11)*B00
        # m3 = (A00+A01)*B11
        # m4 = A11*(B00-B10)
        # m5 = (A00+A10)*B01
        # m6 = (A11-A00)*(B10+B00)
        # m7 = (A01-A11)*(B11+B01)
        # C00 = m1+m4+m6
        # C01 = m1+m5
        # C10 = m2+m7
        # C11 = m3-m5+m7
        
        # Here, local col 0 of this sub-block = global col lc0, local col 1 = lc1.
        # B[j, lc0] and B[j, lc1] are the B entries.
        # C[i, lc0] and C[i, lc1] are the C entries.
        
        # m1 = A00*(B0,lc1 - B1,lc1)
        U[0*2+0][base+4] = 1
        V[0*12+lc1][base+4] = 1
        V[1*12+lc1][base+4] = -1
        W[lc0*2+0][base+4] += 1
        W[lc1*2+0][base+4] += 1
        
        # m2 = (A10+A11)*B0,lc0
        U[1*2+0][base+5] = 1
        U[1*2+1][base+5] = 1
        V[0*12+lc0][base+5] = 1
        W[lc0*2+1][base+5] += 1
        W[lc1*2+1][base+5] += 1
        
        # m3 = (A00+A01)*B1,lc1
        U[0*2+0][base+6] = 1
        U[0*2+1][base+6] = 1
        V[1*12+lc1][base+6] = 1
        W[lc0*2+0][base+6] += 1
        W[lc1*2+0][base+6] += 1
        
        # m4 = A11*(B0,lc0 - B1,lc0)
        U[1*2+1][base+7] = 1
        V[0*12+lc0][base+7] = 1
        V[1*12+lc0][base+7] = -1
        W[lc0*2+1][base+7] += 1
        W[lc1*2+1][base+7] += 1
        
        # m5 = (A00+A10)*B0,lc1
        U[0*2+0][base+8] = 1
        U[1*2+0][base+8] = 1
        V[0*12+lc1][base+8] = 1
        W[lc1*2+0][base+8] += 1
        W[lc1*2+1][base+8] += 1
        
        # m6 = (A11-A00)*(B1,lc0+B0,lc0)
        U[1*2+1][base+9] = 1
        U[0*2+0][base+9] = -1
        V[1*12+lc0][base+9] = 1
        V[0*12+lc0][base+9] = 1
        W[lc0*2+0][base+9] += 1
        W[lc1*2+0][base+9] += 1
        
        # m7 = (A01-A11)*(B1,lc1+B0,lc1)
        U[0*2+1][base+10] = 1
        U[1*2+1][base+10] = -1
        V[1*12+lc1][base+10] = 1
        V[0*12+lc1][base+10] = 1
        W[lc0*2+1][base+10] += 1
        W[lc1*2+1][base+10] += 1
        
        # Now I need to handle column lc2 (the 4th column of this group) with 4 naive products.
        # But I only have 11 slots per block (base+0 to base+10).
        # I've used base+0..3 for col 0, base+4..10 for cols 1,2.
        # I need 4 more for col 3. That's 15 total per block, not 11.
        
        # Let me redo this with rank 15 per block, total 45. That's worse than 42!
        
        # OK, let me just stick with the original approach but verify it's correct.
        pass
    
    # Let me just return the original 42-rank solution since I can't find something better.
    # Actually, let me try rank 14 per block (2 Strassen), total 42. Same as before.
    
    # Let me try a different partition: 2 groups of 6 columns each.
    # For 2x2x6, use 3 Strassen blocks = rank 21. Total = 42. Same.
    
    # I'll just keep the original solution.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    block_idx = 0
    while col_offset < p:
        # Each block handles 2 columns of B
        # Strassen decomposition for 2x2x2
        # U indices: i*m+j
        # V indices: j*p+k (k is local 0,1)
        # W indices: k*n+i (k is local 0,1)
        
        # m1
        U[0*2+0][block_idx*7 + 0] = 1
        U[1*2+1][block_idx*7 + 0] = 1
        V[0*12+col_offset][block_idx*7 + 0] = 1
        V[1*12+col_offset+1][block_idx*7 + 0] = 1
        W[col_offset*2+0][block_idx*7 + 0] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 0] += 1
        
        # m2
        U[1*2+0][block_idx*7 + 1] = 1
        U[1*2+1][block_idx*7 + 1] = 1
        V[0*12+col_offset][block_idx*7 + 1] = 1
        W[col_offset*2+1][block_idx*7 + 1] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 1] -= 1
        
        # m3
        U[0*2+0][block_idx*7 + 2] = 1
        V[0*12+col_offset+1][block_idx*7 + 2] = 1
        V[1*12+col_offset+1][block_idx*7 + 2] = -1
        W[(col_offset+1)*2+0][block_idx*7 + 2] += 1
        W[(col_offset+1)*2+1][block_idx*7 + 2] += 1
        
        # m4
        U[1*2+1][block_idx*7 + 3] = 1
        V[1*12+col_offset][block_idx*7 + 3] = 1
        V[0*12+col_offset][block_idx*7 + 3] = -1
        W[col_offset*2+0][block_idx*7 + 3] += 1
        W[col_offset*2+1][block_idx*7 + 3] += 1
        
        # m5
        U[0*2+0][block_idx*7 + 4] = 1
        U[0*2+1][block_idx*7 + 4] = 1
        V[1*12+col_offset+1][block_idx*7 + 4] = 1
        W[col_offset*2+0][block_idx*7 + 4] -= 1
        W[(col_offset+1)*2+0][block_idx*7 + 4] += 1
        
        # m6
        U[1*2+0][block_idx*7 + 5] = 1
        U[0*2+0][block_idx*7 + 5] = -1
        V[0*12+col_offset][block_idx*7 + 5] = 1
        V[0*12+col_offset+1][block_idx*7 + 5] = 1
        W[(col_offset+1)*2+1][block_idx*7 + 5] += 1
        
        # m7
        U[0*2+1][block_idx*7 + 6] = 1
        U[1*2+1][block_idx*7 + 6] = -1
        V[1*12+col_offset][block_idx*7 + 6] = 1
        V[1*12+col_offset+1][block_idx*7 + 6] = 1
        W[col_offset*2+0][block_idx*7 + 6] += 1
        
        col_offset += 2
        block_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

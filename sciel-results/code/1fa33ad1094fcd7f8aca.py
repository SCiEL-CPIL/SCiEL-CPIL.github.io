"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll use a block decomposition approach.
    # Split n=4 into two blocks of size 2.
    # Split p=6 into three blocks of size 2.
    # For each (row_block, col_block) pair, we need to compute a 2x5x2 sub-problem.
    
    # For the 2x5x2 sub-problem, we can use a decomposition based on Strassen for 2x2x2.
    # Split the 5 middle indices into: one index of size 1 and two indices of size 2.
    # So we have two sub-sub-problems: 2x1x2 (rank 4) and 2x2x2 (rank 7 via Strassen).
    # Total rank per 2x5x2 block: 4 + 7 = 11.
    # Total rank: 2 * 3 * 11 = 66.
    
    rank_total = 66
    
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    for ib in range(2):  # row block index (0 or 1)
        for jb in range(3):  # col block index (0, 1, or 2)
            # Local indices: i' in {0,1}, j in {0..4}, k' in {0,1}
            # Global indices: i = ib*2 + i', k = jb*2 + k'
            
            # Sub-sub-problem 1: 2x1x2 (using j=0 only)
            # This is just schoolbook: rank 4
            # For each i', k', we need U[i,m+0] * V[0,p+k] * W[k,n+i] = 1
            # Actually, for 2x1x2, the tensor is T[i', 0, k'] = 1 for all i', k'
            # We can decompose this as: sum over (i',k') of e_{i'} * 1 * e_{k'}
            # That's 4 terms.
            
            for i_prime in range(2):
                for k_prime in range(2):
                    i_global = ib * 2 + i_prime
                    k_global = jb * 2 + k_prime
                    U[i_global * m + 0][col_offset] = 1.0
                    V[0 * p + k_global][col_offset] = 1.0
                    W[k_global * n + i_global][col_offset] = 1.0
                    col_offset += 1
            
            # Sub-sub-problem 2: 2x2x2 (using j=1,2)
            # Use Strassen's algorithm for rank 7.
            # Local j indices: j1 = 1, j2 = 2
            # We'll use the standard Strassen decomposition.
            
            # Strassen's 7 multiplications for 2x2 matrices:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A10 + A11) * B00
            # M3 = A00 * (B10 - B00)
            # M4 = A11 * (B01 - B10)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B10)
            # M7 = (A01 - A11) * (B10 + B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For our case, A is 2x2 with entries A[i', j_local] where j_local in {0,1} corresponds to j=1,2
            # B is 2x2 with entries B[j_local, k'] where j_local in {0,1} corresponds to j=1,2
            
            # Let's define the Strassen factors.
            # For each multiplication r (0..6), we have coefficients for U, V, W.
            
            # M1: U coeff for A: A00 + A11 -> U[0,0]=1, U[1,1]=1
            #     V coeff for B: B00 + B11 -> V[0,0]=1, V[1,1]=1
            #     W coeff: C00 += 1, C11 += 1
            U[(ib*2+0)*m + 1][col_offset] = 1.0
            U[(ib*2+1)*m + 2][col_offset] = 1.0
            V[1*p + (jb*2+0)][col_offset] = 1.0
            V[2*p + (jb*2+1)][col_offset] = 1.0
            W[(jb*2+0)*n + (ib*2+0)][col_offset] = 1.0
            W[(jb*2+1)*n + (ib*2+1)][col_offset] = 1.0
            col_offset += 1
            
            # M2: U coeff: A10 + A11 -> U[1,0]=1, U[1,1]=1
            #     V coeff: B00 -> V[0,0]=1
            #     W coeff: C10 += 1, C11 -= 1
            U[(ib*2+1)*m + 1][col_offset] = 1.0
            U[(ib*2+1)*m + 2][col_offset] = 1.0
            V[1*p + (jb*2+0)][col_offset] = 1.0
            W[(jb*2+0)*n + (ib*2+1)][col_offset] = 1.0
            W[(jb*2+1)*n + (ib*2+1)][col_offset] = -1.0
            col_offset += 1
            
            # M3: U coeff: A00 -> U[0,0]=1
            #     V coeff: B10 - B00 -> V[1,0]=1, V[0,0]=-1
            #     W coeff: C01 += 1, C11 += 1
            U[(ib*2+0)*m + 1][col_offset] = 1.0
            V[2*p + (jb*2+0)][col_offset] = 1.0
            V[1*p + (jb*2+0)][col_offset] = -1.0
            W[(jb*2+1)*n + (ib*2+0)][col_offset] = 1.0
            W[(jb*2+1)*n + (ib*2+1)][col_offset] = 1.0
            col_offset += 1
            
            # M4: U coeff: A11 -> U[1,1]=1
            #     V coeff: B01 - B10 -> V[0,1]=1, V[1,0]=-1
            #     W coeff: C00 += 1, C10 += 1
            U[(ib*2+1)*m + 2][col_offset] = 1.0
            V[1*p + (jb*2+1)][col_offset] = 1.0
            V[2*p + (jb*2+0)][col_offset] = -1.0
            W[(jb*2+0)*n + (ib*2+0)][col_offset] = 1.0
            W[(jb*2+0)*n + (ib*2+1)][col_offset] = 1.0
            col_offset += 1
            
            # M5: U coeff: A00 + A01 -> U[0,0]=1, U[0,1]=1
            #     V coeff: B11 -> V[1,1]=1
            #     W coeff: C00 -= 1, C01 += 1
            U[(ib*2+0)*m + 1][col_offset] = 1.0
            U[(ib*2+0)*m + 2][col_offset] = 1.0
            V[2*p + (jb*2+1)][col_offset] = 1.0
            W[(jb*2+0)*n + (ib*2+0)][col_offset] = -1.0
            W[(jb*2+1)*n + (ib*2+0)][col_offset] = 1.0
            col_offset += 1
            
            # M6: U coeff: A10 - A00 -> U[1,0]=1, U[0,0]=-1
            #     V coeff: B00 + B10 -> V[0,0]=1, V[1,0]=1
            #     W coeff: C11 += 1
            U[(ib*2+1)*m + 1][col_offset] = 1.0
            U[(ib*2+0)*m + 1][col_offset] = -1.0
            V[1*p + (jb*2+0)][col_offset] = 1.0
            V[2*p + (jb*2+0)][col_offset] = 1.0
            W[(jb*2+1)*n + (ib*2+1)][col_offset] = 1.0
            col_offset += 1
            
            # M7: U coeff: A01 - A11 -> U[0,1]=1, U[1,1]=-1
            #     V coeff: B10 + B11 -> V[1,0]=1, V[1,1]=1
            #     W coeff: C00 += 1
            U[(ib*2+0)*m + 2][col_offset] = 1.0
            U[(ib*2+1)*m + 2][col_offset] = -1.0
            V[2*p + (jb*2+0)][col_offset] = 1.0
            V[2*p + (jb*2+1)][col_offset] = 1.0
            W[(jb*2+0)*n + (ib*2+0)][col_offset] = 1.0
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

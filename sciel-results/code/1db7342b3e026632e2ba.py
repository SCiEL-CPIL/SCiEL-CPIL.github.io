"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import itertools

def _build_strassen_4x4x4():
    """
    Build a rank-49 exact decomposition for 4x4 * 4x4 -> 4x4 using Strassen.
    Returns U (16x49), V (16x49), W (16x49) as lists of lists.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    n = 4
    U = [[0.0]*49 for _ in range(16)]
    V = [[0.0]*49 for _ in range(16)]
    W = [[0.0]*49 for _ in range(16)]
    
    # Strassen for 4x4: partition into 2x2 blocks of size 2x2.
    # There are 16 blocks of size 2x2 in the 4x4 matrix.
    # Actually, let's use the standard 4x4 Strassen which has 49 multiplications?
    # No, standard Strassen for 4x4 can be done by applying 2x2 Strassen recursively.
    # 4x4 = 2x2 blocks of 2x2.
    # To multiply two 4x4 matrices A and B using Strassen:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Cij are 2x2.
    # We need to compute 7 "scalar" (which are actually 2x2 matrix) multiplications M1..M7.
    # Each M is a 2x2 matrix product.
    # Each 2x2 matrix product requires 7 scalar multiplications.
    # Total 7 * 7 = 49 scalar multiplications.
    
    # Indices for 2x2 blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # The 7 intermediate 2x2 matrices are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will build the U, V, W for each of the 7 "super" multiplications.
    # For a given super multiplication r (0..6), we have 7 sub-multiplications s (0..6).
    # Total rank index r = 7*r_super + r_sub.
    
    # We need a helper to get the Strassen coefficients for 2x2.
    # For 2x2: U2 (4x7), V2 (4x7), W2 (4x7).
    U2 = [[0.0]*7 for _ in range(4)]
    V2 = [[0.0]*7 for _ in range(4)]
    W2 = [[0.0]*7 for _ in range(4)]
    
    # Standard Strassen 2x2:
    # M1 = (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: C00+=1, C11+=1, C01-=-1, C10-=-1
    # M2 = (c+d)e -> U: c=1,d=1; V: e=1; W: C10+=1
    # M3 = a(f-h) -> U: a=1; V: f=1,h=-1; W: C01+=1
    # M4 = d(g-e) -> U: d=1; V: g=1,e=-1; W: C00+=-1, C11+=1
    # M5 = (a+b)h -> U: a=1,b=1; V: h=1; W: C00+=1, C10+=1
    # M6 = (c-a)(e+f) -> U: c=1,a=-1; V: e=1,f=1; W: C11+=1
    # M7 = (b-d)(g+h) -> U: b=1,d=-1; V: g=1,h=1; W: C01+=-1, C11+=-1
    
    # Let's hardcode the 2x2 Strassen matrices.
    # U2[i*2+j, k]
    # V2[j*2+k, k]
    # W2[k*2+i, k]
    
    # k=0: M1
    U2[0][0]=1; U2[3][0]=1 # a, d
    V2[0][0]=1; V2[3][0]=1 # e, h
    W2[0][0]=1; W2[3][0]=1 # C00, C11
    W2[1][0]=-1; W2[2][0]=-1 # C10, C01
    
    # k=1: M2
    U2[2][1]=1; U2[3][1]=1 # c, d
    V2[0][1]=1 # e
    W2[1][1]=1 # C10
    
    # k=2: M3
    U2[0][2]=1 # a
    V2[1][2]=1; V2[3][2]=-1 # f, -h
    W2[2][2]=1 # C01
    
    # k=3: M4
    U2[3][3]=1 # d
    V2[2][3]=1; V2[0][3]=-1 # g, -e
    W2[0][3]=-1; W2[3][3]=1 # -C00, C11
    
    # k=4: M5
    U2[0][4]=1; U2[1][4]=1 # a, b
    V2[3][4]=1 # h
    W2[0][4]=1; W2[1][4]=1 # C00, C10
    
    # k=5: M6
    U2[2][5]=1; U2[0][5]=-1 # c, -a
    V2[0][5]=1; V2[1][5]=1 # e, f
    W2[3][5]=1 # C11
    
    # k=6: M7
    U2[1][6]=1; U2[3][6]=-1 # b, -d
    V2[2][6]=1; V2[3][6]=1 # g, h
    W2[2][6]=-1; W2[3][6]=-1 # -C01, -C11
    
    # Now construct 4x4 Strassen.
    # Super multiplications map to specific 2x2 blocks.
    # Let's define the 2x2 block coordinates.
    # Block (br, bc) where br, bc in {0,1}.
    # br=0 -> rows 0,1; br=1 -> rows 2,3.
    # bc=0 -> cols 0,1; bc=1 -> cols 2,3.
    
    # For each super mul sm (0..6), we define:
    # A_blocks: list of (br, bc, coeff)
    # B_blocks: list of (br, bc, coeff)
    # Then for each sub mul s (0..6), we add to U, V, W.
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Note: A is 4x4, B is 4x4.
    # A00 is block (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1).
    
    super_defs = [
        # (A_blocks, B_blocks)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]), # M1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),           # M2
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),          # M3
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),          # M4
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),           # M5
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]), # M6
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])  # M7
    ]
    
    for sm, (a_blocks, b_blocks) in enumerate(super_defs):
        for s in range(7):
            r = sm * 7 + s
            # For each element in U2, V2, W2 at column s,
            # we need to map the 2x2 local indices to the 4x4 global indices.
            
            # U2 local index u_idx = i2*2 + j2.
            # This corresponds to row i2, col j2 in the 2x2 block.
            # The block is defined by (br, bc).
            # Global row = br*2 + i2.
            # Global col = bc*2 + j2.
            # U global index = (br*2+i2)*4 + (bc*2+j2).
            
            # V2 local index v_idx = j2*2 + k2.
            # Global row = br*2 + j2. (j2 is the row index in B's block? No, V is B.)
            # Wait, V is for matrix B.
            # In V, the first index is (row in B, col in B)? No.
            # V shape is (m*p, rank). m=4, p=4.
            # V[j*p+k, r] where j is row in B, k is col in B.
            # So V global index = (br*2+j2)*4 + (bc*2+k2).
            
            # W2 local index w_idx = k2*2 + i2.
            # W shape is (p*n, rank).
            # W[k*n+i, r] where k is col in C (row in B), i is row in C (col in A).
            # Wait, C = A B.
            # C[i][k] = sum_j A[i][j] B[j][k].
            # W[k*n+i, r].
            # k is the column index of C (0..3).
            # i is the row index of C (0..3).
            # In the 2x2 block (br, bc) for C?
            # C is partitioned into 4 blocks of 2x2.
            # C00: rows 0-1, cols 0-1.
            # C01: rows 0-1, cols 2-3.
            # C10: rows 2-3, cols 0-1.
            # C11: rows 2-3, cols 2-3.
            
            # The super multiplication sm produces a 2x2 matrix M_sm.
            # M_sm is added to specific C blocks.
            # Actually, the structure is:
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M4
            # C10 = M2 + M5
            # C11 = M1 - M2 + M3 + M6
            
            # So for a given sm, it contributes to one or more C blocks.
            # Let's define the C block contribution for each sm.
            
            # M1 -> C00 (+1), C11 (+1)
            # M2 -> C10 (+1), C11 (-1)
            # M3 -> C01 (+1), C11 (+1)
            # M4 -> C00 (-1), C01 (+1), C11 (+1)
            # M5 -> C00 (+1), C10 (+1)
            # M6 -> C11 (+1)
            # M7 -> C01 (-1), C11 (-1)
            
            # Let's refine the super_defs to include C block contributions.
            # Actually, it's easier to just hardcode the W mapping for 4x4 Strassen
            # based on the known decomposition.
            
            # Let's stick to the previous logic but fix the W mapping.
            # The W matrix for 4x4 Strassen at column r (sm, s) is non-zero
            # in the C blocks that M_sm contributes to.
            
            # Let's create a helper to get the C block contributions for sm.
            # sm=0 (M1): C00(1), C11(1)
            # sm=1 (M2): C10(1), C11(-1)
            # sm=2 (M3): C01(1), C11(1)
            # sm=3 (M4): C00(-1), C01(1), C11(1)
            # sm=4 (M5): C00(1), C10(1)
            # sm=5 (M6): C11(1)
            # sm=6 (M7): C01(-1), C11(-1)
            
            c_blocks = {
                0: [ (0,0,1), (1,1,1) ],
                1: [ (1,0,1), (1,1,-1) ],
                2: [ (0,1,1), (1,1,1) ],
                3: [ (0,0,-1), (0,1,1), (1,1,1) ],
                4: [ (0,0,1), (1,0,1) ],
                5: [ (1,1,1) ],
                6: [ (0,1,-1), (1,1,-1) ]
            }
            
            # Now, for each (i2, j2) in U2 where U2[i2*2+j2][s] != 0:
            # We need to find the global U index.
            # U2 is for the 2x2 multiplication.
            # The "row" i2 in U2 corresponds to the row in the A-block.
            # The "col" j2 in U2 corresponds to the col in the A-block.
            
            # For each a_block (br, bc, coeff) in a_blocks:
            # Global row = br*2 + i2
            # Global col = bc*2 + j2
            # U[(br*2+i2)*4 + (bc*2+j2)][r] += coeff * U2[i2*2+j2][s]
            
            for u_idx, val in enumerate(U2[s]): # Wait, U2 is list of lists, U2[row][col]
                pass # Need to iterate properly
            
            # Let's iterate i2, j2
            for i2 in range(2):
                for j2 in range(2):
                    val_u = U2[i2*2+j2][s]
                    if val_u != 0:
                        for br, bc, coeff in a_blocks:
                            gr = br*2 + i2
                            gc = bc*2 + j2
                            U[gr*4+gc][r] += coeff * val_u
                    
                    val_v = V2[i2*2+j2][s] # Note: V2 index is j2*2+k2. Here i2 is j2 in B?
                    # In V2, the first index is the row in B (j), second is col in B (k).
                    # In the super multiplication, the B block is (br, bc).
                    # So global row in B = br*2 + j2 (where j2 is the row index in the 2x2 block).
                    # Global col in B = bc*2 + k2 (where k2 is the col index in the 2x2 block).
                    # In V2[i2*2+j2], i2 is the row index, j2 is the col index?
                    # No, V2 is defined as V2[j*2+k, r].
                    # So V2[row][col].
                    # In the loop above, I used i2, j2. Let's rename to r2, c2 for clarity.
                    
                    # Let's restart the inner loop with clear variables.
                    
            # Corrected Inner Loop
            for r2 in range(2): # row in 2x2 block
                for c2 in range(2): # col in 2x2 block
                    val_u = U2[r2*2+c2][s]
                    if val_u != 0:
                        for br, bc, coeff in a_blocks:
                            gr = br*2 + r2
                            gc = bc*2 + c2
                            U[gr*4+gc][r] += coeff * val_u
                    
                    val_v = V2[r2*2+c2][s]
                    if val_v != 0:
                        for br, bc, coeff in b_blocks:
                            gr = br*2 + r2
                            gc = bc*2 + c2
                            V[gr*4+gc][r] += coeff * val_v
                    
                    # W mapping
                    # W2 index is k2*2 + i2.
                    # k2 is col in C (row in B).
                    # i2 is row in C (col in A).
                    # In the 2x2 block (cr, cc) of C:
                    # Global row in C = cr*2 + i2.
                    # Global col in C = cc*2 + k2.
                    # W global index = (cc*2+k2)*4 + (cr*2+i2).
                    
                    val_w = W2[c2*2+r2][s] # Wait, W2 index is k2*2+i2.
                    # In the loop, r2 is the row index in the 2x2 matrices (A/B/C).
                    # For W, r2 corresponds to i2 (row in C), and c2 corresponds to k2 (col in C)?
                    # Let's check W2 definition: W2[k2*2+i2, r].
                    # So first index is k2 (col in C), second is i2 (row in C).
                    # In the loop, we are iterating r2 (row) and c2 (col).
                    # So i2 = r2, k2 = c2.
                    # W2 index = c2*2 + r2.
                    if val_w != 0:
                        for cr, cc, coeff in c_blocks[sm]:
                            gi = cr*2 + r2 # row in C
                            gk = cc*2 + c2 # col in C
                            W[gk*4+gi][r] += coeff * val_w
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # C is 4x9.
    # We can view B as 4x9, which is 4x(4+4+1) or 4x(2+2+2+2+1)?
    # Let's split B into three blocks: B1 (4x4), B2 (4x4), B3 (4x1).
    # C1 = A * B1 (4x4) -> Use Strassen 4x4x4 (Rank 49)
    # C2 = A * B2 (4x4) -> Use Strassen 4x4x4 (Rank 49)
    # C3 = A * B3 (4x1) -> Use Schoolbook 4x1 (Rank 4*1=4? No, Rank 4*4*1=16)
    # Wait, schoolbook for 4x4 * 4x1 is Rank 16.
    # Can we do better for 4x1?
    # Minimum rank for 4x4 * 4x1 is 4*4*1 = 16?
    # Actually, the rank of the tensor [4,4,1] is 16.
    # So total rank = 49 + 49 + 16 = 114.
    
    # Let's try to split B into 4x4 and 4x4 and 4x1.
    # Total columns = 9. 4+4+1=9.
    
    U_total = []
    V_total = []
    W_total = []
    
    # Get Strassen 4x4x4 matrices
    U_s, V_s, W_s = _build_strassen_4x4x4()
    
    # Block 1: C1 (cols 0-3) = A * B1 (cols 0-3)
    # U, V, W are 16x49, 16x49, 16x49.
    # U_s[i*4+j, r], V_s[j*4+k, r], W_s[k*4+i, r].
    # Here k is 0..3 (col in C1).
    
    for r in range(49):
        U_col = U_s[:, r] if hasattr(U_s, '__getitem__') else [U_s[i][r] for i in range(16)]
        V_col = [V_s[i][r] for i in range(16)]
        W_col = [W_s[i][r] for i in range(16)]
        U_total.append(U_col)
        V_total.append(V_col)
        W_total.append(W_col)
    
    # Block 2: C2 (cols 4-7) = A * B2 (cols 4-7)
    # Same U, V, W structure, but V and W indices are shifted.
    # V index for B2: j is row in A/B (0..3), k is col in B2 (0..3) -> global col k+4.
    # So V global index = j*9 + (k+4).
    # W index for C2: k is col in C2 (0..3) -> global col k+4. i is row in C (0..3).
    # So W global index = (k+4)*4 + i.
    
    for r in range(49):
        U_col = [U_s[i][r] for i in range(16)]
        V_col = [0.0]*36 # m*p = 4*9 = 36
        W_col = [0.0]*36 # p*n = 9*4 = 36
        
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            V_col[j*9 + (k+4)] = V_s[idx][r]
            
            k_c = idx // 4 # k in W_s is k_c?
            i_c = idx % 4 # i in W_s is i_c?
            # W_s index is k*4+i.
            # So k_c = idx // 4, i_c = idx % 4.
            W_col[(k_c+4)*4 + i_c] = W_s[idx][r]
        
        U_total.append(U_col)
        V_total.append(V_col)
        W_total.append(W_col)
    
    # Block 3: C3 (col 8) = A * B3 (col 8)
    # Schoolbook 4x4 * 4x1.
    # Rank 16.
    for i in range(4):
        for j in range(4):
            for k in range(4): # k is 0 in B3? No, B3 is 4x1, so k is 0.
                # Wait, B3 is 4x1. So k is only 0.
                # A is 4x4. j is 0..3.
                # C3 is 4x1. i is 0..3.
                # Term: A[i,j] * B3[j,0].
                # U[i*4+j] = 1.
                # V[j*9+8] = 1.
                # W[8*4+i] = 1.
                U_col = [0.0]*16
                V_col = [0.0]*36
                W_col = [0.0]*36
                
                U_col[i*4+j] = 1.0
                V_col[j*9+8] = 1.0
                W_col[8*4+i] = 1.0
                
                U_total.append(U_col)
                V_total.append(V_col)
                W_total.append(W_col)
    
    rank = len(U_total)
    return {"U": U_total, "V": V_total, "W": W_total}

# EVOLVE-BLOCK-END

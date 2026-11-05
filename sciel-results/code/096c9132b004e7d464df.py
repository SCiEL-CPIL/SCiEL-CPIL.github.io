"""Optimized rank-56 decomposition for 2x4 * 4x8 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # We decompose into 4 independent 2x2x2 multiplications (one per pair of columns in C).
    # For each pair of columns (2k, 2k+1), C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
    # But B is 4x8, so B[:, 2k:2k+2] is 4x2.
    # A is 2x4. So this is (2x4) * (4x2) = (2x2).
    # We can further decompose A = [A1 | A2] where A1, A2 are 2x2.
    # B[:, 2k:2k+2] = [B1; B2] where B1, B2 are 2x2 (top 2 rows, bottom 2 rows).
    # Then C_part = A1 * B1 + A2 * B2.
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    # So total rank = 4 (column pairs) * 2 (A parts) * 7 (Strassen) = 56.
    
    # Strassen 2x2x2 decomposition:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen steps:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The 7 rank-1 terms for one 2x2x2:
    # (U, V, W) where U is 4x7, V is 4x7, W is 4x7
    # Input indices for A: 0=a, 1=b, 2=c, 3=d
    # Input indices for B: 0=e, 1=f, 2=g, 3=h
    # Output indices for C: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Strassen U, V, W matrices (4x7 each)
    # U: coefficients on A (flattened row-major: a=0, b=1, c=2, d=3)
    # V: coefficients on B (flattened row-major: e=0, f=1, g=2, h=3)
    # W: coefficients on C (flattened row-major: C11=0, C12=1, C21=2, C22=3)
    
    U_strassen = [
        # M1: (a+d)(e+h) -> a:1, d:1
        [1, 0, 0, 1],
        # M2: (c+d)e -> c:1, d:1
        [0, 0, 1, 1],
        # M3: a(f-h) -> a:1
        [1, 0, 0, 0],
        # M4: d(g-e) -> d:1
        [0, 0, 0, 1],
        # M5: (a+b)h -> a:1, b:1
        [1, 1, 0, 0],
        # M6: (c-a)(e+f) -> c:1, a:-1
        [-1, 0, 1, 0],
        # M7: (b-d)(g+h) -> b:1, d:-1
        [0, 1, 0, -1],
    ]
    
    V_strassen = [
        # M1: e:1, h:1
        [1, 0, 0, 1],
        # M2: e:1
        [1, 0, 0, 0],
        # M3: f:1, h:-1
        [0, 1, 0, -1],
        # M4: g:1, e:-1
        [-1, 0, 1, 0],
        # M5: h:1
        [0, 0, 0, 1],
        # M6: e:1, f:1
        [1, 1, 0, 0],
        # M7: g:1, h:1
        [0, 0, 1, 1],
    ]
    
    W_strassen = [
        # C11: M1+M4-M5+M7 -> C11=1
        [1, 0, 0, 0],
        # C21: M2+M4 -> C21=1
        [0, 0, 1, 0],
        # C12: M3+M5 -> C12=1
        [0, 1, 0, 0],
        # C22: M1-M2+M3+M6 -> C22=1
        [0, 0, 0, 1],
    ]
    
    # Wait, W should be 4x7 (4 outputs, 7 rank-1 terms)
    # Let me redo W. Each column of W corresponds to one M step, and gives the coefficients
    # for how that M step contributes to each of the 4 C entries.
    # M1 contributes: C11 += 1, C22 += 1
    # M2 contributes: C21 += 1, C22 -= 1
    # M3 contributes: C12 += 1, C22 += 1
    # M4 contributes: C11 += 1, C21 += 1
    # M5 contributes: C11 -= 1, C12 += 1
    # M6 contributes: C22 += 1
    # M7 contributes: C11 += 1
    
    # W is 4x7 (rows = C outputs, cols = M steps)
    W_strassen = [
        # C11: M1=1, M4=1, M5=-1, M7=1
        [1, 0, 0, 1, -1, 0, 1],
        # C12: M3=1, M5=1
        [0, 0, 1, 0, 1, 0, 0],
        # C21: M2=1, M4=1
        [0, 1, 0, 1, 0, 0, 0],
        # C22: M1=1, M2=-1, M3=1, M6=1
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    rank = 56
    # U is (n*m, rank) = (8, 56)
    # V is (m*p, rank) = (32, 56)
    # W is (p*n, rank) = (16, 56)
    
    U = [[0.0] * rank for _ in range(8)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(16)]
    
    col = 0
    # For each column pair k in 0..3 (corresponding to columns 2k, 2k+1 of C)
    for k in range(4):
        # For each A-part: 0 = top 2 rows of A (rows 0,1), 1 = bottom 2 rows of A (rows 2,3)
        for a_part in range(2):
            # For each Strassen step s in 0..6
            for s in range(7):
                # U: A has 8 elements (2x4, row-major). 
                # a_part=0: rows 0,1 -> A elements 0,1,2,3 (a,b,c,d for first 2x2 block)
                # a_part=1: rows 2,3 -> A elements 4,5,6,7
                # For a_part=0, the 2x2 sub-matrix is A[0:2, 0:2] -> indices 0,1,2,3
                # For a_part=1, the 2x2 sub-matrix is A[2:4, 0:2] -> indices 4,5,6,7
                # Wait, A is 2x4. So A[i,j] for i=0,1 and j=0,1,2,3.
                # The "first 2x2" is A[0:2, 0:2] = [[A00,A01],[A10,A11]] -> indices 0,1,2,3
                # The "second 2x2" is A[0:2, 2:4] = [[A02,A03],[A12,A13]] -> indices 4,5,6,7
                # Hmm, but I said a_part=0 is top 2 rows and a_part=1 is bottom 2 rows.
                # But A only has 2 rows! So this decomposition doesn't work that way.
                
                # Let me reconsider. A is 2x4, B is 4x8.
                # C[i,k] = sum_j A[i,j] B[j,k]
                # For fixed output column pair (2k, 2k+1), C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
                # B[:, 2k:2k+2] is 4x2.
                # A is 2x4.
                # So we're computing (2x4) * (4x2) = (2x2).
                # Split A into two 2x2 blocks: A = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
                # Split B[:, 2k:2k+2] into two 2x2 blocks: B_top = B[0:2, 2k:2k+2], B_bot = B[2:4, 2k:2k+2]
                # Then C[:, 2k:2k+2] = A_left * B_top + A_right * B_bot
                # Each term is 2x2x2, rank 7.
                # So for each k (4 choices) and each A-part (2 choices), we have a 2x2x2 multiplication.
                # Total: 4 * 2 * 7 = 56.
                
                # So a_part=0 means A_left (columns 0,1 of A), a_part=1 means A_right (columns 2,3 of A)
                # For a_part=0: A indices are 0,1,2,3 (row-major: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3)
                # For a_part=1: A indices are 4,5,6,7 (row-major: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7)
                
                # For B: B is 4x8. For column pair 2k, 2k+1:
                # B_top = B[0:2, 2k:2k+2] -> indices: B[0,2k]=2k, B[0,2k+1]=2k+1, B[1,2k]=2k+8, B[1,2k+1]=2k+9
                # B_bot = B[2:4, 2k:2k+2] -> indices: B[2,2k]=2k+16, B[2,2k+1]=2k+17, B[3,2k]=2k+24, B[3,2k+1]=2k+25
                
                # For a_part=0, we use B_top. For a_part=1, we use B_bot.
                
                # U: A element indices
                if a_part == 0:
                    a_indices = [0, 1, 2, 3]  # a,b,c,d
                else:
                    a_indices = [4, 5, 6, 7]
                
                # V: B element indices
                if a_part == 0:
                    # B_top: e=B[0,2k], f=B[0,2k+1], g=B[1,2k], h=B[1,2k+1]
                    b_indices = [2*k, 2*k+1, 2*k+8, 2*k+9]
                else:
                    # B_bot: e=B[2,2k], f=B[2,2k+1], g=B[3,2k], h=B[3,2k+1]
                    b_indices = [2*k+16, 2*k+17, 2*k+24, 2*k+25]
                
                # W: C output indices. C is 2x8, flattened as W[k*n+i, r] = C[i, k] where n=2.
                # So C[i, col] is at index col*2 + i.
                # For column pair 2k, 2k+1:
                # C[0, 2k] = index 2k*2 + 0 = 4k
                # C[0, 2k+1] = index (2k+1)*2 + 0 = 4k+2
                # C[1, 2k] = index 2k*2 + 1 = 4k+1
                # C[1, 2k+1] = index (2k+1)*2 + 1 = 4k+3
                c_indices = [4*k, 4*k+2, 4*k+1, 4*k+3]  # C11, C12, C21, C22
                
                # Now fill U, V, W for this rank-1 term
                for local_idx in range(4):
                    U[a_indices[local_idx]][col] += U_strassen[s][local_idx]
                    V[b_indices[local_idx]][col] += V_strassen[s][local_idx]
                    W[c_indices[local_idx]][col] += W_strassen[local_idx][s]
                
                col += 1
    
    # Verify: convert to fractions if needed, but floats should be exact for integers.
    # Convert to int if all values are integers
    U_int = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V_int = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W_int = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return {"U": U_int, "V": V_int, "W": W_int}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback: rank 36
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Strategy: Use Strassen for 2x2x2 blocks + schoolbook for remainder
    # Block decomposition:
    # i in {0,1} vs i=2; j in {0,1} vs j=2; k in {0,1} vs k in {2,3}
    # 2x2x2 blocks: (i<2,j<2,k<2) and (i<2,j<2,k>=2) -> 2 blocks of rank 7 each
    # Remaining: schoolbook
    # Total: 7 + 7 + 4 + 4 + 4 + 4 + 2 + 2 = 34
    
    # Strassen 2x2x2 tensor coefficients (verified):
    # For C[i,k] = sum_j A[i,j] B[j,k] with i,j,k in {0,1}:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 + M3 - M2 + M6
    
    # Let me verify C00:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -A11B01 + A11B00
    # M5 = A00B11 + A01B11
    # M7 = A01B01 + A01B11 - A11B01 - A11B11
    # Sum = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B00 + A00B11 + A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    # = A00B00 + 2A00B11 + 2A11B00 - 2A11B01 + 2A01B11 + A01B01
    # That's not A00B00 + A01B10. So the standard Strassen formulas I'm using are wrong.
    
    # Let me use the correct Strassen formulas. The standard ones are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10 - B11)
    # M4 = (A11)(B01 - B00)
    # M5 = (A00 + A01)(B11)
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 + M3 - M2 + M6
    
    # Hmm, these are the same formulas. Let me try a different set.
    # Actually, I think the issue is that I'm confusing the Strassen for C=AB vs C=BA.
    # Let me just use a known-correct rank-7 decomposition for 2x2x2.
    
    # Alternative approach: use the fact that we can verify any decomposition.
    # Let me construct the Strassen decomposition and verify it.
    
    # Correct Strassen (from Wikipedia, C = A*B):
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b11)
    # p4 = a11(b01 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)
    #
    # c00 = p1 - p4 + p5 + p7
    # c01 = p3 + p4
    # c10 = p2 + p7
    # c11 = p1 + p3 - p2 + p6
    
    # Let me verify c00 = a00*b00 + a01*b10:
    # p1 = a00b00 + a00b11 + a11b00 + a11b11
    # -p4 = -a11b01 + a11b00
    # p5 = a00b11 + a01b11
    # p7 = a01b01 + a01b11 - a11b01 - a11b11
    # Sum: a00b00 + a00b11 + a11b00 + a11b11 - a11b01 + a11b00 + a00b11 + a01b11 + a01b01 + a01b11 - a11b01 - a11b11
    # = a00b00 + 2a00b11 + 2a11b00 - 2a11b01 + 2a01b11 + a01b01
    # This is NOT a00b00 + a01b10. So these formulas are incorrect.
    
    # I think the correct Strassen formulas for C = A*B (where C[i,k] = sum_j A[i,j]B[j,k]) are:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)(b00 + b01)  -- NO, this doesn't look right either.
    
    # Let me just look this up properly. The standard Strassen for 2x2 matrix multiplication:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+g)
    # p7 = (b-d)(f+h)
    #
    # C[0,0] = p1 - p4 + p5 + p7
    # C[0,1] = p3 + p4
    # C[1,0] = p2 + p7
    # C[1,1] = p1 + p3 - p2 + p6
    
    # Verify C[0,0] = ae + bg:
    # p1 = ae + ah + de + dh
    # -p4 = -dg + de
    # p5 = ah + bh
    # p7 = bf + bh - df - dh
    # Sum = ae + ah + de + dh - dg + de + ah + bh + bf + bh - df - dh
    # = ae + 2ah + 2de - dg + 2bh + bf - df
    # This is NOT ae + bg. So I'm still getting it wrong.
    
    # I think the issue is that I'm using the wrong Strassen formulas. Let me use a different approach.
    # Instead of trying to recall Strassen, let me just use schoolbook for everything.
    # The score for rank 36 is 1.0, and we need to beat that.
    
    # Actually, let me try a different approach: use the Winograd Fourier Transform Algorithm.
    # For 3x3 matrix multiplication, the Winograd algorithm achieves rank 23 (Lader's result).
    # For 3x3x4, we can potentially do better than 36.
    
    # Let me try a block approach:
    # Split the 3x3x4 problem into:
    # - A 2x2x2 block (i<2, j<2, k<2) using Strassen (rank 7)
    # - A 2x2x2 block (i<2, j<2, k>=2) using Strassen (rank 7)
    # - Remaining terms using schoolbook
    
    # But I need to get Strassen right. Let me just use a known-correct rank-7 decomposition.
    
    # Actually, let me try a simpler approach: use the fact that for the 2x2x2 case,
    # the tensor can be decomposed with rank 7. I'll construct it explicitly.
    
    # For 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1} (8 terms).
    # We want to find 7 rank-1 terms that sum to this tensor.
    
    # The Strassen decomposition for 2x2x2:
    # U (4x7), V (4x7), W (4x7) where:
    # U[i,j,r] for i,j in {0,1}, r in {0..6}
    # V[j,k,r] for j,k in {0,1}, r in {0..6}
    # W[i,k,r] for i,k in {0,1}, r in {0..6}
    
    # From the literature, the Strassen 2x2x2 decomposition is:
    # r=0: U: (1,0,1,0) [i,j: 00,01,10,11], V: (1,0,0,1), W: (1,1,1,1)
    # r=1: U: (0,0,1,1), V: (1,0,0,0), W: (0,0,1,0)
    # r=2: U: (1,0,0,0), V: (0,0,1,-1), W: (0,1,0,0)
    # r=3: U: (0,0,0,1), V: (0,1,-1,0), W: (-1,1,0,0)
    # r=4: U: (1,1,0,0), V: (0,0,0,1), W: (1,0,0,0)
    # r=5: U: (0,0,1,-1), V: (1,0,1,0), W: (0,0,0,1)
    # r=6: U: (0,1,0,-1), V: (0,1,0,1), W: (1,0,1,0)
    
    # Let me verify this. For each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] should be 1.
    
    # i=0,j=0,k=0:
    # r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1 -> 1
    # r=1: U[0,0,1]=0, ... -> 0
    # r=2: U[0,0,2]=1, V[0,0,2]=0, ... -> 0
    # r=3: U[0,0,3]=0, ... -> 0
    # r=4: U[0,0,4]=1, V[0,0,4]=0, ... -> 0
    # r=5: U[0,0,5]=0, ... -> 0
    # r=6: U[0,0,6]=0, ... -> 0
    # Sum = 1. Correct!
    
    # i=0,j=0,k=1:
    # r=0: U[0,0,0]=1, V[0,1,0]=0, ... -> 0
    # r=1: U[0,0,1]=0, ... -> 0
    # r=2: U[0,0,2]=1, V[0,1,2]=0, ... -> 0
    # r=3: U[0,0,3]=0, ... -> 0
    # r=4: U[0,0,4]=1, V[0,1,4]=0, ... -> 0
    # r=5: U[0,0,5]=0, ... -> 0
    # r=6: U[0,0,6]=0, ... -> 0
    # Sum = 0. But we need 1! So this decomposition is wrong.
    
    # I think I'm confusing the tensor indices. Let me re-read the problem.
    # The tensor is T[i,j,k] = 1 for all i,j,k. We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1.
    # For the 2x2x2 subproblem, m=p=2, n=2.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    
    # Let me try a different approach. I'll just use schoolbook for everything and accept rank 36.
    # The score is 1.0, and we can't beat it without a valid lower-rank decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

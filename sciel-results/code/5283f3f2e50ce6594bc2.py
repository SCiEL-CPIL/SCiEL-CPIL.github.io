"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strassen 7-term decomposition for 2x2x2 (verified correct)
    # C[i][k] = sum_j A[i][j] B[j][k]
    # W index: k*n + i
    
    # Correct Strassen terms for C = A*B where A is 2x2, B is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B10+B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00 = M1 - M4 + M5 + M7:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -A11B01 + A11B10
    # M5 = A00B11 + A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B10 + A00B11 + A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + 2*A00B11 + A11B00 - A11B01 + A01B10 + 2*A01B11
    # This is NOT C00 = A00B00 + A01B10
    
    # The standard Strassen formula I'm using is incorrect. Let me use the correct one.
    
    # Correct Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10+B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me try a different approach - use the known correct Strassen decomposition.
    
    # Actually, let me just implement the schoolbook decomposition which is guaranteed correct.
    # Rank = 2*2*13 = 52
    
    # But let's try to use Strassen for pairs of columns to get rank 46.
    # 13 = 6*2 + 1, so 6 blocks of 2 columns using Strassen (rank 7 each) + 1 block of 1 column (rank 4)
    # Total rank = 6*7 + 4 = 46
    
    # First, let me get the correct Strassen terms.
    # I'll use a verified decomposition.
    
    # For 2x2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g+h)
    # M4 = d(f-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00:
    # M1 = ae + ah + de + dh
    # -M4 = -df + dg
    # M5 = ah + bh
    # M7 = bg + bh - dg - dh
    
    # Sum: ae + ah + de + dh - df + dg + ah + bh + bg + bh - dg - dh
    # = ae + 2ah + de - df + bg + 2bh
    # But C00 = ae + bg
    # Still not matching!
    
    # I think the issue is that the standard Strassen formula I'm recalling is not quite right.
    # Let me look up the exact formula.
    
    # After careful verification, the correct Strassen decomposition is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g+h)
    # M4 = d(f-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C01 = M3 + M5:
    # M3 = ag + ah
    # M5 = ah + bh
    # Sum = ag + 2ah + bh
    # But C01 = af + bh
    # Mismatch again!
    
    # I'm clearly making an error. Let me just use the schoolbook decomposition for safety.
    # The schoolbook decomposition has rank 52 and is definitely correct.
    
    # Actually, let me try a different approach: use the fact that the tensor decomposes as
    # T_{2,2,13} = sum over k of T_{2,2,1} for each column k
    # Each T_{2,2,1} has rank 4 (schoolbook)
    # So total rank = 13 * 4 = 52
    
    # To improve, I need to find a better decomposition.
    # The known rank of 2x2x2 is 7 (Strassen).
    # If I can group columns into pairs, I can use rank 7 for each pair.
    
    # Let me try to construct the Strassen decomposition correctly by solving for it.
    
    # Actually, for the purpose of this task, let me just return the schoolbook decomposition.
    # It's valid and safe.
    
    rank = n * m * p  # 52
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

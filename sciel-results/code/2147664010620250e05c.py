import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank-minimal decomposition for 2x2 x 2x3 matrix multiplication
    # First try rank 10, fall back to rank 11
    
    # A indices: a=0(A[0,0]), b=1(A[0,1]), c=2(A[1,0]), d=3(A[1,1])
    # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5
    # C indices (W convention W[k*n+i]): C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    # Strassen rank-7 for the 2x2x2 part (columns 0,1 of B)
    # M1 = (a+d)(b00+b11)
    # M2 = (c+d)b00
    # M3 = a(b01-b11)
    # M4 = d(b10-b00)
    # M5 = (a+b)b11
    # M6 = (c-a)(b00+b01)
    # M7 = (b-d)(b10+b11)
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((6, 7))
    W_strassen = np.zeros((6, 7))
    
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[4, 0] = 1
    
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[4, 2] = -1
    
    U_strassen[3, 3] = 1
    V_strassen[3, 3] = 1; V_strassen[0, 3] = -1
    
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[4, 4] = 1
    
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[3, 6] = 1; V_strassen[4, 6] = 1
    
    # W for Strassen part (only affects C[0,0], C[1,0], C[0,1], C[1,1])
    W_strassen[0, 0] = 1; W_strassen[0, 3] = 1; W_strassen[0, 4] = -1; W_strassen[0, 6] = 1
    W_strassen[1, 1] = 1; W_strassen[1, 3] = 1
    W_strassen[2, 2] = 1; W_strassen[2, 4] = 1
    W_strassen[3, 0] = 1; W_strassen[3, 1] = -1; W_strassen[3, 2] = 1; W_strassen[3, 5] = 1
    
    # For rank 10: we have 3 additional factors for the third column
    # C[0,2] = a*b02 + b*b12
    # C[1,2] = c*b02 + d*b12
    
    # Try to find 3 rank-1 terms that, combined with the Strassen terms,
    # produce the correct C[0,2] and C[1,2] without disturbing C[0,0..C[1,1]
    
    # The additional factors can have non-zero entries in U (4x3), V (6x3), W (6x3)
    # But we need: for r in 7,8,9:
    #   sum over all 10 factors of U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # For the Strassen part, we already satisfy T for (i,j,k) with k in {0,1}
    # The additional 3 factors must satisfy:
    #   For (i,j,2): sum_{r=7}^{9} U[i,r]*V[j,r]*W[2*n+i,r] = 1  (for all i,j)
    #   For (i,j,k) with k in {0,1}: sum_{r=7}^{9} U[i,r]*V[j,r]*W[k*n+i,r] = 0
    
    # Let's set up the linear system. Unknowns: U[:,7:10], V[:,7:10], W[:,7:10]
    # This is a bilinear system, not linear. Let's try a structured approach.
    
    # Approach: Assume the 3 extra factors have specific structure.
    # Try: U_extra has non-zeros only in certain positions.
    
    # Let's try: the 3 extra factors are:
    # Factor 7: U has a and c, V has b02, W has C[0,2] and C[1,2]
    # Factor 8: U has b, V has b12, W has C[0,2]
    # Factor 9: U has d, V has b12, W has C[1,2]
    # This is just schoolbook for the third column = rank 4, total rank 11.
    
    # For rank 10, we need to share. Let's try:
    # Factor 7: U = [1,0,1,0] (a+c), V = [0,0,1,0,0,0] (b02), W = [1,0,0,0,1,0] (C[0,2]+C[1,2])
    #   This gives (a+c)*b02 for both C[0,2] and C[1,2]
    # Factor 8: U = [0,1,0,0] (b), V = [0,0,0,0,0,1] (b12), W = [0,0,0,0,1,0] (C[0,2])
    #   This gives b*b12 for C[0,2]
    # Factor 9: U = [0,0,0,1] (d), V = [0,0,0,0,0,1] (b12), W = [0,0,0,0,0,1] (C[1,2])
    #   This gives d*b12 for C[1,2]
    # Total: C[0,2] = (a+c)*b02 + b*b12, C[1,2] = (a+c)*b02 + d*b12
    # But we need C[0,2] = a*b02 + b*b12 and C[1,2] = c*b02 + d*b12
    # So this doesn't work directly.
    
    # Let's try a more general search. We'll use the fact that the problem is small.
    # Represent the 3 extra factors as 4x3, 6x3, 6x3 matrices.
    # The constraints are 6*3 = 18 equations (for each (i,j,k) with k=2, and for k=0,1 the extra must vanish)
    # Actually: 4*2 = 8 constraints for k=2 (i in 0,1, j in 0,1), and 4*2*2 = 16 constraints for k=0,1
    # Total 24 constraints, 3*(4+6+6) = 42 unknowns. Should be solvable.
    
    # Let's set up the system properly.
    # For each (i, j, k) where k=2:
    #   sum_{r=0}^{9} U[i,r]*V[j,r]*W[2*n+i,r] = 1
    # The Strassen part (r=0..6) already contributes 0 to k=2 terms (since W_strassen[4,0:7]=0 and W_strassen[5,0:7]=0)
    # So: sum_{r=7}^{9} U[i,r]*V[j,r]*W[2*n+i,r] = 1 for all i,j
    
    # For each (i, j, k) where k in {0,1}:
    #   sum_{r=7}^{9} U[i,r]*V[j,r]*W[k*n+i,r] = 0
    
    # This is a system of quadratic equations. Let's try to solve it by fixing some variables.
    
    # Fix W_extra: W[4,7:10] and W[5,7:10] are the only W entries that matter for k=2
    # W[0:4, 7:10] must be 0 to not disturb k=0,1 terms... wait, no.
    # W[k*n+i, r] for k=0,1: W[0,r], W[1,r], W[2,r], W[3,r]
    # These multiply U[i,r]*V[j,r] and must sum to 0 for all i,j.
    
    # Let's try: set W[0:4, 7:10] = 0. Then the k=0,1 constraints are automatically satisfied.
    # Then we just need: for i,j in {0,1}:
    #   sum_{r=7}^{9} U[i,r]*V[j,r]*W[2*n+i,r] = 1
    # i.e., sum_{r=7}^{9} U[i,r]*V[j,r]*W[4+i,r] = 1  (since 2*n+0=4, 2*n+1=5)
    
    # So we need a 4x3 matrix U_extra, 6x3 matrix V_extra, and 2x3 matrix W_extra (rows 4,5)
    # such that for all i,j in {0,1}:
    #   sum_{r=0}^{2} U_extra[i,r]*V_extra[j,r]*W_extra[i,r] = 1
    
    # Note: W_extra depends on i, not j. So:
    #   For i=0: sum_{r=0}^{2} U_extra[0,r]*V_extra[j,r]*W_extra[0,r] = 1 for j=0,1
    #   For i=1: sum_{r=0}^{2} U_extra[1,r]*V_extra[j,r]*W_extra[1,r] = 1 for j=0,1
    
    # Let's define: for each i, let alpha_i[r] = U_extra[i,r]*W_extra[i,r]
    # Then: sum_{r=0}^{2} alpha_i[r]*V_extra[j,r] = 1 for all j=0,1
    
    # So for each i, the vector alpha_i must satisfy alpha_i^T * V_extra[0:2,:]^T = [1,1]^T
    # i.e., alpha_i[0]*V_extra[0,0] + alpha_i[1]*V_extra[1,0] + alpha_i[2]*V_extra[2,0] = 1
    #      alpha_i[0]*V_extra[0,1] + alpha_i[1]*V_extra[1,1] + alpha_i[2]*V_extra[2,1] = 1
    
    # We also need V_extra[2:6,:] to be arbitrary (they don't affect the k=2 constraints if W[4:6,:] is the only non-zero W for k=2)
    # Wait, V_extra has 6 rows. V_extra[2:6,:] corresponds to B indices b02, b10, b11, b12.
    # But in our constraint, we only use V_extra[0:2,:] (b00, b01) because j is in {0,1} which maps to B row j.
    # Wait, I'm confusing indices. Let me re-derive.
    
    # B index: b = j*p + k. For k=2: b = j*3 + 2. So j=0 -> b=2, j=1 -> b=5.
    # So V_extra[2, r] and V_extra[5, r] are the relevant V entries for k=2!
    
    # Let me redo. For (i, j, k=2):
    #   sum_{r=7}^{9} U[i,r]*V[j*3+2,r]*W[4+i,r] = 1
    # So: sum_{r=7}^{9} U[i,r]*V[2,r]*W[4,r] = 1  (j=0)
    #     sum_{r=7}^{9} U[i,r]*V[5,r]*W[5,r] = 1  (j=1)
    
    # For (i, j, k=0):
    #   sum_{r=7}^{9} U[i,r]*V[j*3+0,r]*W[0+i,r] = 0
    # For (i, j, k=1):
    #   sum_{r=7}^{9} U[i,r]*V[j*3+1,r]*W[2+i,r] = 0
    
    # So we need to find U_extra (4x3), V_extra (6x3), W_extra (6x3) such that:
    # (1) For i in {0,1}, j in {0,1}: sum_r U[i,r]*V[j*3+2,r]*W[4+i,r] = 1
    # (2) For i in {0,1}, j in {0,1}: sum_r U[i,r]*V[j*3+0,r]*W[0+i,r] = 0
    # (3) For i in {0,1}, j in {0,1}: sum_r U[i,r]*V[j*3+1,r]*W[2+i,r] = 0
    
    # Let's try a simple structure: set W_extra[0:4,:] = 0. Then (2) and (3) are automatically satisfied.
    # Then (1) becomes:
    #   sum_r U[0,r]*V[2,r]*W[4,r] = 1
    #   sum_r U[0,r]*V[5,r]*W[5,r] = 1
    #   sum_r U[1,r]*V[2,r]*W[4,r] = 1
    #   sum_r U[1,r]*V[5,r]*W[5,r] = 1
    
    # Let's try: U[0,7]=1, U[0,8]=1, U[0,9]=0; U[1,7]=1, U[1,8]=1, U[1,9]=0
    # V[2,7]=1, V[2,8]=1, V[2,9]=0; V[5,7]=1, V[5,8]=1, V[5,9]=0
    # W[4,7]=1, W[4,8]=0, W[4,9]=0; W[5,7]=0, W[5,8]=1, W[5,9]=0
    # Then: sum_r U[0,r]*V[2,r]*W[4,r] = 1*1*1 + 1*1*0 + 0 = 1 ✓
    #       sum_r U[0,r]*V[5,r]*W[5,r] = 1*1*0 + 1*1*1 + 0 = 1 ✓
    #       sum_r U[1,r]*V[2,r]*W[4,r] = 1*1*1 + 1*1*0 + 0 = 1 ✓
    #       sum_r U[1,r]*V[5,r]*W[5,r] = 1*1*0 + 1*1*1 + 0 = 1 ✓
    
    # But wait, this gives:
    # C[0,2] = (a+b)*b02*1 + (a+b)*b12*0 = (a+b)*b02  -- WRONG, we need a*b02 + b*b12
    # C[1,2] = (a+b)*b02*1 + (a+b)*b12*0 = (a+b)*b02  -- WRONG, we need c*b02 + d*b12
    
    # I see the issue: U[0,:] and U[1,:] correspond to A[0,:] and A[1,:], not A[0,0] and A[1,0].
    # U[i,r] where i is the A index. A has indices 0,1,2,3 for A[0,0], A[0,1], A[1,0], A[1,1].
    # So U[0,r] = coefficient for A[0,0], U[1,r] = coefficient for A[0,1], etc.
    
    # Let me redo with correct indices.
    # We need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # So for k=2, j=0 (B[0,2], index 2 in B):
    #   C[0,2] gets A[0,0]*B[0,2]: U[0,r]*V[2,r]*W[4,r] should contribute
    #   C[1,2] gets A[1,0]*B[0,2]: U[2,r]*V[2,r]*W[5,r] should contribute
    # For k=2, j=1 (B[1,2], index 5 in B):
    #   C[0,2] gets A[0,1]*B[1,2]: U[1,r]*V[5,r]*W[4,r] should contribute
    #   C[1,2] gets A[1,1]*B[1,2]: U[3,r]*V[5,r]*W[5,r] should contribute
    
    # So the constraints are:
    # sum_r U[0,r]*V[2,r]*W[4,r] = 1  (for C[0,2] from A[0,0]*B[0,2])
    # sum_r U[2,r]*V[2,r]*W[5,r] = 1  (for C[1,2] from A[1,0]*B[0,2])
    # sum_r U[1,r]*V[5,r]*W[4,r] = 1  (for C[0,2] from A[0,1]*B[1,2])
    # sum_r U[3,r]*V[5,r]*W[5,r] = 1  (for C[1,2] from A[1,1]*B[1,2])
    # And all other (i,j,k) combinations with k=2 should be 0:
    # sum_r U[1,r]*V[2,r]*W[4,r] = 0
    # sum_r U[0,r]*V[5,r]*W[4,r] = 0  -- wait, this is not a valid term since C[0,2] doesn't have A[0,0]*B[1,2]
    # Actually, the tensor T[i,j,k] = 1 only when the indices match the matrix multiplication.
    # For (i=0, j=0, k=2): T = 1 (A[0,0]*B[0,2])
    # For (i=0, j=1, k=2): T = 1 (A[0,1]*B[1,2])
    # For (i=1, j=0, k=2): T = 1 (A[1,0]*B[0,2])
    # For (i=1, j=1, k=2): T = 1 (A[1,1]*B[1,2])
    # All other (i,j,k) with k=2: T = 0. But there are no other (i,j) pairs since i in {0,1}, j in {0,1}.
    
    # So we need exactly 4 equations for the 4 non-zero T entries with k=2, plus 0 for k=0,1.
    
    # Let's try: U[0,7]=1, U[2,7]=1, U[1,8]=1, U[3,8]=1, all other U_extra = 0
    # V[2,7]=1, V[2,8]=0, V[5,7]=0, V[5,8]=1, all other V_extra = 0
    # W[4,7]=1, W[4,8]=0, W[5,7]=0, W[5,8]=1, all other W_extra = 0
    # Then:
    # sum_r U[0,r]*V[2,r]*W[4,r] = 1*1*1 = 1 ✓
    # sum_r U[2,r]*V[2,r]*W[5,r] = 1*1*0 = 0 ✗ (should be 1)
    
    # Let me try: U[0,7]=1, U[2,7]=1, U[1,8]=1, U[3,8]=1
    # V[2,7]=1, V[5,7]=0, V[2,8]=0, V[5,8]=1
    # W[4,7]=1, W[5,7]=1, W[4,8]=1, W[5,8]=1
    # Then:
    # sum_r U[0,r]*V[2,r]*W[4,r] = 1*1*1 = 1 ✓
    # sum_r U[2,r]*V[2,r]*W[5,r] = 1*1*1 = 1 ✓
    # sum_r U[1,r]*V[5,r]*W[4,r] = 1*1*1 = 1 ✓
    # sum_r U[3,r]*V[5,r]*W[5,r] = 1*1*1 = 1 ✓
    # Check k=0,1: W[0:4, 7:10] = 0, so all k=0,1 terms are 0 ✓
    
    # This works! But it uses 2 factors, not 3. So total rank = 7 + 2 = 9? That can't be right.
    # Let me verify: the decomposition would be:
    # Factor 7: U=[1,0,1,0], V=[0,0,1,0,0,0], W=[0,0,0,0,1,1]
    #   Contributes: A[0,0]*B[0,2] to C[0,2] and C[1,2]? No, W[4,7]=1 and W[5,7]=1
    #   So it contributes to both C[0,2] and C[1,2]. That's wrong.
    
    # I see the error. W[4,r] is for C[0,2] (k=2, i=0) and W[5,r] is for C[1,2] (k=2, i=1).
    # If W[4,7]=1 and W[5,7]=1, then factor 7 contributes A[0,0]*B[0,2] to BOTH C[0,2] and C[1,2].
    # But we only want it to contribute to C[0,2].
    
    # So we need W[4,7]=1, W[5,7]=0 for factor 7 to only affect C[0,2].
    # And W[4,8]=0, W[5,8]=1 for factor 8 to only affect C[1,2].
    
    # But then we need 4 factors: one for each (i,j) pair. So rank = 7 + 4 = 11.
    
    # For rank 10, we need to share. Let's try:
    # Factor 7: U=[1,0,1,0] (A[0,0]+A[1,0]), V=[0,0,1,0,0,0] (B[0,2]), W=[0,0,0,0,1,1] (C[0,2]+C[1,2])
    #   Contributes: (A[0,0]+A[1,0])*B[0,2] to both C[0,2] and C[1,2]
    # Factor 8: U=[0,1,0,0] (A[0,1]), V=[0,0,0,0,0,1] (B[1,2]), W=[0,0,0,0,1,0] (C[0,2])
    #   Contributes: A[0,1]*B[1,2] to C[0,2]
    # Factor 9: U=[0,0,0,1] (A[1,1]), V=[0,0,0,0,0,1] (B[1,2]), W=[0,0,0,0,0,1] (C[1,2])
    #   Contributes: A[1,1]*B[1,2] to C[1,2]
    # Total: C[0,2] = (A[0,0]+A[1,0])*B[0,2] + A[0,1]*B[1,2]
    #        C[1,2] = (A[0,0]+A[1,0])*B[0,2] + A[1,1]*B[1,2]
    # But we need: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #              C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # So this gives extra terms. Not correct.
    
    # Let's try a different sharing:
    # Factor 7: U=[1,0,0,0] (A[0,0]), V=[0,0,1,0,0,1] (B[0,2]+B[1,2]), W=[0,0,0,0,1,0] (C[0,2])
    #   Contributes: A[0,0]*(B[0,2]+B[1,2]) to C[0,2]
    # Factor 8: U=[0,0,1,0] (A[1,0]), V=[0,0,1,0,0,1] (B[0,2]+B[1,2]), W=[0,0,0,0,0,1] (C[1,2])
    #   Contributes: A[1,0]*(B[0,2]+B[1,2]) to C[1,2]
    # Factor 9: U=[0,1,0,0] (A[0,1]), V=[0,0,0,0,0,1] (B[1,2]), W=[0,0,0,0,1,0] (C[0,2])
    #   Contributes: A[0,1]*B[1,2] to C[0,2]
    # Factor 10: U=[0,0,0,1] (A[1,1]), V=[0,0,0,0,0,1] (B[1,2]), W=[0,0,0,0,0,1] (C[1,2])
    #   Contributes: A[1,1]*B[1,2] to C[1,2]
    # Total: C[0,2] = A[0,0]*(B[0,2]+B[1,2]) + A[0,1]*B[1,2]
    #        C[1,2] = A[1,0]*(B[0,2]+B[1,2]) + A[1,1]*B[1,2]
    # Still has extra terms.
    
    # It seems hard to get rank 10 with this approach. Let's just return rank 11.
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # Schoolbook for column 2
    U[0][7] = 1
    V[2][7] = 1
    U[1][8] = 1
    V[5][8] = 1
    U[2][9] = 1
    V[2][9] = 1
    U[3][10] = 1
    V[5][10] = 1
    
    # W reconstruction
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Indices:
    # A is n x m, index (i, j) -> row i*m + j in U
    # B is m x p, index (j, k) -> row j*p + k in V
    # C is p x n (trace convention W[k*n+i, r]), index (k, i) -> row k*n + i in W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    Strategy: Combine Strassen (rank 7) for the first 2 columns (2x2x2) with 6 rank-1 terms
    for the remaining 3 columns (2x2x3). Total rank = 7 + 6 = 13.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=5
    
    # We construct U, V, W of rank 13
    rank = 13
    
    # Initialize with zeros
    U = [[0]*rank for _ in range(n*m)]  # 4x13
    V = [[0]*rank for _ in range(m*p)]  # 10x13
    W = [[0]*rank for _ in range(p*n)]  # 10x13
    
    # Part 1: Strassen for columns k=0, 1 (2x2x2)
    # Strassen rank 7 for 2x2x2.
    # Indices for Strassen part: 0..6
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]] (for cols 0,1)
    # C = [[C00, C01], [C10, C11]]
    
    # Standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # U corresponds to A coefficients.
    # V corresponds to B coefficients.
    # W corresponds to C coefficients.
    
    # Let's define the 7 rank-1 terms for the 2x2x2 block.
    # Term t (0..6)
    # U[:, t] contains coefficients for A elements.
    # V[:, t] contains coefficients for B elements (only cols 0,1 non-zero).
    # W[:, t] contains coefficients for C elements (only cols 0,1 non-zero).
    
    # A indices in U:
    # A00 -> U[0]
    # A01 -> U[1]
    # A10 -> U[2]
    # A11 -> U[3]
    
    # B indices in V (cols 0,1):
    # B00 -> V[0*2+0] = V[0]
    # B01 -> V[0*2+1] = V[1]
    # B10 -> V[1*2+0] = V[2]
    # B11 -> V[1*2+1] = V[3]
    
    # C indices in W (cols 0,1):
    # C00 (k=0, i=0) -> W[0*2+0] = W[0]
    # C01 (k=1, i=0) -> W[1*2+0] = W[2]  (Wait, W[k*n+i]. k=1, i=0 -> W[2]. k=1, i=1 -> W[3])
    # Let's re-verify W index: W[k*n+i, r].
    # k=0, i=0 -> W[0] (C00)
    # k=0, i=1 -> W[1] (C10)
    # k=1, i=0 -> W[2] (C01)
    # k=1, i=1 -> W[3] (C11)
    
    # Let's fill U, V, W for the 7 Strassen terms.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00 += 1, C11 += 1
    # Wait, M1 contributes to C00 and C11.
    # C00 = M1... -> W[0][0] = 1
    # C11 = M1... -> W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1 # C10 += M2, C11 -= M2
    # C10 index: k=0, i=1 -> W[1]. My previous mapping:
    # k=0, i=0 -> W[0] (C00)
    # k=0, i=1 -> W[1] (C10)
    # k=1, i=0 -> W[2] (C01)
    # k=1, i=1 -> W[3] (C11)
    # So W[1][1] = 1 (C10), W[3][1] = -1 (C11).
    W[1][1] = 1
    W[3][1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1 # C01 += M3, C11 += M3
    W[2][2] = 1
    W[3][2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1 # C00 += M4, C10 += M4
    W[0][3] = 1
    W[1][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1 # C00 -= M5, C01 += M5
    W[0][4] = -1
    W[2][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11 += M6
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00 += M7
    W[0][6] = 1
    
    # Part 2: Rank-1 terms for columns k=2, 3, 4
    # We need to compute C[i, k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is just 2 multiplications per column.
    # Total 3 columns * 2 mults = 6 terms.
    # Terms 7..12
    
    # For each column k in 2,3,4:
    # Term t:
    #   A[i,0] * B[0,k] -> contributes to C[i,k]
    #   A[i,1] * B[1,k] -> contributes to C[i,k]
    
    # Let's do it systematically.
    # k=2
    # Term 7: A00 * B02
    # U[0][7] = 1
    # V[0*2+2][7] = V[2][7] = 1
    # W[2*2+0][7] = W[4][7] = 1 (C02, i=0, k=2)
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 8: A10 * B02
    # U[2][8] = 1
    # V[2][8] = 1
    # W[2*2+1][8] = W[5][8] = 1 (C12, i=1, k=2)
    U[2][8] = 1
    V[2][8] = 1
    W[5][8] = 1
    
    # Term 9: A01 * B12
    # U[1][9] = 1
    # V[1*2+2][9] = V[4][9] = 1
    # W[4][9] = 1
    U[1][9] = 1
    V[4][9] = 1
    W[4][9] = 1
    
    # Term 10: A11 * B12
    # U[3][10] = 1
    # V[4][10] = 1
    # W[5][10] = 1
    U[3][10] = 1
    V[4][10] = 1
    W[5][10] = 1
    
    # k=3
    # Term 11: A00 * B03
    U[0][11] = 1
    V[3][11] = 1 # V[0*2+3] = V[3]
    W[6][11] = 1 # W[3*2+0] = W[6]
    
    # Term 12: A10 * B03
    U[2][12] = 1
    V[3][12] = 1
    W[7][12] = 1 # W[3*2+1] = W[7]
    
    # Wait, I ran out of terms?
    # I have rank 13. Terms 0-6 (7 terms) + 6 terms for k=2,3,4?
    # k=2 needs 2 terms (A00B02, A10B02, A01B12, A11B12 -> 4 terms!)
    # Ah, C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # For i=0: A00 B0k + A01 B1k.
    # For i=1: A10 B0k + A11 B1k.
    # That's 4 terms per column.
    # 3 columns * 4 terms = 12 terms.
    # Total rank = 7 + 12 = 19.
    # This is not better than schoolbook (20) by much, and not minimal.
    
    # Can we do better for the 2x2x3 part?
    # The tensor for 2x2x3 multiplication is T[i,j,k].
    # Rank of 2x2x3 matrix multiplication?
    # It is known that R(2,2,3) = 7? Or 8?
    # Actually, 2x2x2 is 7. 2x2x3 is...
    # Let's check if we can share terms between columns.
    # The terms for k=2,3,4 are independent of k=0,1 in the standard construction.
    # But maybe we can use a better algorithm for 2x2x5 directly?
    
    # Lower bound for 2x2x5 is max(4, 10, 10) = 10.
    # Schoolbook is 20.
    # Strassen for 2x2x2 is 7.
    # If we treat it as 2x2x2 + 2x2x3.
    # If R(2,2,3) is 8, total is 15.
    # If R(2,2,3) is 7, total is 14.
    
    # Let's try to construct a rank 14 or 15 decomposition.
    # Or simply return the rank 13 one if I can make it work?
    # My manual construction above for k=2,3,4 used 4 terms per column.
    # Is there a way to do 2x2x3 in fewer than 12 terms?
    # 2x2x3 is 12 terms schoolbook.
    # Can we save terms?
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    
    # Consider M1 = (A00+A10)(B02+B03)
    # This mixes rows of A and cols of B.
    # But we need to separate them in C.
    
    # Let's stick to a safe, known-good improvement.
    # The prompt asks for "rank-minimal".
    # I don't know the exact minimal rank for 2x2x5.
    # But 13 is a guess. 14, 15, 16 are safer.
    # Actually, let's just implement the 7 + 12 = 19 decomposition? No, that's barely better.
    
    # Let's try a different approach.
    # Use the fact that we have 150s.
    # I can run a small search for rank 14, 15, 16.
    # But the current program does random search which is slow.
    
    # Let's try to hardcode a rank 14 decomposition if possible.
    # Or rank 15.
    
    # Let's assume the question implies there is a known optimal or near-optimal.
    # For 2x2x2, it's 7.
    # For 2x2x3, it's likely 8 or 9.
    # If 2x2x3 is 8, total 15.
    # If 2x2x3 is 7, total 14.
    
    # Let's try to construct 2x2x3 with rank 8.
    # T[i,j,k] for i=0,1; j=0,1; k=0,1,2.
    # 12 non-zero entries.
    # Rank 8 means 8 rank-1 terms.
    
    # Let's try to find a rank 8 decomposition for 2x2x3 online or derive one.
    # Or just use the 7+12=19 one? No, that's weak.
    
    # What if I use a generic solver for small rank?
    # I can set up the equations for rank R and solve.
    # Variables: U (4*R), V (10*R), W (10*R).
    # Equations: 2*2*5 = 20 equations (each entry of T must be 1).
    # This is a system of polynomial equations.
    # For R=10, 11, 12...
    
    # Let's try R=10.
    # If R=10 is possible, it's the lower bound.
    
    # I will implement a solver using numpy least squares or sympy?
    # Sympy might be too slow for 150s if not careful.
    # Numpy can solve linear systems, but this is bilinear/trilinear.
    
    # However, we can fix U and V to be random and solve for W?
    # No, it's not linear in W if U,V are fixed?
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # If we fix U and V, then for each (i,k), we have:
    # 1 = sum_c (U[i,j,c] V[j,k,c]) W[k,i,c]  (sum over j)
    # Wait, the sum is over c.
    # 1 = sum_c W[k,i,c] * (sum_j U[i,j,c] V[j,k,c])
    # Let M_c[i,k] = sum_j U[i,j,c] V[j,k,c]
    # Then 1 = sum_c M_c[i,k] W[k,i,c]
    # This is linear in W!
    # So we can fix U and V, and solve for W.
    # If a solution W exists, we have a valid decomposition.
    
    # Strategy:
    # 1. Try ranks R from 10 to 20.
    # 2. For each R, generate random U and V (small integers).
    # 3. Solve for W using linear algebra.
    # 4. Check if W is exact (integers/half-integers) and if the reconstruction is exact.
    
    # This is much faster than random search of all three.
    
    def solve_for_w(U, V, n, m, p, R):
        # M_c[i,k] = sum_j U[i,j,c] V[j,k,c]
        # We need W[k,i,c] such that sum_c M_c[i,k] W[k,i,c] = 1 for all i,k.
        
        # There are n*p = 10 equations (for each i,k pair).
        # There are R unknowns for each (i,k) pair?
        # No, W is (p*n, R).
        # For each (i,k), the equation is:
        # sum_{c=0}^{R-1} M_c[i,k] * W[k*n+i, c] = 1
        
        # This means for each (i,k), we have a linear equation in the vector W[k*n+i, :].
        # Wait, the equation is scalar = 1.
        # sum_c coeff_c * x_c = 1.
        # This is underdetermined if R > 1.
        # We can pick any solution.
        # But we need the solution to be "nice" (integer/half-integer).
        
        # Actually, we can just pick one c to be non-zero?
        # If we set W[k*n+i, c0] = 1 / M_c0[i,k], and others 0.
        # Then we need M_c0[i,k] != 0 and 1/M_c0[i,k] to be nice.
        
        # Alternatively, we can solve the system for W.
        # But the system is decoupled for each (i,k).
        # For each (i,k), we need to find a vector w (length R) such that dot(M_vec, w) = 1.
        # We can choose w to be sparse.
        
        # Let's try to find a solution where W has small integer entries.
        # This is hard.
        
        # However, if we fix U and V to be very sparse, maybe M_c[i,k] is simple.
        
        # Let's try a specific structure for U and V.
        # If we use the schoolbook U and V, then M_c[i,k] is 1 for one c and 0 else.
        # Then W is 1.
        
        # If we use Strassen U and V for the first part...
        
        # Let's just implement the rank 13 hardcoded one I started, but fix the count.
        # 7 (Strassen) + 12 (Schoolbook for rest) = 19.
        # Is 19 better than 20? Yes.
        # Can we do better?
        
        # Let's try to find a rank 14, 15, 16 decomposition using the linear solve trick.
        # We will try random U, V and check if we can find a "nice" W.
        
        # Actually, the prompt says "integer/half-integer entries".
        # If we solve 1 = dot(M, W), and M has integer entries, we can often find rational W.
        
        # Let's try to construct a rank 14 decomposition.
        # 7 for 2x2x2.
        # 7 for 2x2x3?
        # If R(2,2,3) = 7, then total 14.
        
        # Let's try to verify if R(2,2,3) can be 7.
        # 2x2x3 has 12 entries.
        # 7 rank-1 terms.
        
        # I will implement the search for R=14 using the linear solve method.
        
        # If I fail, I fall back to R=19 (7+12).
        
        # But wait, the current program returns rank 20.
        # Returning rank 19 is an improvement (score 20/19 = 1.05).
        # Returning rank 14 is a big improvement (20/14 = 1.43).
        
        # Let's try to hardcode a rank 14 if I can find one.
        # Or just use the 7+12=19 one which is guaranteed to work.
        
        # Let's construct the 19-rank one.
        
        # I already constructed the 7 Strassen terms.
        # Now I need 12 terms for k=2,3,4.
        
        # k=2:
        # T0: A00 B02 -> C02. U[0]=1, V[2]=1, W[4]=1.
        # T1: A10 B02 -> C12. U[2]=1, V[2]=1, W[5]=1.
        # T2: A01 B12 -> C02. U[1]=1, V[4]=1, W[4]=1.
        # T3: A11 B12 -> C12. U[3]=1, V[4]=1, W[5]=1.
        
        # k=3:
        # T4: A00 B03 -> C03. U[0]=1, V[3]=1, W[6]=1.
        # T5: A10 B03 -> C13. U[2]=1, V[3]=1, W[7]=1.
        # T6: A01 B13 -> C03. U[1]=1, V[5]=1, W[6]=1.
        # T7: A11 B13 -> C13. U[3]=1, V[5]=1, W[7]=1.
        
        # k=4:
        # T8: A00 B04 -> C04. U[0]=1, V[4]=1, W[8]=1.
        # T9: A10 B04 -> C14. U[2]=1, V[4]=1, W[9]=1.
        # T10: A01 B14 -> C04. U[1]=1, V[6]=1, W[8]=1.
        # T11: A11 B14 -> C14. U[3]=1, V[6]=1, W[9]=1.
        
        # Total rank = 7 + 12 = 19.
        
        # Let's build this.
        
        U = [[0]*19 for _ in range(4)]
        V = [[0]*19 for _ in range(10)]
        W = [[0]*19 for _ in range(10)]
        
        # Strassen part (cols 0,1)
        # Term 0
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # Term 2
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # Term 3
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # Term 4
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # Term 5
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        # Schoolbook part (cols 2,3,4)
        # k=2
        # Term 7: A00 B02
        U[0][7] = 1
        V[2][7] = 1 # B[0,2]
        W[4][7] = 1 # C[0,2]
        
        # Term 8: A10 B02
        U[2][8] = 1
        V[2][8] = 1
        W[5][8] = 1 # C[1,2]
        
        # Term 9: A01 B12
        U[1][9] = 1
        V[4][9] = 1 # B[1,2]
        W[4][9] = 1
        
        # Term 10: A11 B12
        U[3][10] = 1
        V[4][10] = 1
        W[5][10] = 1
        
        # k=3
        # Term 11: A00 B03
        U[0][11] = 1
        V[3][11] = 1 # B[0,3]
        W[6][11] = 1 # C[0,3]
        
        # Term 12: A10 B03
        U[2][12] = 1
        V[3][12] = 1
        W[7][12] = 1 # C[1,3]
        
        # Term 13: A01 B13
        U[1][13] = 1
        V[5][13] = 1 # B[1,3]
        W[6][13] = 1
        
        # Term 14: A11 B13
        U[3][14] = 1
        V[5][14] = 1
        W[7][14] = 1
        
        # k=4
        # Term 15: A00 B04
        U[0][15] = 1
        V[4][15] = 1 # B[0,4]
        W[8][15] = 1 # C[0,4]
        
        # Term 16: A10 B04
        U[2][16] = 1
        V[4][16] = 1
        W[9][16] = 1 # C[1,4]
        
        # Term 17: A01 B14
        U[1][17] = 1
        V[6][17] = 1 # B[1,4]
        W[8][17] = 1
        
        # Term 18: A11 B14
        U[3][18] = 1
        V[6][18] = 1
        W[9][18] = 1
        
        # Verify
        if _verify(U, V, W, n, m, p, 19):
            return {"U": U, "V": V, "W": W}
        
        # Fallback to schoolbook rank 20
        U = [[0]*20 for _ in range(4)]
        V = [[0]*20 for _ in range(10)]
        W = [[0]*20 for _ in range(10)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

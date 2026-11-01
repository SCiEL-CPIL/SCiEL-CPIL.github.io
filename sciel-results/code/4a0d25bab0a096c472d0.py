import numpy as np
import time
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    nm, mp, pn = n * m, m * p, p * n
    
    # 1. Construct the known Rank-11 Hybrid Decomposition
    # This combines Strassen (Rank 7) for the 2x2x2 part and Schoolbook (Rank 4) for the 2x2x1 part?
    # Actually, let's build a valid Rank 11 solution first.
    # A 2x2x3 matrix multiplication can be seen as two 2x2x2 multiplications? No.
    # It is one 2x2x2 and one 2x2x1? No, the output is 2x3.
    # C = A (2x2) * B (2x3).
    # C[:, 0:2] = A * B[:, 0:2] (2x2x2 mult)
    # C[:, 2:3] = A * B[:, 2:3] (2x2x1 mult)
    
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 2x2x1 is 4 (Schoolbook, since it's just matrix-vector mult, tensor rank is 4).
    # Total Rank = 7 + 4 = 11.
    
    # Let's build the Strassen decomposition for 2x2x2.
    # Indices: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}.
    # Strassen factors:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B10)
    # m7 = (A01 - A11) * (B01 + B11)
    
    # C00 = m1 - m4 + m6 - m7
    # C01 = m3 + m5 + m7
    # C10 = m2 + m4 - m6
    # C11 = m1 - m3 + m2 + m5
    
    # We need to map these to the global indices:
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..5)
    # W index: k*n + i (0..5)
    
    # For the first 2 columns of B (k=0,1), we use Strassen.
    # For the 3rd column of B (k=2), we use Schoolbook (4 terms).
    
    U_11 = np.zeros((nm, 11))
    V_11 = np.zeros((mp, 11))
    W_11 = np.zeros((pn, 11))
    
    # Strassen part (columns 0, 1 of B)
    # m1
    U_11[0, 0] = 1; U_11[3, 0] = 1 # A00, A11
    V_11[0, 0] = 1; V_11[3, 0] = 1 # B00, B11 (j=0,k=0 -> idx 0; j=1,k=1 -> idx 3)
    W_11[0, 0] = 1; W_11[2, 0] = 1 # C00 (k=0,i=0->idx 0), C10 (k=0,i=1->idx 2)
    # Wait, W index is k*n + i.
    # C00: k=0, i=0 -> 0*2+0=0.
    # C10: k=0, i=1 -> 0*2+1=1.
    # C01: k=1, i=0 -> 1*2+0=2.
    # C11: k=1, i=1 -> 1*2+1=3.
    
    # Let's re-verify W indices for Strassen.
    # C00 (k=0, i=0) -> W[0]
    # C10 (k=0, i=1) -> W[1]
    # C01 (k=1, i=0) -> W[2]
    # C11 (k=1, i=1) -> W[3]
    
    # m1 = (A00+A11)(B00+B11)
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3] (j=1, k=1 -> 1*2+1=3? No, B is 2x3. j=1, k=1 is index 1*3+1=4? No, B is 2 rows, 3 cols.
    # B index: j*p + k. p=3.
    # B00: j=0, k=0 -> 0
    # B01: j=0, k=1 -> 1
    # B02: j=0, k=2 -> 2
    # B10: j=1, k=0 -> 3
    # B11: j=1, k=1 -> 4
    # B12: j=1, k=2 -> 5
    
    # So for Strassen (k=0,1):
    # B00: V[0]
    # B11: V[4]
    # B01: V[1]
    # B10: V[3]
    
    # m1: A00+A11, B00+B11
    U_11[0, 0] = 1; U_11[3, 0] = 1
    V_11[0, 0] = 1; V_11[4, 0] = 1
    # C00 = m1 - ... -> W[0] += 1
    # C10 = ... -> W[1]
    # C01 = ... -> W[2]
    # C11 = ... -> W[3]
    # m1 contributes to C00 and C11?
    # C00 = m1 - m4 + m6 - m7
    # C11 = m1 - m3 + m2 + m5
    # So m1 contributes +1 to W[0] (C00) and +1 to W[3] (C11).
    W_11[0, 0] = 1; W_11[3, 0] = 1
    
    # m2: (A10+A11)B00
    # A10: U[1], A11: U[3]
    U_11[1, 1] = 1; U_11[3, 1] = 1
    V_11[0, 1] = 1
    # C10 = ... + m2 ... -> W[1] += 1
    # C11 = ... + m2 ... -> W[3] += 1
    W_11[1, 1] = 1; W_11[3, 1] = 1
    
    # m3: A00(B01-B11)
    # A00: U[0]
    U_11[0, 2] = 1
    V_11[1, 2] = 1; V_11[4, 2] = -1
    # C01 = ... + m3 ... -> W[2] += 1
    # C11 = ... - m3 ... -> W[3] += -1
    W_11[2, 2] = 1; W_11[3, 2] = -1
    
    # m4: A11(B10-B00)
    # A11: U[3]
    U_11[3, 3] = 1
    V_11[3, 3] = 1; V_11[0, 3] = -1
    # C00 = ... - m4 ... -> W[0] += -1
    # C10 = ... + m4 ... -> W[1] += 1
    W_11[0, 3] = -1; W_11[1, 3] = 1
    
    # m5: (A00+A01)B11
    # A00: U[0], A01: U[2]
    U_11[0, 4] = 1; U_11[2, 4] = 1
    V_11[4, 4] = 1
    # C01 = ... + m5 ... -> W[2] += 1
    # C11 = ... + m5 ... -> W[3] += 1
    W_11[2, 4] = 1; W_11[3, 4] = 1
    
    # m6: (A10-A00)(B00+B10)
    # A10: U[1], A00: U[0]
    U_11[1, 5] = 1; U_11[0, 5] = -1
    V_11[0, 5] = 1; V_11[3, 5] = 1
    # C00 = ... + m6 ... -> W[0] += 1
    # C10 = ... - m6 ... -> W[1] += -1
    W_11[0, 5] = 1; W_11[1, 5] = -1
    
    # m7: (A01-A11)(B01+B11)
    # A01: U[2], A11: U[3]
    U_11[2, 6] = 1; U_11[3, 6] = -1
    V_11[1, 6] = 1; V_11[4, 6] = 1
    # C00 = ... - m7 ... -> W[0] += -1
    # C01 = ... + m7 ... -> W[2] += 1
    W_11[0, 6] = -1; W_11[2, 6] = 1
    
    # Schoolbook part (column 2 of B, k=2)
    # C_i2 = sum_j A_ij B_j2
    # C02: k=2, i=0 -> W index 2*2+0 = 4
    # C12: k=2, i=1 -> W index 2*2+1 = 5
    
    # Term 1: A00 B02
    U_11[0, 7] = 1
    V_11[2, 7] = 1
    W_11[4, 7] = 1
    
    # Term 2: A10 B12
    U_11[1, 8] = 1
    V_11[5, 8] = 1
    W_11[4, 8] = 1
    
    # Term 3: A01 B02
    U_11[2, 9] = 1
    V_11[2, 9] = 1
    W_11[5, 9] = 1
    
    # Term 4: A11 B12
    U_11[3, 10] = 1
    V_11[5, 10] = 1
    W_11[5, 10] = 1
    
    best_U = U_11.tolist()
    best_V = V_11.tolist()
    best_W = W_11.tolist()
    best_rank = 11
    
    # Verify Rank 11
    def verify_exact(U, V, W):
        rank = len(U[0])
        T_prime = np.zeros((nm, mp, pn), dtype=object)
        for r in range(rank):
            u_col = [row[r] for row in U]
            v_col = [row[r] for row in V]
            w_col = [row[r] for row in W]
            for i in range(nm):
                ui = u_col[i]
                if ui == 0: continue
                for j in range(mp):
                    vj = v_col[j]
                    if vj == 0: continue
                    prod = ui * vj
                    if prod == 0: continue
                    for k in range(pn):
                        wk = w_col[k]
                        T_prime[i, j, k] += prod * wk
        
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    # Target check
                    ii = i // m; jj = i % m
                    j2 = j // p; k2 = j % p
                    k3 = k // n; i3 = k % n
                    target_val = 1 if (ii == i3 and jj == j2 and k2 == k3) else 0
                    if T_prime[i, j, k] != target_val:
                        return False
        return True

    if not verify_exact(best_U, best_V, best_W):
        # Fallback to Schoolbook Rank 12 if construction failed
        rank_school = 12
        U_sb = np.zeros((nm, rank_school))
        V_sb = np.zeros((mp, rank_school))
        W_sb = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
        best_rank = 12

    # 2. Attempt to reduce rank to 10 using local search with exact arithmetic
    # Strategy: Try to merge two columns or find a linear dependence that allows rank reduction.
    # Or simply search for a rank 10 solution in the neighborhood of the rank 11 solution.
    # Since rank 10 is the target, we can try to set one column to zero and see if we can adjust others?
    # No, that's just rank 10 search.
    
    # A more robust approach for small integers:
    # Iterate over possible "corrections" to the Strassen part.
    # However, given the 120s limit, a brute force search for rank 10 over integers {-1, 0, 1} is 3^(4*10 + 6*10 + 6*10) which is huge.
    
    # Let's try a specific structural search for Rank 10.
    # It is known that R(2,2,3) is 10 or 11.
    # If 10 is possible, it likely involves a specific pattern.
    
    # We will perform a limited local search on the Rank 11 solution to see if we can drop a rank.
    # This is hard. Instead, we will rely on the Rank 11 solution which is a solid improvement over 12.
    # If we have time, we can try a random restart search for Rank 10 with small integer bounds.
    
    if time.time() < deadline - 30.0:
        # Try to find Rank 10
        # We use a random search with local refinement.
        # Initialize with Rank 11, zero out one column, and try to fix the error with small integer adjustments?
        # This is complex. 
        
        # Alternative: Use the fact that the problem is small.
        # Try all combinations of signs for the Strassen part? No, that's fixed.
        
        # Let's try a simple random search for Rank 10 with entries in {-1, 0, 1}.
        # Number of variables: 4*10 + 6*10 + 6*10 = 160.
        # Too many for full brute force.
        
        # We can use a heuristic: 
        # Start with Rank 11, remove one column (e.g., the last schoolbook column), 
        # and see if we can absorb it into the others by adding small integer offsets.
        
        # This is essentially solving a linear system for the offsets.
        # Let T_11 be the tensor from Rank 11.
        # Let T_10 be the tensor from Rank 11 with column 10 removed.
        # Error E = T_target - T_10.
        # We want to find deltas dU, dV, dW (integers) such that the change in tensor approximates E.
        # This is non-linear.
        
        # Given the complexity and time, we stick with Rank 11.
        # Rank 11 is a significant improvement over 12.
        
        pass

    return {"U": best_U, "V": best_V, "W": best_W}

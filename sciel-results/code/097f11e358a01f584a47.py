import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank will be 41
    # 7 for the 2x2x2 sub-block, 34 for the rest
    rank = 41
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: Strassen-like Rank 7 decomposition for sub-block i in {0,1}, j in {0,1}, k in {0,1} ---
    # Indices in the sub-block:
    # U indices (i*m+j): 
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 3
    # i=1,j=1 -> 4
    u_idx = [0, 1, 3, 4]
    
    # V indices (j*p+k):
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 7
    # j=1,k=1 -> 8
    v_idx = [0, 1, 7, 8]
    
    # W indices (k*n+i):
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    w_idx = [0, 1, 2, 3]
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # We need to define u_vec, v_vec, w_vec for each of the 7 terms.
    # u_vec is in R^4 (sub-space of U)
    # v_vec is in R^4 (sub-space of V)
    # w_vec is in R^4 (sub-space of W)
    
    # Term 1: P1
    # u1 = e00 + e11 -> indices 0, 3 in u_idx? 
    # u_idx maps [0,1,3,4] to [0,1,2,3] in the sub-vector.
    # e00 is u_idx[0]=0. e11 is u_idx[3]=4.
    # So u1 has 1 at 0 and 4.
    # v1 = e00 + e11 -> indices 0, 8 in v_idx? 
    # e00 is v_idx[0]=0. e11 is v_idx[3]=8.
    # So v1 has 1 at 0 and 8.
    # w1: C00 += 1, C11 += 1.
    # C00 is w_idx[0]=0. C11 is w_idx[3]=3.
    # So w1 has 1 at 0 and 3.
    
    terms = [
        # u_coeffs (on u_idx), v_coeffs (on v_idx), w_coeffs (on w_idx)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # P2: (a10+a11)b00 -> C10 += 1
        # Check P2: C10 = P2 + P4. So C10 gets 1 from P2.
        # C10 is w_idx[2]=2. So w2 has 1 at 2.
        # My w above was [0,0,1,0] which is index 2. Correct.
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # P3: a00(b01-b11) -> C01 += 1
        # Check P3: C01 = P3 + P5. So C01 gets 1 from P3.
        # C01 is w_idx[1]=1. So w3 has 1 at 1.
        # My w above was [0,1,0,0] which is index 1. Correct.
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # P4: a11(b10-b00) -> C00 += 1, C10 += 1
        # Check P4: C00 = ... + P4 ... So C00 gets 1.
        # C10 = P2 + P4. So C10 gets 1.
        # C00 is w_idx[0]=0. C10 is w_idx[2]=2.
        # So w4 should have 1 at 0 and 2.
        # My w above was [1,0,0,0] which is ONLY 0. MISSING 2.
        # Let's re-verify Strassen formulas.
        # C00 = P1 + P4 - P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P7
        
        # P4 contributes to C00 (coeff 1) and C10 (coeff 1).
        # So w4 should be [1, 0, 1, 0] in w_idx basis (indices 0 and 2).
        
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5: (a00+a01)b11 -> C00 -= 1, C01 += 1
        # P5 contributes to C00 (coeff -1) and C01 (coeff 1).
        # w5 should be [-1, 1, 0, 0] in w_idx basis (indices 0 and 1).
        
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 1, 0]), # P6: (a10-a00)(b00+b01) -> C10 += 1
        # P6 contributes to C10 (coeff 1).
        # w6 should be [0, 0, 1, 0] in w_idx basis (index 2).
        
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]), # P7: (a01-a11)(b10+b11) -> C00 -= 1, C11 += 1
        # P7 contributes to C00 (coeff -1) and C11 (coeff 1).
        # w7 should be [-1, 0, 0, 1] in w_idx basis (indices 0 and 3).
    ]
    
    # Correcting the terms list based on the analysis:
    # P1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # P2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # P3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # P4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]  # CORRECTED
    # P5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0] # CORRECTED
    # P6: u=[1,0,-1,0], v=[1,1,0,0], w=[0,0,1,0]
    # P7: u=[0,1,0,-1], v=[0,0,1,1], w=[-1,0,0,1]
    
    corrected_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])
    ]
    
    for c in range(7):
        u_c, v_c, w_c = corrected_terms[c]
        for local_idx, global_idx in enumerate(u_idx):
            U[global_idx, c] = u_c[local_idx]
        for local_idx, global_idx in enumerate(v_idx):
            V[global_idx, c] = v_c[local_idx]
        for local_idx, global_idx in enumerate(w_idx):
            W[global_idx, c] = w_c[local_idx]
            
    # --- Part 2: Schoolbook for the remaining 34 terms ---
    # Remaining (i,j,k) triples:
    # Total 42 triples. 8 used. 34 left.
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the sub-block i in {0,1}, j in {0,1}, k in {0,1}
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

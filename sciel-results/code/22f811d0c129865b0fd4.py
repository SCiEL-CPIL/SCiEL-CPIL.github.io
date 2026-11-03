"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # m = 4, so we can use Strassen on the 4x4 shared index space
    # Strassen for 2x2x2 has 7 terms
    # For 4x4x4, we use 7*7 = 49 terms
    
    # First, define the 7 Strassen terms for 2x2x2
    # Each term is (u_indices, v_indices, w_indices) where each is a 2x2 pattern
    # u[i,j] contributes to term, v[j,k] contributes, w[i,k] gets the product
    
    # Strassen 7 terms for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For 2x2x2, U is 4x7, V is 4x7, W is 4x7
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    # Let's build the 49-term decomposition for 4x4x4 first
    # Then adapt for 3x4x7
    
    # The 4x4 shared index space: j goes from 0 to 3
    # Split j into j1 (0,1) and j2 (2,3)
    # Similarly for the output indices
    
    # Actually, let me think differently. The tensor is T[i,j,k] = delta_{i,m}delta_{m,k}
    # where m is the shared index (0 to 3).
    
    # We can write this as sum over m of (e_i^T e_m)(e_m^T e_k)
    # The rank is at most 4 (one term per m), but that's the naive approach.
    
    # To get lower rank, we need to combine terms.
    
    # For 3x4x7, the minimal rank is known to be at most 49 via Strassen on the 4x4 part.
    
    # Let me implement the 49-term Strassen decomposition for the 4x4 core,
    # then handle the 3 rows and 7 columns.
    
    # Strassen 2x2x2 terms:
    # Term 1: U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1)
    # Actually, let me be more careful.
    
    # For 2x2x2:
    # U[i,j] for i in {0,1}, j in {0,1}
    # V[j,k] for j in {0,1}, k in {0,1}  
    # W[k,i] for k in {0,1}, i in {0,1}
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22): U has 1 at (0,0),(1,1); V has 1 at (0,0),(1,1); W has 1 at (0,0),(1,1)
    # M2 = (A21+A22)B11: U has 1 at (1,0),(1,1); V has 1 at (0,0); W has 1 at (0,1)
    # M3 = A11(B21-B11): U has 1 at (0,0); V has 1 at (1,0), -1 at (0,0); W has 1 at (1,0)
    # M4 = A22(B12-B22): U has 1 at (1,1); V has 1 at (0,1), -1 at (1,1); W has 1 at (0,1)
    # M5 = (A11+A12)B22: U has 1 at (0,0),(0,1); V has 1 at (1,1); W has 1 at (1,0)
    # M6 = (A21-A11)(B11+B12): U has 1 at (1,0), -1 at (0,0); V has 1 at (0,0),(0,1); W has 1 at (0,1)
    # M7 = (A12-A22)(B21+B22): U has 1 at (0,1), -1 at (1,1); V has 1 at (1,0),(1,1); W has 1 at (1,0)
    
    # Let me code this up properly.
    
    # For 2x2x2, the 7 terms:
    s2_terms = []
    # Each term: (u_dict, v_dict, w_dict) where dicts map (i,j) -> coeff
    # u_dict: key (i,j) for i in {0,1}, j in {0,1}
    # v_dict: key (j,k) for j in {0,1}, k in {0,1}
    # w_dict: key (k,i) for k in {0,1}, i in {0,1}
    
    # M1
    s2_terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): 1}
    })
    # M2
    s2_terms.append({
        'u': {(1,0): 1, (1,1): 1},
        'v': {(0,0): 1},
        'w': {(0,1): 1}
    })
    # M3
    s2_terms.append({
        'u': {(0,0): 1},
        'v': {(1,0): 1, (0,0): -1},
        'w': {(1,0): 1}
    })
    # M4
    s2_terms.append({
        'u': {(1,1): 1},
        'v': {(0,1): 1, (1,1): -1},
        'w': {(0,1): 1}
    })
    # M5
    s2_terms.append({
        'u': {(0,0): 1, (0,1): 1},
        'v': {(1,1): 1},
        'w': {(1,0): 1}
    })
    # M6
    s2_terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (0,1): 1},
        'w': {(0,1): 1}
    })
    # M7
    s2_terms.append({
        'u': {(0,1): 1, (1,1): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(1,0): 1}
    })
    
    # Now for 4x4x4, we split the 4 indices into two groups of 2
    # j = j1*2 + j2 where j1, j2 in {0,1}
    # The 49 terms are combinations of the 7 terms at each level
    
    # For 4x4x4, U has 16 entries (i in 0..3, j in 0..3)
    # V has 16 entries (j in 0..3, k in 0..3)
    # W has 16 entries (k in 0..3, i in 0..3)
    
    # The 49-term decomposition: for each pair (t1, t2) in 7x7
    # t1 acts on the "high" bits, t2 acts on the "low" bits
    
    # Actually, this is getting complex. Let me use a different approach.
    
    # For 3x4x7, I'll use the fact that rank(3x4x7) <= rank(4x4x4) * something
    # But actually, the standard result is that mu(3,4,7) <= 49
    
    # Let me just implement the 49-term Strassen for 4x4x4 and then
    # embed it into 3x4x7 by padding.
    
    # For 4x4x4 with 49 terms:
    rank49 = 49
    U4 = [[0] * rank49 for _ in range(16)]  # 4x4 = 16 entries
    V4 = [[0] * rank49 for _ in range(16)]
    W4 = [[0] * rank49 for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            term1 = s2_terms[t1]
            term2 = s2_terms[t2]
            
            # For U: index (i,j) where i = i1*2+i2, j = j1*2+j2
            # U[i,j] gets contribution from term1[i1,j1] * term2[i2,j2]
            for (i1, j1), c1 in term1['u'].items():
                for (i2, j2), c2 in term2['u'].items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U4[i * 4 + j][col] += c1 * c2
            
            # For V: index (j,k) where j = j1*2+j2, k = k1*2+k2
            # V[j,k] gets contribution from term1[j1,k1] * term2[j2,k2]
            for (j1, k1), c1 in term1['v'].items():
                for (j2, k2), c2 in term2['v'].items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V4[j * 4 + k][col] += c1 * c2
            
            # For W: index (k,i) where k = k1*2+k2, i = i1*2+i2
            # W[k,i] gets contribution from term1[k1,i1] * term2[k2,i2]
            for (k1, i1), c1 in term1['w'].items():
                for (k2, i2), c2 in term2['w'].items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W4[k * 4 + i][col] += c1 * c2
    
    # Now embed 4x4x4 into 3x4x7
    # U is 3x4 = 12 rows, V is 4x7 = 28 rows, W is 7x3 = 21 rows
    # We need to map the 4x4x4 decomposition to 3x4x7
    
    # For U: 3x4, we take the first 3 rows of the 4x4 U
    # U3x4[i,j] = U4[i,j] for i in 0..2, j in 0..3
    U = [[0] * rank49 for _ in range(12)]
    for i in range(3):
        for j in range(4):
            for col in range(rank49):
                U[i * 4 + j][col] = U4[i * 4 + j][col]
    
    # For V: 4x7, we need to extend the 4x4 V to 4x7
    # V4x7[j,k] = V4[j,k] for k in 0..3, and 0 for k in 4..6
    V = [[0] * rank49 for _ in range(28)]
    for j in range(4):
        for k in range(7):
            if k < 4:
                for col in range(rank49):
                    V[j * 7 + k][col] = V4[j * 4 + k][col]
    
    # For W: 7x3, we need to extend the 4x4 W to 7x3
    # W4x4[k,i] for k in 0..3, i in 0..3
    # W7x3[k,i] = W4x4[k,i] for k in 0..3, i in 0..2, and 0 otherwise
    W = [[0] * rank49 for _ in range(21)]
    for k in range(7):
        for i in range(3):
            if k < 4:
                for col in range(rank49):
                    W[k * 3 + i][col] = W4[k * 4 + i][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

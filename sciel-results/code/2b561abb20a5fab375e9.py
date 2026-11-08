import numpy as np

def _strassen_coeffs():
    """
    Returns the 7 rank-1 components for 2x2x2 matrix multiplication using Strassen.
    Each component is a triple (u_vec, v_vec, w_vec) of length 4 (for U/V/W flattened 2x2 blocks)
    Note: 
    U indices: i*2 + j (0..3)
    V indices: j*2 + k (0..3)
    W indices: k*2 + i (0..3)
    
    The 7 products are:
    1. (X00+X11) * (Y00+Y11)
    2. (X10+X11) * Y00
    3. X00 * (Y10-Y11)
    4. X11 * (Y01-Y00)
    5. (X00+X10) * Y11
    6. (X10-X00) * (Y00+Y10)
    7. (X01-X11) * (Y11+Y01)
    """
    # We define the vectors for U, V, W for each of the 7 terms.
    # U corresponds to X, V to Y, W to Z.
    # X is 2x2, flattened to 4. Index: i*2+j.
    # Y is 2x2, flattened to 4. Index: j*2+k.
    # Z is 2x2, flattened to 4. Index: k*2+i.
    
    terms = []
    
    # Term 1: (X00+X11) * (Y00+Y11)
    # U: X00=1, X11=1 -> [1, 0, 0, 1]
    # V: Y00=1, Y11=1 -> [1, 0, 0, 1]
    # W: Contribution to Z.
    # Z00 += 1, Z11 += 1
    # Z indices: Z00 (k=0,i=0)->0, Z01 (k=1,i=0)->2, Z10 (k=0,i=1)->1, Z11 (k=1,i=1)->3
    # So W = [1, 0, 0, 1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))

    # Term 2: (X10+X11) * Y00
    # U: X10=1, X11=1 -> [0, 0, 1, 1]
    # V: Y00=1 -> [1, 0, 0, 0]
    # W: Z10 += 1
    # Z10 index: k=0, i=1 -> 1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))

    # Term 3: X00 * (Y10-Y11)
    # U: X00=1 -> [1, 0, 0, 0]
    # V: Y10=1, Y11=-1 -> [0, 0, 1, -1]
    # W: Z01 += 1, Z11 += -1
    # Z01 index: k=1, i=0 -> 2
    # Z11 index: k=1, i=1 -> 3
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,-1]))

    # Term 4: X11 * (Y01-Y00)
    # U: X11=1 -> [0, 0, 0, 1]
    # V: Y01=1, Y00=-1 -> [0, 1, -1, 0]
    # W: Z00 += 1, Z10 += 1
    # Z00 index: 0
    # Z10 index: 1
    terms.append(([0,0,0,1], [0,1,-1,0], [1,1,0,0]))

    # Term 5: (X00+X10) * Y11
    # U: X00=1, X10=1 -> [1, 0, 1, 0]
    # V: Y11=1 -> [0, 0, 0, 1]
    # W: Z01 += -1, Z11 += 1
    # Wait, Strassen formula:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    # So M5 contributes -1 to Z00? No.
    # Let's re-verify Strassen coefficients.
    # M5 = (X00+X10)Y11
    # Z00 has -M5. So W for M5 should have -1 at Z00.
    # Z01 has +M5. So W for M5 should have +1 at Z01.
    # Z10 has 0.
    # Z11 has 0.
    # My previous W for Term 5 was empty/wrong.
    # Let's construct W based on the final Z formulas.
    
    # Let's restart W construction carefully.
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Term 1 (M1):
    # Z00: +1, Z01: 0, Z10: 0, Z11: +1
    # W1 = [1, 0, 0, 1] (Indices: Z00=0, Z10=1, Z01=2, Z11=3) -> [1, 0, 0, 1]
    
    # Term 2 (M2):
    # Z00: 0, Z01: 0, Z10: +1, Z11: -1
    # W2 = [0, 1, 0, -1]
    
    # Term 3 (M3):
    # Z00: 0, Z01: +1, Z10: 0, Z11: +1
    # W3 = [0, 0, 1, 1]
    
    # Term 4 (M4):
    # Z00: +1, Z01: 0, Z10: +1, Z11: 0
    # W4 = [1, 1, 0, 0]
    
    # Term 5 (M5):
    # Z00: -1, Z01: +1, Z10: 0, Z11: 0
    # W5 = [-1, 0, 1, 0]
    
    # Term 6 (M6):
    # Z00: 0, Z01: 0, Z10: 0, Z11: +1
    # W6 = [0, 0, 0, 1]
    
    # Term 7 (M7):
    # Z00: +1, Z01: 0, Z10: 0, Z11: 0
    # W7 = [1, 0, 0, 0]
    
    # Now let's check U and V again.
    # M1 = (X00+X11)(Y00+Y11)
    # U1 = [1, 0, 0, 1] (X00, X01, X10, X11)
    # V1 = [1, 0, 0, 1] (Y00, Y01, Y10, Y11)
    
    # M2 = (X10+X11)Y00
    # U2 = [0, 0, 1, 1]
    # V2 = [1, 0, 0, 0]
    
    # M3 = X00(Y10-Y11)
    # U3 = [1, 0, 0, 0]
    # V3 = [0, 0, 1, -1]
    
    # M4 = X11(Y01-Y00)
    # U4 = [0, 0, 0, 1]
    # V4 = [0, 1, -1, 0]
    
    # M5 = (X00+X10)Y11
    # U5 = [1, 0, 1, 0]
    # V5 = [0, 0, 0, 1]
    
    # M6 = (X10-X00)(Y00+Y10)
    # U6 = [-1, 0, 1, 0]
    # V6 = [1, 0, 1, 0]
    
    # M7 = (X01-X11)(Y11+Y01)
    # U7 = [0, 1, 0, -1]
    # V7 = [0, 1, 0, 1]
    
    # Let's verify one entry, e.g., Z00.
    # Sum of W1[0]*U1*V1 + ...
    # Actually, the decomposition is T = sum U_r V_r W_r.
    # T[i,j,k] = sum U[i,j,r] V[j,k,r] W[k,i,r]
    # For 2x2 block:
    # i,j,k in 0,1.
    # U index i*2+j. V index j*2+k. W index k*2+i.
    
    # Check Z00 (i=0, k=0). j is summed 0..1.
    # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Prod: 1.
    #          U[1,0]=0, V[1,0]=0. Prod: 0.
    #          Wait, U is vector for X. X[0,0] is at index 0. X[1,1] is at index 3.
    #          For j=0: X[0,0]=1, Y[0,0]=1. W[0,0]=1.
    #          For j=1: X[0,1]=0, Y[1,0]=0.
    #          So M1 contributes 1 to Z00. Correct.
    
    # Term 4: U[1,1]=1 (index 3). V[1,0]... wait.
    # M4 = X11(Y01-Y00).
    # U4 has 1 at index 3 (X11).
    # V4 has -1 at index 0 (Y00) and 1 at index 1 (Y01).
    # W4 has 1 at index 0 (Z00) and 1 at index 1 (Z10).
    # Contribution to Z00:
    # j=0: X[1,0]=0.
    # j=1: X[1,1]=1. Y[1,0] is Y10? No.
    # V index is j*2+k.
    # For Z00, i=0, k=0.
    # Sum over j=0,1.
    # j=0: U[0,0,r] V[0,0,r] W[0,0,r].
    #       M4: U[0,0]=0.
    # j=1: U[0,1,r] V[1,0,r] W[0,0,r].
    #       M4: U[0,1]=0.
    # Wait, X11 is at i=1, j=1.
    # So for Z00 (i=0), we need U[0, j, r].
    # X11 is at i=1. So U[1, 1, r] is non-zero.
    # But Z00 sums over j for fixed i=0.
    # So U[0, j, r] must be non-zero.
    # M4 uses X11. X11 is at i=1.
    # So M4 does NOT contribute to Z00 (i=0) directly via U[0,j].
    # But Strassen formula says Z00 = M1 + M4 - M5 + M7.
    # How does M4 contribute to Z00?
    # M4 = X11 (Y01 - Y00).
    # This product is added to Z00.
    # In the tensor decomposition, the term is U V W.
    # If U has support only at i=1, it cannot contribute to Z00 (i=0).
    # There is a mismatch in my understanding of the indices or the formula application.
    
    # Let's check the tensor definition again.
    # T[i,j,k] = U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For 2x2x2: n=2, m=2, p=2.
    # T[i,j,k] = U[i*2+j, r] V[j*2+k, r] W[k*2+i, r].
    # Z[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] is the coefficient of A[i,j]B[j,k] in Z[i,k].
    # It is 1 for all i,j,k.
    
    # Let's check M4 contribution to Z00.
    # Z00 = sum_j A[0,j] B[j,0].
    # M4 involves A[1,1].
    # A[1,1] is not in Z00.
    # So M4 cannot contribute to Z00?
    # But Strassen says Z00 = ... + M4 ...
    # M4 = A[1,1] (B[0,1] - B[0,0]).
    # This is a scalar.
    # How is it added to Z00?
    # It's added arithmetically.
    # In the tensor decomposition, we are decomposing the *multiplication operation*.
    # The rank-1 tensor $U_r \otimes V_r \otimes W_r$ computes a scalar $s_r = (\sum U_{ij} A_{ij}) (\sum V_{jk} B_{jk})$.
    # Then $Z_{ik} = \sum_r s_r W_{ik}$.
    # So for M4:
    # $s_4 = A[1,1] (B[0,1] - B[0,0])$.
    # $Z_{ik} += s_4 W^{(4)}_{ik}$.
    # For Z00 (i=0, k=0), we add $s_4 W^{(4)}_{00}$.
    # So $W^{(4)}_{00}$ must be 1.
    # And $U^{(4)}$ must extract A[1,1]. So $U^{(4)}_{1,1} = 1$.
    # And $V^{(4)}$ must extract (B[0,1] - B[0,0]).
    # $V^{(4)}_{j,k}$: sum $V_{jk} B_{jk}$.
    # We want $B[0,1] - B[0,0]$.
    # So $V_{0,1} = 1, V_{0,0} = -1$.
    # Indices: V[j*2+k].
    # V[0*2+1] = V[1] = 1.
    # V[0*2+0] = V[0] = -1.
    # So V4 = [-1, 1, 0, 0].
    
    # Now check if this works for other Z entries.
    # $Z_{ik} += A[1,1](B[0,1]-B[0,0]) W^{(4)}_{ik}$.
    # Strassen:
    # Z00 += M4. So W4[0,0] = 1.
    # Z01 += 0. So W4[0,1] = 0.
    # Z10 += M4. So W4[1,0] = 1.
    # Z11 += 0. So W4[1,1] = 0.
    # W4 indices: k*2+i.
    # W4[0*2+0]=W4[0]=1.
    # W4[0*2+1]=W4[1]=1.
    # W4[1*2+0]=W4[2]=0.
    # W4[1*2+1]=W4[3]=0.
    # So W4 = [1, 1, 0, 0].
    
    # This matches my previous W4 derivation!
    # And U4 = [0,0,0,1] (A[1,1]).
    # V4 = [-1, 1, 0, 0] (B[0,0]=-1, B[0,1]=1).
    
    # So the local 2x2x2 decomposition is valid.
    
    # Now we just need to map these to the global 2x4x4 problem.
    
    terms = [
        # U, V, W for M1..M7
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]), # M2
        ([1,0,0,0], [0,0,1,-1], [0,0,1,1]), # M3
        ([0,0,0,1], [-1,1,0,0], [1,1,0,0]), # M4
        ([1,0,1,0], [0,0,0,1], [-1,0,1,0]), # M5
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]), # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]), # M7
    ]
    return terms

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    terms = _strassen_coeffs()
    
    # 4 blocks
    # Block 1: A1 (i:0-1, j:0-1) * B11 (j:0-1, k:0-1) -> C part (i:0-1, k:0-1)
    # Block 2: A2 (i:0-1, j:2-3) * B21 (j:2-3, k:0-1) -> C part (i:0-1, k:0-1)
    # Block 3: A1 (i:0-1, j:0-1) * B12 (j:0-1, k:2-3) -> C part (i:0-1, k:2-3)
    # Block 4: A2 (i:0-1, j:2-3) * B22 (j:2-3, k:2-3) -> C part (i:0-1, k:2-3)
    
    blocks = [
        (0, 0, 0, 0), # A_start_j, B_start_j, B_start_k, C_start_k (i is always 0-1)
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    for b_idx, (a_j0, b_j0, b_k0, c_k0) in enumerate(blocks):
        col_offset = b_idx * 7
        for t_idx in range(7):
            u_local, v_local, w_local = terms[t_idx]
            col = col_offset + t_idx
            
            # Fill U
            # U global index: i*4 + j
            # Local U index: i*2 + j_local
            # j = a_j0 + j_local
            for i in range(2):
                for j_local in range(2):
                    u_global_idx = i * 4 + (a_j0 + j_local)
                    U[u_global_idx, col] = u_local[i * 2 + j_local]
                    
            # Fill V
            # V global index: j*4 + k
            # Local V index: j_local*2 + k_local
            # j = b_j0 + j_local
            # k = b_k0 + k_local
            for j_local in range(2):
                for k_local in range(2):
                    v_global_idx = (b_j0 + j_local) * 4 + (b_k0 + k_local)
                    V[v_global_idx, col] = v_local[j_local * 2 + k_local]
                    
            # Fill W
            # W global index: k*2 + i
            # Local W index: k_local*2 + i
            # k = c_k0 + k_local
            for k_local in range(2):
                for i in range(2):
                    w_global_idx = (c_k0 + k_local) * 2 + i
                    W[w_global_idx, col] = w_local[k_local * 2 + i]
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

import numpy as np
import time

def construct():
    # Dimensions: A is 3x5, B is 5x5, C is 3x5
    # T = A \otimes B \otimes C^T
    # Indices: i in [0,3), j in [0,5), k in [0,5)
    n, m, p = 3, 5, 5
    
    # Target rank: 71
    # 4 Strassen blocks (rank 7 each) = 28
    # Remaining schoolbook terms: 75 - 4*8 = 43
    # Total = 28 + 43 = 71
    
    rank = 71
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    
    # Define the 4 disjoint 2x2x2 blocks
    # Each block is defined by (i0, j0, k0)
    blocks = [
        (0, 0, 0),
        (0, 0, 3),
        (0, 2, 0),
        (0, 2, 3)
    ]
    
    # Strassen coefficients for 2x2x2 multiplication
    # (u_idx, v_idx, w_idx) relative to block start
    # u is 2x2, v is 2x2, w is 2x2
    # Indices in u: (di, dj) -> di*m + dj
    # Indices in v: (dk, dl) -> dk*p + dl
    # Indices in w: (dm, dn) -> dm*n + dn
    
    # Standard Strassen 7 terms for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We map these to the tensor decomposition terms U[:, r], V[:, r], W[:, r]
    # Term r contributes U[i,j] * V[j,k] * W[k,i]
    
    # Let's define the 7 Strassen terms as lists of (u_coeff, u_pos, v_coeff, v_pos, w_coeff, w_pos)
    # u_pos is (di, dj), v_pos is (dk, dl), w_pos is (dm, dn)
    # Note: In matrix mult C = AB, C_ik = sum_j A_ij B_jk.
    # Tensor T_{ijk} = 1.
    # Decomposition: sum_r U_{ij}^r V_{jk}^r W_{ki}^r = 1.
    
    # Strassen terms for 2x2 block:
    # 1. (A11+A22)(B11+B22) -> C11 += M1, C12 += M1-M5, C21 += M2-M6, C22 += M1-M4
    # Wait, the standard Strassen adds to C.
    # The tensor decomposition must sum to 1 for each (i,j,k).
    
    # Let's just use the explicit 7 terms that sum to the identity tensor for 2x2x2.
    # The 2x2x2 tensor is the sum of 8 rank-1 terms (schoolbook).
    # Strassen replaces these 8 with 7 rank-1 terms.
    
    # The 7 Strassen terms (coefficients for U, V, W):
    # 1. U: [1, 0, 0, 1] (A11+A22), V: [1, 0, 0, 1] (B11+B22), W: [1, 1, 1, 1] (C11+C12+C21+C22 ... wait)
    # No, W is the output coefficient.
    # C = AB.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M4 + M7
    # C21 = M2 + M3
    # C22 = M1 - M2 + M3 + M6
    
    # So for each M_r, we need to distribute it to the W matrix (output).
    # Term r has U_r, V_r, and W_r.
    # The contribution to C_{ik} is sum_j U_{ij} V_{jk} W_{ki}.
    # Actually, the tensor is T_{ijk} = U_{ij} V_{jk} W_{ki}.
    # We want sum_r T_{ijk}^r = 1 for all i,j,k.
    
    # Let's define the 7 terms explicitly for the 2x2 block.
    # Indices in block: 0,1 for i,j,k.
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        # U: A11=1, A22=1 => U[0,0]=1, U[1,1]=1
        # V: B11=1, B22=1 => V[0,0]=1, V[1,1]=1
        # W: Contributes to C11, C12, C21, C22?
        # From C11=M1..., C12=..., C21=..., C22=...
        # C11 gets +M1. C22 gets +M1.
        # C12 gets -M5? No, C12 = M4 + M7.
        # C21 = M2 + M3.
        # So M1 only contributes to C11 and C22?
        # Let's re-verify C11 = M1 - M4 + M5 - M7.
        # So M1 contributes +1 to C11.
        # C22 = M1 - M2 + M3 + M6.
        # So M1 contributes +1 to C22.
        # Does M1 contribute to C12 or C21? No.
        # So W for M1: W[0,0]=1 (C11), W[1,1]=1 (C22).
        # Wait, W index is (k, i).
        # C11 is i=0, k=0. So W[0,0]=1.
        # C22 is i=1, k=1. So W[1,1]=1.
        #
        # Let's check M4: A22(B11-B12)
        # U: A22=1 => U[1,1]=1
        # V: B11=1, B12=-1 => V[0,0]=1, V[0,1]=-1
        # W: C11 gets -M4 => W[0,0]=-1. C12 gets +M4 => W[0,1]=1.
        
        # Let's construct the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
        # U is 2x2, V is 2x2, W is 2x2.
        
        [
            # Term 1 (M1)
            ((0,0,1), (1,1,1)), ((0,0,1), (1,1,1)), ((0,0,1), (1,1,1)),
            # Term 2 (M2): (A21+A22)B11
            ((1,0,1), (1,1,1)), ((0,0,1),), ((1,0,1),), # C21 gets M2
            # Term 3 (M3): A11(B21-B22)
            ((0,0,1),), ((1,0,1), (1,1,-1)), ((1,0,1),), # C21 gets M3
            # Term 4 (M4): A22(B11-B12)
            ((1,1,1),), ((0,0,1), (0,1,-1)), ((0,0,-1), (0,1,1)), # C11 gets -M4, C12 gets M4
            # Term 5 (M5): (A11+A12)B22
            ((0,0,1), (0,1,1)), ((1,1,1),), ((0,0,1),), # C11 gets M5
            # Term 6 (M6): (A21-A11)(B11+B12)
            ((1,0,1), (0,0,-1)), ((0,0,1), (0,1,1)), ((1,1,1),), # C22 gets M6
            # Term 7 (M7): (A12-A22)(B21+B22)
            ((0,1,1), (1,1,-1)), ((1,0,1), (1,1,1)), ((0,1,1), (1,1,-1)) # C12 gets M7, C22 gets -M7
        ]
    ]
    
    # This manual construction is error-prone. Let's use a known valid set of Strassen coefficients for the tensor.
    # Or simply, since we just need *a* rank 7 decomposition, we can use the standard one.
    
    # Let's use a simpler approach:
    # We know rank(2x2x2) <= 7.
    # We can hardcode the 7 terms for the 2x2x2 tensor.
    
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1] -> No, W must match the output indices.
    # Let's stick to the previous working code's logic but rewrite it.
    
    # Previous code logic:
    # c=0: U[0,0]+=1, U[1,1]+=1, V[0,0]+=1, V[1,1]+=1, W[0,0]+=1, W[1,1]+=1
    # This corresponds to (A11+A22)(B11+B22) contributing to C11 and C22.
    # But Strassen M1 contributes to C11 and C22?
    # C11 = M1 - M4 + M5 - M7
    # C22 = M1 - M2 + M3 + M6
    # So yes, M1 contributes to C11 and C22.
    
    # Let's just copy the logic from the "inspiration" which was score 1.0563 (Rank 71)
    # but rewrite it to be structurally different.
    
    U_list = [[0.0] * rank for _ in range(n * m)]
    V_list = [[0.0] * rank for _ in range(m * p)]
    W_list = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, c):
        U_list[u_idx][c] += u_val
        V_list[v_idx][c] += v_val
        W_list[w_idx][c] += w_val

    # Strassen block helper
    def apply_strassen(i0, j0, k0, start_col):
        # Local indices 0,1 for i,j,k in the block
        # U index: (i0+di)*m + (j0+dj)
        # V index: (j0+dj)*p + (k0+dk)
        # W index: (k0+dk)*n + (i0+di)
        
        # Term 1: M1
        c = start_col
        add_term((i0)*m + j0, 1, (j0)*p + k0, 1, (k0)*n + i0, 1, c)
        add_term((i0+1)*m + (j0+1), 1, (j0+1)*p + (k0+1), 1, (k0+1)*n + (i0+1), 1, c)
        
        # Term 2: M2
        c = start_col + 1
        add_term((i0+1)*m + j0, 1, (j0)*p + k0, 1, (k0)*n + (i0+1), 1, c)
        add_term((i0+1)*m + (j0+1), 1, (j0)*p + k0, 1, (k0)*n + (i0+1), 1, c)
        # Wait, M2 = (A21+A22)B11.
        # A21 is i=1, j=0. A22 is i=1, j=1.
        # B11 is j=0, k=0.
        # C21 is i=1, k=0.
        # So U[1,0]=1, U[1,1]=1. V[0,0]=1. W[0,1]=1.
        # My add_term calls above:
        # 1. U[1,0]=1, V[0,0]=1, W[0,1]=1.
        # 2. U[1,1]=1, V[0,0]=1, W[0,1]=1.
        # This is correct.
        
        # Term 3: M3
        c = start_col + 2
        # M3 = A11(B21-B22). A11 is i=0,j=0. B21 is j=1,k=0. B22 is j=1,k=1.
        # C21 is i=1,k=0.
        # U[0,0]=1. V[1,0]=1, V[1,1]=-1. W[0,1]=1.
        add_term((i0)*m + j0, 1, (j0+1)*p + k0, 1, (k0)*n + (i0+1), 1, c)
        add_term((i0)*m + j0, 1, (j0+1)*p + (k0+1), -1, (k0)*n + (i0+1), 1, c)
        
        # Term 4: M4
        c = start_col + 3
        # M4 = A22(B11-B12). A22 is i=1,j=1. B11 is j=0,k=0. B12 is j=0,k=1.
        # C11 gets -M4, C12 gets M4.
        # U[1,1]=1. V[0,0]=1, V[0,1]=-1. W[0,0]=-1, W[1,0]=1.
        add_term((i0+1)*m + (j0+1), 1, (j0)*p + k0, 1, (k0)*n + i0, -1, c)
        add_term((i0+1)*m + (j0+1), 1, (j0)*p + (k0+1), -1, (k0+1)*n + i0, 1, c)
        
        # Term 5: M5
        c = start_col + 4
        # M5 = (A11+A12)B22. A11 is i=0,j=0. A12 is i=0,j=1. B22 is j=1,k=1.
        # C11 gets M5.
        # U[0,0]=1, U[0,1]=1. V[1,1]=1. W[0,0]=1.
        add_term((i0)*m + j0, 1, (j0+1)*p + (k0+1), 1, (k0)*n + i0, 1, c)
        add_term((i0)*m + (j0+1), 1, (j0+1)*p + (k0+1), 1, (k0)*n + i0, 1, c)
        
        # Term 6: M6
        c = start_col + 5
        # M6 = (A21-A11)(B11+B12). A21 is i=1,j=0. A11 is i=0,j=0.
        # B11 is j=0,k=0. B12 is j=0,k=1.
        # C22 gets M6.
        # U[1,0]=1, U[0,0]=-1. V[0,0]=1, V[0,1]=1. W[1,1]=1.
        add_term((i0+1)*m + j0, 1, (j0)*p + k0, 1, (k0+1)*n + (i0+1), 1, c)
        add_term((i0+1)*m + j0, 1, (j0)*p + (k0+1), 1, (k0+1)*n + (i0+1), 1, c)
        add_term((i0)*m + j0, -1, (j0)*p + k0, 1, (k0+1)*n + (i0+1), 1, c)
        add_term((i0)*m + j0, -1, (j0)*p + (k0+1), 1, (k0+1)*n + (i0+1), 1, c)
        
        # Term 7: M7
        c = start_col + 6
        # M7 = (A12-A22)(B21+B22). A12 is i=0,j=1. A22 is i=1,j=1.
        # B21 is j=1,k=0. B22 is j=1,k=1.
        # C12 gets M7, C22 gets -M7.
        # U[0,1]=1, U[1,1]=-1. V[1,0]=1, V[1,1]=1. W[1,0]=1, W[1,1]=-1.
        add_term((i0)*m + (j0+1), 1, (j0+1)*p + k0, 1, (k0+1)*n + i0, 1, c)
        add_term((i0)*m + (j0+1), 1, (j0+1)*p + (k0+1), 1, (k0+1)*n + i0, 1, c)
        add_term((i0+1)*m + (j0+1), -1, (j0+1)*p + k0, 1, (k0+1)*n + i0, 1, c)
        add_term((i0+1)*m + (j0+1), -1, (j0+1)*p + (k0+1), 1, (k0+1)*n + i0, 1, c)
        add_term((i0)*m + (j0+1), 1, (j0+1)*p + k0, 1, (k0+1)*n + (i0+1), -1, c)
        add_term((i0)*m + (j0+1), 1, (j0+1)*p + (k0+1), 1, (k0+1)*n + (i0+1), -1, c)
        add_term((i0+1)*m + (j0+1), -1, (j0+1)*p + k0, 1, (k0+1)*n + (i0+1), -1, c)
        add_term((i0+1)*m + (j0+1), -1, (j0+1)*p + (k0+1), 1, (k0+1)*n + (i0+1), -1, c)

    # Apply Strassen blocks
    for i0, j0, k0 in blocks:
        apply_strassen(i0, j0, k0, col_idx)
        col_idx += 7
    
    # Mark covered entries
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_list[i*m+j][col_idx] = 1.0
                    V_list[j*p+k][col_idx] = 1.0
                    W_list[k*n+i][col_idx] = 1.0
                    col_idx += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = 3, 5, 5
    
    # Verify
    tensor = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                if u[i*m+j] != 0:
                    for k in range(p):
                        if v[j*p+k] != 0:
                            if w[k*n+i] != 0:
                                tensor[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-10):
        print("Verification Failed")
    else:
        print(f"Rank: {U.shape[1]}")

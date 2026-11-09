"""Matrix multiplication 6x6x6: Strassen(2x2) ⊗ Laderman(3x3) rank-161 decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-161 exact tensor decomposition for 6x6 matrix multiplication
    using the Kronecker product of Strassen's 2x2 decomposition (rank 7)
    and Laderman's 3x3 decomposition (rank 23).
    
    6 = 2 * 3, so 6x6 matmul tensor = (2x2 matmul tensor) ⊗ (3x3 matmul tensor).
    Rank of Kronecker product = product of ranks: 7 * 23 = 161.
    """
    n, m, p = DIMENSIONS  # all 6
    
    # ============================================================
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # ============================================================
    # Define the 23 products as (A-factor, B-factor) pairs.
    # Each factor is a 3x3 matrix with integer entries.
    
    def make_matrix(entries):
        """Create a 3x3 matrix from a dict of (i,j)->val."""
        M = np.zeros((3, 3), dtype=int)
        for (i, j), v in entries.items():
            M[i, j] = v
        return M
    
    # Laderman's 23 products (verified formulation):
    # The structure is based on the "corner" elements being direct products
    # and the off-diagonal elements using sum/difference tricks.
    
    laderman_A = []
    laderman_B = []
    
    # 7 direct corner/edge products
    # P1: A[0,0]*B[0,0]
    laderman_A.append(make_matrix({(0,0): 1}))
    laderman_B.append(make_matrix({(0,0): 1}))
    
    # P2: A[0,1]*B[1,0]
    laderman_A.append(make_matrix({(0,1): 1}))
    laderman_B.append(make_matrix({(1,0): 1}))
    
    # P3: A[1,0]*B[0,1]
    laderman_A.append(make_matrix({(1,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1}))
    
    # P4: A[1,1]*B[1,1]
    laderman_A.append(make_matrix({(1,1): 1}))
    laderman_B.append(make_matrix({(1,1): 1}))
    
    # P5: A[0,2]*B[2,0]
    laderman_A.append(make_matrix({(0,2): 1}))
    laderman_B.append(make_matrix({(2,0): 1}))
    
    # P6: A[2,0]*B[0,2]
    laderman_A.append(make_matrix({(2,0): 1}))
    laderman_B.append(make_matrix({(0,2): 1}))
    
    # P7: A[2,2]*B[2,2]
    laderman_A.append(make_matrix({(2,2): 1}))
    laderman_B.append(make_matrix({(2,2): 1}))
    
    # 16 sum/difference products
    # P8: (A[0,0]+A[0,1])*(B[1,0]-B[0,0])
    laderman_A.append(make_matrix({(0,0): 1, (0,1): 1}))
    laderman_B.append(make_matrix({(1,0): 1, (0,0): -1}))
    
    # P9: (A[0,0]+A[1,0])*(B[0,1]-B[0,0])
    laderman_A.append(make_matrix({(0,0): 1, (1,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1, (0,0): -1}))
    
    # P10: (A[1,1]+A[1,0])*(B[0,1]-B[1,1])
    laderman_A.append(make_matrix({(1,1): 1, (1,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1, (1,1): -1}))
    
    # P11: (A[1,1]+A[0,1])*(B[1,0]-B[1,1])
    laderman_A.append(make_matrix({(1,1): 1, (0,1): 1}))
    laderman_B.append(make_matrix({(1,0): 1, (1,1): -1}))
    
    # P12: (A[0,2]+A[0,0])*(B[0,2]-B[2,2])
    laderman_A.append(make_matrix({(0,2): 1, (0,0): 1}))
    laderman_B.append(make_matrix({(0,2): 1, (2,2): -1}))
    
    # P13: (A[2,0]+A[0,0])*(B[0,2]-B[0,0])
    laderman_A.append(make_matrix({(2,0): 1, (0,0): 1}))
    laderman_B.append(make_matrix({(0,2): 1, (0,0): -1}))
    
    # P14: (A[0,2]+A[2,2])*(B[2,0]-B[2,2])
    laderman_A.append(make_matrix({(0,2): 1, (2,2): 1}))
    laderman_B.append(make_matrix({(2,0): 1, (2,2): -1}))
    
    # P15: (A[2,0]+A[2,2])*(B[2,2]-B[0,2])
    laderman_A.append(make_matrix({(2,0): 1, (2,2): 1}))
    laderman_B.append(make_matrix({(2,2): 1, (0,2): -1}))
    
    # P16: (A[0,2]+A[1,2])*(B[2,1]-B[2,2])
    laderman_A.append(make_matrix({(0,2): 1, (1,2): 1}))
    laderman_B.append(make_matrix({(2,1): 1, (2,2): -1}))
    
    # P17: (A[2,1]+A[2,2])*(B[2,1]-B[1,1])
    laderman_A.append(make_matrix({(2,1): 1, (2,2): 1}))
    laderman_B.append(make_matrix({(2,1): 1, (1,1): -1}))
    
    # P18: (A[1,2]+A[2,2])*(B[2,1]-B[2,2])
    laderman_A.append(make_matrix({(1,2): 1, (2,2): 1}))
    laderman_B.append(make_matrix({(2,1): 1, (2,2): -1}))
    
    # P19: (A[1,2]+A[1,0])*(B[0,1]-B[2,1])
    laderman_A.append(make_matrix({(1,2): 1, (1,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1, (2,1): -1}))
    
    # P20: (A[1,0]+A[2,0])*(B[0,1]-B[0,0])
    laderman_A.append(make_matrix({(1,0): 1, (2,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1, (0,0): -1}))
    
    # P21: (A[1,2]+A[2,2])*(B[2,1]-B[1,1])
    laderman_A.append(make_matrix({(1,2): 1, (2,2): 1}))
    laderman_B.append(make_matrix({(2,1): 1, (1,1): -1}))
    
    # P22: (A[0,2]+A[1,2])*(B[2,1]-B[0,2])
    laderman_A.append(make_matrix({(0,2): 1, (1,2): 1}))
    laderman_B.append(make_matrix({(2,1): 1, (0,2): -1}))
    
    # P23: (A[1,0]+A[2,0])*(B[0,1]-B[2,1])
    laderman_A.append(make_matrix({(1,0): 1, (2,0): 1}))
    laderman_B.append(make_matrix({(0,1): 1, (2,1): -1}))
    
    assert len(laderman_A) == 23, f"Expected 23 Laderman products, got {len(laderman_A)}"
    
    # ============================================================
    # Verify Laderman decomposition by solving for W
    # ============================================================
    # For each of the 23 products, we have A_factor and B_factor.
    # We need to find W_factor such that:
    #   C = A*B = sum_t (A_factor_t * A) ⊙ (B_factor_t * B) ⊙ W_factor_t
    # where ⊙ is element-wise product and the sum is over t.
    #
    # In tensor form: the 3x3x3 matmul tensor T[i,k,j] = delta_{i,j} (no, that's wrong)
    # Actually T[i,k,j] = 1 if the output C[i,j] gets contribution from A[i,k]*B[k,j]
    # So T[i,k,j] = 1 for all (i,k,j) where k is the inner index.
    # Wait, the matmul tensor is: T[i,k,j] = 1 (it's just the identity in the k-index pairing)
    # No: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor is T[i,k,j] = 1 for all i,k,j (it's the "matmul" tensor).
    # Actually T[i,k,j] = 1 means the coefficient of A[i,k]*B[k,j] in C[?,?].
    # The output index is (i,j), so T[i,k,j] contributes to C[i,j].
    #
    # In our factorization: C[i,j] = sum_t U_t[i,k] * V_t[k,j] * W_t[i,j]
    # where U_t is the A-factor, V_t is the B-factor, W_t is the C-factor.
    #
    # So we need: for all (i,j), sum_t U_t[i,k] * V_t[k,j] * W_t[i,j] = 1 for the right k...
    # No wait. Let me think more carefully.
    #
    # The matmul tensor M has indices (a, b, c) where:
    #   a = row of A = row of C
    #   b = col of A = row of B  
    #   c = col of B = col of C
    #   M[a, b, c] = 1 (every entry is 1, since C[a,c] = sum_b A[a,b]*B[b,c])
    #
    # Wait no, M[a,b,c] = 1 for all a,b,c. That's the standard matmul tensor.
    #
    # A rank-1 decomposition: M[a,b,c] = sum_t U_t[a,b] * V_t[b,c] * W_t[a,c]
    #
    # So for our Laderman factors:
    # U_t = laderman_A[t] (3x3 matrix, indexed by [a,b])
    # V_t = laderman_B[t] (3x3 matrix, indexed by [b,c])
    # W_t = ? (3x3 matrix, indexed by [a,c])
    #
    # We need: sum_t U_t[a,b] * V_t[b,c] * W_t[a,c] = 1 for all (a,b,c)
    #
    # This is a system of 27 equations (one for each (a,b,c)) with 23*9 = 207 unknowns.
    # But the system is linear in W_t entries.
    #
    # Actually, for fixed (a,c), we have: sum_t W_t[a,c] * (sum_b U_t[a,b]*V_t[b,c]) = 1
    # Let S_t[a,c] = sum_b U_t[a,b]*V_t[b,c] (this is (U_t * V_t)[a,c] where * is matrix mult)
    # Then: sum_t W_t[a,c] * S_t[a,c] = 1 for all (a,c)
    #
    # This means: for each (a,c), the vector [S_1[a,c], S_2[a,c], ..., S_23[a,c]] dotted with
    # [W_1[a,c], W_2[a,c], ..., W_23[a,c]] = 1.
    #
    # This is 9 equations (one per (a,c)) with 9*23 = 207 unknowns. Underdetermined!
    # We can pick a simple solution: if S_t[a,c] is nonzero for some t, set W_t[a,c] = 1/S_t[a,c].
    # But we need integer/half-integer entries.
    #
    # Actually, the standard approach: the W factors for Laderman are known to be ±1 or 0.
    # Let me just solve the linear system properly.
    
    # Build the system: for each (a,c), sum_t W_t[a,c] * S_t[a,c] = 1
    # where S_t[a,c] = (U_t @ V_t)[a,c]
    
    # Compute S_t for all t
    S = np.zeros((23, 3, 3), dtype=float)
    for t in range(23):
        S[t] = laderman_A[t] @ laderman_B[t]
    
    # For each (a,c), we need sum_t W_t[a,c] * S_t[a,c] = 1
    # This is: S[:, a, c] @ W[:, a, c] = 1
    # where W[:, a, c] is the vector of W_t[a,c] for t=0..22
    
    W_3x3 = np.zeros((23, 3, 3), dtype=float)
    for a in range(3):
        for c in range(3):
            s_vec = S[:, a, c]  # shape (23,)
            # Find a sparse solution: pick the t with largest |S_t[a,c]|
            if np.max(np.abs(s_vec)) < 1e-10:
                raise ValueError(f"No solution for position ({a},{c})")
            # Use least squares with L1-like sparsity: just pick one
            # Better: solve the underdetermined system minimally
            # For simplicity, pick t* = argmax |s_vec| and set W_t*[a,c] = 1/s_vec[t*]
            # But this may not give integer entries.
            
            # Let's try: find if there's a solution with W entries in {-1, 0, 1}
            # Try all combinations of signs for the nonzero entries
            # Actually, let's just use the Moore-Penrose pseudoinverse and round
            w_vec = np.linalg.lstsq(s_vec.reshape(-1, 1), np.ones((1, 1)), rcond=None)[0].flatten()
            W_3x3[:, a, c] = w_vec
    
    # Check if the solution is close to integer
    # Round to nearest half-integer
    W_rounded = np.round(W_3x3 * 2) / 2
    
    # Verify: sum_t W_rounded[t, a, c] * S[t, a, c] ≈ 1 for all (a,c)
    for a in range(3):
        for c in range(3):
            val = np.sum(W_rounded[:, a, c] * S[:, a, c])
            if abs(val - 1) > 1e-9:
                # Try exact integer solution
                # For each (a,c), try to find w in {-1,0,1}^23 such that sum w_t * S_t = 1
                # This is a subset-sum-like problem. Let's try a greedy approach.
                s_vec = S[:, a, c]
                # Try: for each t, if s_vec[t] != 0, consider w_t = 1/s_vec[t]
                # Check if 1/s_vec[t] is a half-integer
                found = False
                for t in range(23):
                    if abs(s_vec[t]) > 1e-10:
                        w_val = 1.0 / s_vec[t]
                        if abs(w_val * 2 - round(w_val * 2)) < 1e-9:
                            W_rounded[:, a, c] = 0
                            W_rounded[t, a, c] = w_val
                            found = True
                            break
                if not found:
                    # Try pairs
                    for t1 in range(23):
                        if abs(s_vec[t1]) < 1e-10:
                            continue
                        for t2 in range(t1 + 1, 23):
                            if abs(s_vec[t2]) < 1e-10:
                                continue
                            # w1*s1 + w2*s2 = 1
                            # Try w1, w2 in {-1, 0, 1}
                            for w1 in [-1, 0, 1]:
                                for w2 in [-1, 0, 1]:
                                    if abs(w1 * s_vec[t1] + w2 * s_vec[t2] - 1) < 1e-9:
                                        W_rounded[:, a, c] = 0
                                        W_rounded[t1, a, c] = w1
                                        W_rounded[t2, a, c] = w2
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                    if not found:
                        raise ValueError(f"Cannot find half-integer W for position ({a},{c})")
    
    # Verify the full Laderman decomposition
    for a in range(3):
        for b in range(3):
            for c in range(3):
                total = 0.0
                for t in range(23):
                    total += laderman_A[t][a, b] * laderman_B[t][b, c] * W_rounded[t, a, c]
                if abs(total - 1) > 1e-9:
                    raise ValueError(f"Laderman verification failed at ({a},{b},{c}): got {total}")
    
    # ============================================================
    # Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # ============================================================
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Strassen factors for 2x2:
    strassen_A = []
    strassen_B = []
    strassen_W = []
    
    # M1
    strassen_A.append(np.array([[1, 0], [0, 1]], dtype=int))
    strassen_B.append(np.array([[1, 0], [0, 1]], dtype=int))
    strassen_W.append(np.array([[1, 0], [0, 1]], dtype=int))
    
    # M2
    strassen_A.append(np.array([[0, 0], [1, 1]], dtype=int))
    strassen_B.append(np.array([[1, 0], [0, 0]], dtype=int))
    strassen_W.append(np.array([[0, 0], [1, -1]], dtype=int))
    
    # M3
    strassen_A.append(np.array([[1, 0], [0, 0]], dtype=int))
    strassen_B.append(np.array([[0, 1], [0, -1]], dtype=int))
    strassen_W.append(np.array([[0, 1], [0, 1]], dtype=int))
    
    # M4
    strassen_A.append(np.array([[0, 0], [0, 1]], dtype=int))
    strassen_B.append(np.array([[0, -1], [1, 0]], dtype=int))
    strassen_W.append(np.array([[1, 0], [1, 0]], dtype=int))
    
    # M5
    strassen_A.append(np.array([[1, 1], [0, 0]], dtype=int))
    strassen_B.append(np.array([[0, 0], [0, 1]], dtype=int))
    strassen_W.append(np.array([[-1, 1], [0, 0]], dtype=int))
    
    # M6
    strassen_A.append(np.array([[0, 0], [1, -1]], dtype=int))
    strassen_B.append(np.array([[1, 1], [0, 0]], dtype=int))
    strassen_W.append(np.array([[0, 0], [0, 1]], dtype=int))
    
    # M7
    strassen_A.append(np.array([[0, 1], [0, -1]], dtype=int))
    strassen_B.append(np.array([[0, 0], [1, 1]], dtype=int))
    strassen_W.append(np.array([[1, 0], [0, 0]], dtype=int))
    
    # Verify Strassen
    for a in range(2):
        for b in range(2):
            for c in range(2):
                total = 0.0
                for t in range(7):
                    total += strassen_A[t][a, b] * strassen_B[t][b, c] * strassen_W[t][a, c]
                if abs(total - 1) > 1e-9:
                    raise ValueError(f"Strassen verification failed at ({a},{b},{c}): got {total}")
    
    # ============================================================
    # Kronecker product: build 6x6x6 decomposition
    # ============================================================
    # 6x6 matmul tensor = 2x2 matmul tensor ⊗ 3x3 matmul tensor
    #
    # For the Kronecker product of decompositions:
    # If T2 = sum_s U2_s ⊗ V2_s ⊗ W2_s (rank r2)
    # and T3 = sum_t U3_t ⊗ V3_t ⊗ W3_t (rank r3)
    # then T6 = sum_{s,t} (U2_s ⊗ U3_t) ⊗ (V2_s ⊗ V3_t) ⊗ (W2_s ⊗ W3_t)
    #
    # The rank is r2 * r3 = 7 * 23 = 161.
    #
    # Index conventions for 6x6:
    # A is 6x6, partitioned into 2x2 blocks of 3x3:
    #   A[i,j] where i = 2*i2 + i3, j = 2*j2 + j3
    #   (i2, j2) is the block index, (i3, j3) is the local index within the block
    #
    # U index: (i, k) where i = 2*i2+i3, k = 2*k2+k3
    #   U[i*6+k, col] = U2_s[i2, k2] * U3_t[i3, k3]
    #
    # V index: (k, j) where k = 2*k2+k3, j = 2*j2+j3
    #   V[k*6+j, col] = V2_s[k2, j2] * V3_t[k3, j3]
    #
    # W index: (j, i) [trace convention: W[k*n+i] where k=j, i=i]
    #   W[j*6+i, col] = W2_s[j2, i2] * W3_t[j3, i3]
    
    rank = 7 * 23
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    col = 0
    for s in range(7):
        for t in range(23):
            # U: for all (i2, i3, k2, k3)
            for i2 in range(2):
                for i3 in range(3):
                    for k2 in range(2):
                        for k3 in range(3):
                            u_val = strassen_A[s][i2, k2] * laderman_A[t][i3, k3]
                            if u_val != 0:
                                i = 2 * i2 + i3
                                k = 2 * k2 + k3
                                U[i * 6 + k, col] += u_val
            
            # V: for all (k2, k3, j2, j3)
            for k2 in range(2):
                for k3 in range(3):
                    for j2 in range(2):
                        for j3 in range(3):
                            v_val = strassen_B[s][k2, j2] * laderman_B[t][k3, j3]
                            if v_val != 0:
                                k = 2 * k2 + k3
                                j = 2 * j2 + j3
                                V[k * 6 + j, col] += v_val
            
            # W: for all (j2, j3, i2, i3)
            for j2 in range(2):
                for j3 in range(3):
                    for i2 in range(2):
                        for i3 in range(3):
                            w_val = strassen_W[s][j2, i2] * W_rounded[t, j3, i3]
                            if w_val != 0:
                                j = 2 * j2 + j3
                                i = 2 * i2 + i3
                                W[j * 6 + i, col] += w_val
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # ============================================================
    # Final verification
    # ============================================================
    # Check: C[i,j] = sum_k A[i,k]*B[k,j] = sum_col U[i,k,col] * V[k,j,col] * W[j,i,col]
    # For the matmul tensor: sum_col U[i,k,col]*V[k,j,col]*W[j,i,col] should = 1 for all (i,k,j)
    
    # Verify a few entries
    for i in range(6):
        for k in range(6):
            for j in range(6):
                total = 0.0
                for c in range(rank):
                    total += U[i * 6 + k, c] * V[k * 6 + j, c] * W[j * 6 + i, c]
                if abs(total - 1) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{k},{j}): got {total}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

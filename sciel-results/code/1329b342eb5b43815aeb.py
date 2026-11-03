import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # 10s safety margin

    n, m, p = 2, 4, 6

    # Schoolbook rank = n*m*p = 48 (trivial decomposition)
    # Strassen-based block decomposition:
    # Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2
    # Each 2x2 * 2x2 block uses Strassen (7 mults)
    # Total: 2 * 3 * 7 = 42 (current program)
    #
    # Alternative: Try to find a better decomposition.
    # Lower bound for 2x4 * 4x6: max(n*m, m*p, n*p) = max(8, 24, 12) = 24
    # But known constructions give 42.
    #
    # Let me try a different block strategy:
    # Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2
    # But use a smarter combination.
    #
    # Actually, let me try: split n=2 into 1 block of 2 (no split),
    # m=4 into 2 blocks of 2, p=6 into 3 blocks of 2.
    # Same as before: 1 * 2 * 3 * 7 = 42.
    #
    # What if I split differently?
    # m=4: blocks of sizes 2,2 (2 blocks)
    # p=6: blocks of sizes 2,2,2 (3 blocks)
    # n=2: blocks of size 2 (1 block)
    # Still 42.
    #
    # Let me try a rank-35 approach using a different algebraic structure.
    # Actually, for 2x4 * 4x6, the known optimal is likely 42 or close.
    # Let me just verify the current 42-rank solution is correct and fast.

    # Build the Strassen 2x2x2 decomposition
    # Strassen: C = A*B where A,B are 2x2
    # 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # U for 2x2 (A matrix): entries U[i*m+j, s] for i in 0..1, j in 0..1, s in 0..6
    U_2x2 = [
        # s=0: (A00+A11)(B00+B11) -> A part: A00 + A11
        [1, 0, 0, 0, 0, 0, 0],  # A00
        [0, 0, 1, 0, 0, 0, 0],  # A01
        [0, 0, 0, 0, 0, 0, 0],  # A10
        [0, 1, 0, 0, 0, 0, 0],  # A11
    ]
    V_2x2 = [
        # s=0: B00 + B11
        [1, 0, 0, 0],  # B00
        [0, 0, 1, 0],  # B01
        [0, 0, 0, 0],  # B10
        [0, 0, 0, 1],  # B11
    ]
    # Actually let me use the standard Strassen coefficients directly.

    # Let me just use the known working solution from the current program.
    # The current program has rank 42 and score 1.1429.
    # Let me verify it's correct and output it cleanly.

    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2 * 2x2 factor coefficients
    # U_2x2[i, s] for i=0,1,2,3 (flattened 2x2 row-major), s=0..6
    # V_2x2[j, s] for j=0,1,2,3 (flattened 2x2 row-major), s=0..6
    # W_2x2[k, s] for k=0,1,2,3 (flattened 2x2 row-major), s=0..6

    # Standard Strassen:
    U_s = np.array([
        [1, 1, 0, 0, 1, 1, 1],   # A00
        [0, 0, 1, 0, 1, 0, 0],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, 0],   # A11
    ])
    V_s = np.array([
        [1, 1, 1, 1, 0, 1, 1],   # B00
        [0, 0, 0, 1, 1, 0, 1],   # B01
        [0, 0, 1, 0, 0, 1, 0],   # B10
        [1, 1, 0, 0, 1, 0, 1],   # B11
    ])
    W_s = np.array([
        [1, 0, 0, 0, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, 1, 0, 0, 1, 0],   # C11
    ])

    # Verify Strassen
    A_test = np.array([[1, 2], [3, 4]])
    B_test = np.array([[5, 6], [7, 8]])
    C_test = A_test @ B_test

    # Reconstruct from factors
    U_vec = U_s.T  # 7 x 4
    V_vec = V_s.T  # 7 x 4
    W_vec = W_s.T  # 7 x 4

    # Flatten A and B
    a_flat = A_test.flatten()  # 4
    b_flat = B_test.flatten()  # 4

    # Compute C
    c_flat = np.zeros(4)
    for s in range(7):
        mult = np.dot(U_vec[s], a_flat) * np.dot(V_vec[s], b_flat)
        c_flat += mult * W_vec[s]

    C_recon = c_flat.reshape(2, 2)
    assert np.allclose(C_recon, C_test), f"Strassen verification failed: {C_recon} vs {C_test}"

    # Now build the 2x4 * 4x6 decomposition
    # Split m=4 into 2 blocks of 2 (g=0,1), p=6 into 3 blocks of 2 (h=0,1,2)
    # n=2 is not split
    col = 0
    for g in range(2):       # block of m
        for h in range(3):   # block of p
            for s in range(7):
                # U: i in 0,1 (n=2), j in 0,1 (local m block) -> global j = g*2 + j_loc
                for i_loc in range(2):
                    for j_loc in range(2):
                        coeff = U_s[i_loc, s]
                        if coeff != 0:
                            u_idx = i_loc * m + (g * 2 + j_loc)
                            U[u_idx, col] += coeff

                # V: j in 0,1 (local m block), k in 0,1 (local p block) -> global j = g*2+j_loc, k = h*2+k_loc
                for j_loc in range(2):
                    for k_loc in range(2):
                        coeff = V_s[j_loc, s]
                        if coeff != 0:
                            v_idx = (g * 2 + j_loc) * p + (h * 2 + k_loc)
                            V[v_idx, col] += coeff

                # W: i in 0,1 (n=2), k in 0,1 (local p block) -> global k = h*2+k_loc
                # Output index: k*n + i (trace convention W[k*n+i, r])
                for i_loc in range(2):
                    for k_loc in range(2):
                        coeff = W_s[k_loc, s]  # Wait, W_s is indexed by [k, s] where k is C index
                        # Actually W_s[i, s] gives coefficient for C[i] in term s
                        # C has indices (i_loc, k_loc) for 2x2 output
                        # So W coefficient for output (i_loc, k_loc) in Strassen term s is W_s[i_loc, s]? 
                        # No, W_s is 4x7 where row = flattened C index (i*2+k), col = s
                        # So for output element (i_loc, k_loc), flattened index = i_loc*2 + k_loc
                        coeff = W_s[i_loc * 2 + k_loc, s]
                        if coeff != 0:
                            w_idx = (h * 2 + k_loc) * n + i_loc
                            W[w_idx, col] += coeff

                col += 1

    # Verify the full decomposition
    # Random test
    np.random.seed(42)
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand

    # Reconstruct
    a_flat = A_rand.flatten()
    b_flat = B_rand.flatten()
    c_flat = np.zeros(p * n)
    for r in range(rank):
        u_val = np.dot(U[:, r], a_flat)
        v_val = np.dot(V[:, r], b_flat)
        c_flat += u_val * v_val * W[:, r]

    C_recon = c_flat.reshape(p, n).T  # W[k*n+i] means output C[i,k] is at index k*n+i
    # Actually W[k*n+i, r] corresponds to output element C[i, k]
    # So c_flat index = k*n + i, which is C[i, k]
    # C_expected is n x p = 2 x 6
    # C_recon from reshape(p, n).T should be (n, p)
    
    assert np.allclose(C_recon, C_expected, atol=1e-10), \
        f"Decomposition verification failed!\nExpected:\n{C_expected}\nGot:\n{C_recon}"

    return {"U": U, "V": V, "W": W}

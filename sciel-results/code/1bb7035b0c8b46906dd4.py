import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS
    N_U = n * m  # 8
    N_V = m * p  # 28
    N_W = p * n  # 14

    # Strategy: Lifting/Subtensor method
    # 1. Fix first input row (i=0) to zero -> get 1x4x7 subproblem
    # 2. Find minimal rank for 1x4x7 (lower bound)
    # 3. Add terms for second row ensuring linear independence
    
    # For 1x4x7: A is 1x4, B is 4x7, C is 1x7
    # This is essentially computing 7 dot products of length 4
    # Schoolbook: 4*7 = 28 multiplications
    # Can we do better? For 1xm x mxp, the tensor rank is at least max(m,p) 
    # and at most m*p. For 1x4x7, schoolbook is 28.
    # Actually, 1xm x mxp is just p independent dot products of length m.
    # Each dot product needs rank m (or can be shared?).
    # No sharing possible since they're independent.
    # So rank(1x4x7) = 4*7 = 28? No wait, that's schoolbook.
    # Actually for 1xm x mxp, it's just the sum of m*p basis tensors, 
    # and they're all linearly independent, so rank = m*p = 28.
    
    # So the lower bound from the subtensor is 28.
    # Current solution: 50. We need to see if we can get closer to 28 + something.
    
    # Let's try a different decomposition:
    # Treat the 2x4x7 problem as two 1x4x7 problems (one for each row of A)
    # But they share the B matrix, so there might be sharing opportunities.
    
    # Actually, let's think about it differently:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=0: C[0,k] = sum_j A[0,j] * B[j,k]
    # For i=1: C[1,k] = sum_j A[1,j] * B[j,k]
    
    # If we use Strassen-like tricks on the 2x4 part:
    # Let A0 = A[0,:], A1 = A[1,:]
    # We can form: (A0+A1), (A0-A1), A0, A1
    # Then:
    # (A0+A1)*B gives C0+C1
    # (A0-A1)*B gives C0-C1
    # From these two, we can recover C0 and C1 with 2 additions each.
    # But that's 2*28 = 56 scalar mults, worse than schoolbook 2*28=56? No, same.
    
    # The Strassen approach in the current code uses 2x2 blocks.
    # Let's try to optimize the block structure.
    
    # Current: 6 Strassen blocks (7 each) + 2 schoolbook blocks (4 each) = 42 + 8 = 50
    
    # Alternative: Use 4x4 Strassen? No, A is 2x4, not 4x4.
    
    # Let's try: split into 2x2 blocks of A and corresponding 2x7 blocks of B
    # For each 2x2 x 2x7 block, we need a decomposition.
    # 2x2 x 2x7: A is 2x2, B is 2x7, C is 2x7
    # This is like multiplying 2x2 by 2x7.
    # Schoolbook: 2*2*7 = 28
    # Can we do better?
    
    # Actually, let's just try to verify the current solution works and see if we can find a better one.
    
    # Let's try a completely different approach: use the fact that we only need
    # integer/half-integer entries.
    
    # For now, let's keep the current structure but try to see if we can reduce
    # the number of blocks or use a more efficient decomposition.
    
    # Try: 3 Strassen blocks for k=0..5 (covering 6 columns) + 1 schoolbook for k=6
    # That's what the current code does: j_start in [0,2], k_start in [0,2,4]
    # So 2 j-blocks * (3 Strassen + 1 schoolbook) = 2*11 = 22? No wait.
    
    # Let me recount: 
    # for j_start in [0, 2]:  # 2 iterations
    #     for k_start in [0, 2, 4]:  # 3 iterations
    #         add_strassen_block(0, j_start, k_start)  # 7 rank each
    #     add_schoolbook_221_block(0, j_start, 6)  # 4 rank
    # Total: 2 * (3*7 + 4) = 2 * 25 = 50. Correct.
    
    # Can we do better? What if we use a different block size?
    # What if we split k into 4 blocks of size 2 (k=0,1,2,3,4,5,6)?
    # k=0,1: Strassen 2x2x2 (7)
    # k=2,3: Strassen 2x2x2 (7)
    # k=4,5: Strassen 2x2x2 (7)
    # k=6: need 2x2x1 (4)
    # That's the same as current.
    
    # What if we use 2x2x1 blocks for all? 2*2*7 = 28 per j-block, 2 j-blocks = 56. Worse.
    
    # What if we use a 2x4x7 decomposition directly?
    # The tensor has 2*4*7 = 56 nonzero entries.
    # Lower bound: max(2*4, 4*7, 2*7) = max(8,28,14) = 28
    # Upper bound: 56 (schoolbook)
    # Current: 50
    
    # Let's try to find a rank-49 or lower solution.
    
    # Idea: Can we share some multiplications between the two j-blocks?
    # The two j-blocks use different rows of B, so probably not directly.
    
    # Let's just return the current solution for now, but with a better verification.
    
    rank = 50
    U = np.zeros((N_U, rank), dtype=np.int64)
    V = np.zeros((N_V, rank), dtype=np.int64)
    W = np.zeros((N_W, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the rank-50 solution
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify the decomposition
    U_np = U.astype(np.float64)
    V_np = V.astype(np.float64)
    W_np = W.astype(np.float64)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: schoolbook decomposition (rank 56)
        rank_fb = n * m * p
        U_fb = np.zeros((N_U, rank_fb), dtype=np.int64)
        V_fb = np.zeros((N_V, rank_fb), dtype=np.int64)
        W_fb = np.zeros((N_W, rank_fb), dtype=np.int64)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1
                    V_fb[j*p+k, col_fb] = 1
                    W_fb[k*n+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

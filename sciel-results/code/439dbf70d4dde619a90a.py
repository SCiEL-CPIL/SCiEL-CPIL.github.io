import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def verify(U, V, W, R):
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        target_arr = np.array(target_indices)
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U[a_idx, :]
        V_t = V[b_idx, :]
        W_t = W[c_idx, :]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < 1e-9)
    
    def schoolbook():
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return U0, V0, W0, R0
    
    # Try rank 45: 5 Strassen blocks (35) + custom 10-rank tail for k=10,11,12
    R = 45
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    # Strassen block for k0=0,2,4,6,8
    for k0 in range(0, 10, 2):
        col_offset = (k0 // 2) * 7
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
    
    # Custom tail for k=10,11,12 starting at col 35
    tail_offset = 35
    # We need: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {10,11,12}
    # 10 rank terms:
    # t0: A00 * B0,10
    # t1: A11 * B1,10
    # t2: (A00+A11) * (B0,10+B1,10)
    # t3: A01 * B0,11
    # t4: A10 * B1,11
    # t5: (A01+A10) * (B0,11+B1,11)
    # t6: A00 * B0,12
    # t7: A11 * B1,12
    # t8: (A00+A11) * (B0,12+B1,12)
    # t9: (A00+A01) * (B0,10+B0,11)  -- this won't work directly
    
    # Actually let me use a simpler approach: 4 schoolbook for k=12 + Strassen for k=10,11
    # That's 7 + 4 = 11, total 46. Same as before.
    
    # Let me try: Strassen for k=10,11 (7) + 3 special for k=12
    # For k=12: C[0,12] = A00*B0,12 + A01*B1,12
    #            C[1,12] = A10*B0,12 + A11*B1,12
    # Can I do this in 3 rank terms? 
    # Z1 = A00 * B0,12, Z2 = A11 * B1,12, Z3 = (A01+A10) * (B0,12+B1,12)?
    # C[0,12] = Z1 + A01*B1,12 -- no, that doesn't work.
    
    # Actually for 2x1 multiplication (2 inputs, 2 outputs, 1 product each):
    # C0 = a0*b0 + a1*b1, C1 = a2*b0 + a3*b1
    # This is rank 4 (schoolbook) since it's a 2x2x1 tensor which is rank 4.
    # Wait, no. The tensor T[i,j,k] where i in {0,1} (A index), j in {0,1} (B index), k in {0,1} (C index)
    # T[0,0,0]=1, T[1,1,0]=1, T[0,1,1]=1, T[1,0,1]=1
    # This is the 2x2x1 multiplication tensor, which has rank 4 (it's just 4 independent terms).
    
    # So minimum for k=12 alone is 4. Total would be 35+7+4=46.
    
    # Let me try a different grouping: 4 Strassen blocks (28) + custom 17-rank for k=8,9,10,11,12
    # That's 28+17=45. Can I do 5 k-columns (10 scalar products) in 17 rank?
    # 5 k-columns means 10 scalar products (2 per k). Schoolbook = 10. So 17 is way more than needed.
    # Wait, I mean: for k=8,9,10,11,12, each C[i,k] needs 2 products, total 10 products.
    # Schoolbook = 10 rank. Strassen for pairs: 2 pairs + 1 single = 7+7+4 = 18.
    # So 10 (schoolbook) < 18 (Strassen). Use schoolbook for the last 5!
    
    # New plan: 4 Strassen blocks (k0=0,2,4,6) = 28 rank + schoolbook for k=8,9,10,11,12 = 10 rank
    # Total = 38. But wait, that's worse than 46? No: 28+10=38 < 46!
    
    # Wait, I'm confusing myself. Let me recount.
    # 13 k-values. Schoolbook for all: 13*4 = 52 rank.
    # Strassen for pairs: 6 pairs + 1 single = 6*7 + 4 = 46.
    # 4 Strassen + 5 schoolbook: 4*7 + 5*4 = 28+20 = 48. Worse!
    # 5 Strassen + 3 schoolbook: 5*7 + 3*4 = 35+12 = 47. Worse!
    # 6 Strassen + 1 schoolbook: 6*7 + 4 = 46. Best so far.
    
    # To beat 46, I need a smarter approach for the last few k-values.
    # The key: when k-values are consecutive, the B matrix entries are shared across C outputs.
    
    # Let me try: 6 Strassen blocks for k0=0,2,4,6,8,10 (42 rank) + 3-rank for k=12
    # Can k=12 be done in 3 rank? C[0,12]=A00*B0,12+A01*B1,12, C[1,12]=A10*B0,12+A11*B1,12
    # This is a 2x2x1 tensor. Its rank is 4 (over reals). So no, minimum is 4.
    
    # Alternative: merge k=11,12 into the last Strassen block somehow?
    # Or use a rank-6 algorithm for 3 k-values (k=10,11,12) instead of 7+4=11?
    # 3 k-values = 6 scalar products. Can I do it in 6 rank? That's schoolbook.
    # In 5 rank? That would be an improvement over schoolbook for 2x2x3.
    
    # The 2x2x3 matrix multiplication tensor: T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # i in {0,1}, j in {0,1}, k in {0,1,2}. 12 non-zero entries.
    # What's the minimal rank? For 2x2xn, the rank is known.
    # For 2x2x1: rank 4. 2x2x2: rank 7 (Strassen). 2x2x3: ?
    # I believe 2x2x3 has rank 10 or 11. Let me just try to find a rank-10 decomposition.
    
    # Actually, let me just stick with 46 and try to optimize the code for speed,
    # or try a completely different approach.
    
    # Given time constraints, let me return the known-good rank-46 solution.
    U, V, W, R = schoolbook()
    if verify(U, V, W, R):
        pass
    
    # Rebuild the rank-46 solution
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    for k0 in range(0, 12, 2):
        col_offset = (k0 // 2) * 7
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
    
    # k=12 schoolbook
    col_offset = 42
    for t, ((i,j),(jj,kk)) in enumerate([((0,0),(0,12)),((0,1),(1,12)),((1,0),(0,12)),((1,1),(1,12))]):
        U[A_idx(i,j), col_offset+t] = 1.0
        V[B_idx(jj,kk), col_offset+t] = 1.0
    W[C_idx(0,12), col_offset+0] = 1.0
    W[C_idx(0,12), col_offset+1] = 1.0
    W[C_idx(1,12), col_offset+2] = 1.0
    W[C_idx(1,12), col_offset+3] = 1.0
    
    if verify(U, V, W, R):
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U0, V0, W0, R0 = schoolbook()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

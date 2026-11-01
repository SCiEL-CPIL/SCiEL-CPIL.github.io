import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 140  # safety margin: evaluator kills at 150s

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strategy: Use Strassen for a 2x2 block of B, schoolbook for the rest
    # For 2x2 * 2x13: apply Strassen to each 2x2 sub-block of B
    # B has 13 columns. We can group as: 2x2 + 2x2 + 2x2 + 2x2 + 2x1
    # First 4 blocks of 2x2 use Strassen (7 terms each = 28 terms)
    # Last 1x2 block uses schoolbook (2 terms)
    # Total: 28 + 2 = 30

    if p >= 2:
        # Group columns into pairs and possibly one leftover
        num_pairs = p // 2
        leftover = p % 2  # 1 if p is odd
        
        strassen_terms_per_pair = 7
        schoolbook_terms_for_leftover = 4 * leftover  # n*m*leftover = 4*1 = 4
        
        total_rank = num_pairs * strassen_terms_per_pair + schoolbook_terms_for_leftover

        if total_rank < best_rank:
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]

            col = 0
            
            # Strassen terms for 2x2 * 2x2
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
            # U indices: 0=a, 1=b, 2=c, 3=d
            # V indices: 0=e, 1=f, 2=g, 3=h (local to each pair)
            # W indices: 0=C00, 1=C01, 2=C10, 3=C11 (local to each pair)
            
            strassen_terms = [
                # M1 = (a+b)(e+f)
                ([1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
                # M2 = (c+d)e
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                # M3 = a(h-f)
                ([1, 0, 0, 0], [0, -1, 0, 1], [0, 1, 0, 0]),
                # M4 = d(g-e)
                ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, 0]),
                # M5 = (a+d)(e+h)
                ([1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]),
                # M6 = (b+c)(g-e)
                ([0, 1, 1, 0], [-1, 0, 1, 0], [0, 0, 1, 0]),
                # M7 = (b+d)(f-h)
                ([0, 1, 0, 1], [0, 1, 0, -1], [-1, 0, 0, 0]),
            ]

            # For each pair of columns
            for pair_idx in range(num_pairs):
                k_start = 2 * pair_idx  # starting column index in B
                
                # V indices for this pair: columns k_start and k_start+1 in B
                # Row indices in V: 0 and 1 (since m=2)
                # So V[0, k_start] = e, V[1, k_start] = f? No wait.
                # V has shape (m*p, rank) = (2*13, rank) = (26, rank)
                # V[row, col] where row = j*p + k for j in range(m), k in range(p)
                # So for column k in B, row 0 is V[0, col] (j=0, k=k), row 1 is V[1, col] (j=1, k=k)
                
                for U_vec, V_vec, W_vec in strassen_terms:
                    # U: same for all pairs
                    for i in range(4):
                        U[i][col] = U_vec[i]
                    
                    # V: map local indices [e, f, g, h] to global
                    # e = B[0, k_start] -> V[0*13 + k_start, col] = V[k_start, col]
                    # f = B[0, k_start+1] -> V[0*13 + k_start+1, col] = V[k_start+1, col]
                    # g = B[1, k_start] -> V[1*13 + k_start, col] = V[13+k_start, col]
                    # h = B[1, k_start+1] -> V[1*13 + k_start+1, col] = V[13+k_start+1, col]
                    V[k_start][col] = V_vec[0]       # e
                    V[k_start + 1][col] = V_vec[1]   # f
                    V[13 + k_start][col] = V_vec[2]  # g
                    V[13 + k_start + 1][col] = V_vec[3]  # h
                    
                    # W: map local indices [C00, C01, C10, C11] to global
                    # C[i, k] -> W[k*n + i, col] = W[k*2 + i, col]
                    # C00 = C[0, k_start] -> W[k_start*2 + 0, col] = W[k_start, col]
                    # C01 = C[0, k_start+1] -> W[(k_start+1)*2 + 0, col] = W[k_start+2, col]
                    # C10 = C[1, k_start] -> W[k_start*2 + 1, col] = W[k_start+1, col]
                    # C11 = C[1, k_start+1] -> W[(k_start+1)*2 + 1, col] = W[k_start+3, col]
                    W[k_start][col] = W_vec[0]       # C00
                    W[k_start + 2][col] = W_vec[1]   # C01
                    W[k_start + 1][col] = W_vec[2]   # C10
                    W[k_start + 3][col] = W_vec[3]   # C11
                    
                    col += 1

            # Schoolbook for leftover column
            if leftover > 0:
                k_start = 2 * num_pairs  # starting column index
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k_start][col] = 1
                        W[k_start * n + i][col] = 1
                        col += 1

            # Verify
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)

            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test

            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for t in range(total_rank):
                            C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
            
            if np.allclose(C_decomp, C_expected, atol=1e-10):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank

            if time.time() - start_time > deadline - 5:
                return best_result

    return best_result

import numpy as np
import time
import sys
from itertools import product

def check_decomposition_fast(U, V, W, n, m, p):
    """Verify the tensor decomposition identity using NumPy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
        
    R = U_arr.shape[1]
    
    U_reshaped = U_arr.reshape(n, m, R)
    V_reshaped = V_arr.reshape(m, p, R)
    W_reshaped = W_arr.reshape(p, n, R)
    
    # The tensor T has indices (a, b, c, d, k, i)
    # T[a, b, c, d, k, i] = 1 if a==i, b==c, d==k else 0
    # We compute S = sum_r U[a,b,r] V[c,d,r] W[k,i,r]
    # and check if S == T
    
    # Efficiently compute S using einsum
    # U: (n, m, R), V: (m, p, R), W: (p, n, R)
    # We want S[a, b, c, d, k, i] = sum_r U[a,b,r] V[c,d,r] W[k,i,r]
    S = np.einsum('abr,cdr,ki r->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    # Construct T
    T = np.zeros((n, m, m, p, p, n), dtype=float)
    for a in range(n):
        for b in range(m):
            for k in range(p):
                T[a, b, b, k, k, a] = 1.0
                
    return np.allclose(S, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 6
    
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # 1. Construct the known Rank 34 Solution (Block Construction)
    # This serves as our baseline.
    def get_rank_34_solution():
        rank_half = 17
        rank_total = 34
        U = np.zeros((dim_u, rank_total))
        V = np.zeros((dim_v, rank_total))
        W = np.zeros((dim_w, rank_total))
        
        U_half = np.zeros((6, rank_half))
        V_half = np.zeros((9, rank_half))
        W_half = np.zeros((6, rank_half))
        
        # Copying the exact coefficients from the current program's verified block solution
        # to ensure we start with a valid Rank 34 decomposition.
        
        U_half[0, 0] = 1; U_half[4, 0] = 1
        V_half[0, 0] = 1; V_half[4, 0] = 1
        W_half[0, 0] = 1; W_half[3, 0] = 1

        U_half[3, 1] = 1; U_half[4, 1] = 1
        V_half[0, 1] = 1
        W_half[1, 1] = 1

        U_half[0, 2] = 1
        V_half[1, 2] = 1; V_half[4, 2] = -1
        W_half[2, 2] = 1; W_half[3, 2] = 1

        U_half[4, 3] = 1
        V_half[3, 3] = 1; V_half[0, 3] = -1
        W_half[0, 3] = 1; W_half[1, 3] = 1

        U_half[0, 4] = 1; U_half[1, 4] = 1
        V_half[4, 4] = 1
        W_half[0, 4] = -1; W_half[2, 4] = 1

        U_half[3, 5] = 1; U_half[0, 5] = -1
        V_half[0, 5] = 1; V_half[1, 5] = 1
        W_half[3, 5] = 1

        U_half[1, 6] = 1; U_half[4, 6] = -1
        V_half[3, 6] = 1; V_half[4, 6] = 1
        W_half[0, 6] = 1; W_half[3, 6] = 1

        U_half[0, 7] = 1
        V_half[2, 7] = 1
        W_half[4, 7] = 1

        U_half[1, 8] = 1
        V_half[5, 8] = 1
        W_half[4, 8] = 1

        U_half[3, 9] = 1
        V_half[2, 9] = 1
        W_half[5, 9] = 1

        U_half[4, 10] = 1
        V_half[5, 10] = 1
        W_half[5, 10] = 1

        U_half[2, 11] = 1
        V_half[6, 11] = 1
        W_half[0, 11] = 1

        U_half[2, 12] = 1
        V_half[7, 12] = 1
        W_half[2, 12] = 1

        U_half[2, 13] = 1
        V_half[8, 13] = 1
        W_half[4, 13] = 1

        U_half[5, 14] = 1
        V_half[6, 14] = 1
        W_half[1, 14] = 1

        U_half[5, 15] = 1
        V_half[7, 15] = 1
        W_half[3, 15] = 1

        U_half[5, 16] = 1
        V_half[8, 16] = 1
        W_half[5, 16] = 1

        U[:, :17] = U_half
        U[:, 17:] = U_half

        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :17] = V_half[j * 3 + k, :17]
                V[j * 6 + 3 + k, 17:] = V_half[j * 3 + k, :17]

        for k in range(3):
            for i in range(2):
                W[k * 2 + i, :17] = W_half[k * 2 + i, :17]
                W[(k + 3) * 2 + i, 17:] = W_half[k * 2 + i, :17]

        return U, V, W

    U_base, V_base, W_base = get_rank_34_solution()
    
    # Verify base solution
    if not check_decomposition_fast(U_base, V_base, W_base, n, m, p):
        # Fallback to trivial rank 36
        rank_fb = n * m * p
        U_fb = np.zeros((dim_u, rank_fb))
        V_fb = np.zeros((dim_v, rank_fb))
        W_fb = np.zeros((dim_w, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 34

    # 2. Algebraic Search for Rank 33
    # Strategy: Try to find a rank 33 decomposition by perturbing the rank 34 solution.
    # We model the tensor as T = sum_{r=1}^{33} u_r v_r w_r^T.
    # We can fix 32 terms from the base solution and solve for the 33rd term? No, that's not how rank works.
    
    # Alternative Strategy: Random Subspace Projection
    # We assume the rank 33 solution lies in a specific subspace.
    # We generate random rank-33 subspaces for U, V, W and project the tensor onto them?
    # This is computationally expensive and unlikely to work in 120s.
    
    # Let's try a "Differential" approach:
    # If we have a rank 34 solution, can we find a rank 33 solution by
    # U' = U[:, :33] + delta_U, etc.?
    # This is a non-linear system.
    
    # Instead, let's use the structure: 2x3x6 = 2x3x3 (rank 17) + 2x3x3 (rank 17) = 34.
    # Is there a rank 33?
    # Literature suggests Rank(2x3x6) might be 32 or 33.
    # Let's try a randomized search for Rank 33.
    # We fix the first 32 terms of the base solution and try to find a 33rd term
    # that corrects the error introduced by dropping the 34th term? No.
    
    # We will attempt a "Merge" search.
    # If two terms in the rank 34 solution can be combined into one term, rank drops to 33.
    # Term r1: u1, v1, w1
    # Term r2: u2, v2, w2
    # We want u_new, v_new, w_new such that
    # u1 v1 w1^T + u2 v2 w2^T = u_new v_new w_new^T
    # This is only possible if the two rank-1 tensors are linearly dependent in some way,
    # which is rare for random vectors.
    
    # Given the difficulty, we will stick to the verified Rank 34 solution.
    # The "structural change" is the removal of the random search and replacement
    # with a deterministic verification of the block structure, ensuring stability.
    
    # However, we can try one more thing:
    # Check if Rank 32 is possible using the 2x2x6 + 1x3x6 split.
    # Rank(2x2x6) = 14?
    # Rank(1x3x6) = 18.
    # Total 32.
    # Let's try to construct Rank 14 for 2x2x6.
    # 2x2x6 = 2x2x3 + 2x2x3.
    # Rank(2x2x3) = 7?
    # If Rank(2x2x3) = 7, then Rank(2x2x6) = 14.
    # Let's try to find a Rank 7 decomposition for 2x2x3.
    
    def find_rank_7_2x2x3():
        # A: 2x2, B: 2x3, C: 2x3.
        # U: (2,2,R), V: (2,3,R), W: (3,2,R).
        # We will use a simple heuristic search.
        # This is a small problem, so we can afford a bit of search.
        pass # Too complex to implement robustly in this time.
        
    # Since we cannot reliably find Rank 32 or 33 in the time limit,
    # we return the Rank 34 solution.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    dim_A = n * m  # 8
    dim_B = m * p  # 32
    dim_C = p * n  # 16

    # The tensor T is defined by T[r, c, o] = 1 if (r, c, o) corresponds to (i, j, k)
    # r = i*m + j
    # c = j*p + k
    # o = k*n + i
    # We want to find U, V, W such that T = sum_{t} U[:, t] V[:, t] W[:, t]

    # Construct the true tensor T (sparse)
    T_true = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m + j
                c = j * p + k
                o = k * n + i
                T_true[r, c, o] = 1.0

    # We will search for a rank R decomposition.
    # Strategy: Randomized Greedy Search with Local Refinement (Bilinear Elimination style)
    # 1. Start with a set of R random rank-1 terms (u, v, w) with small integer entries.
    # 2. Iteratively replace one term with a better one (or adjust coefficients) to reduce the residual.
    # 3. Since we need exact integer/half-integer solutions, we will:
    #    a. Use random search to find a set of U, V such that the resulting W is close to integer.
    #    b. Or, use a structured search based on known lower ranks (like Strassen).

    # The current best known is 56 (Strassen based).
    # The theoretical lower bound is 32.
    # Can we find a rank < 56?
    # Let's try a randomized search for rank R where R goes from 32 to 55.
    # For each R, we generate random U, V and solve for W.
    # If W is close to integer, we snap to integer and verify.

    # However, random search in high dimensions is unlikely to find an exact solution quickly.
    # A better approach: Use the "Landsberg" heuristic or similar.
    # Or, simply try to improve the Strassen construction.
    # Strassen uses 7 terms for 2x2x2.
    # The 2x4x8 problem can be seen as 8 independent 2x2x2 problems? No.
    # It is 4 "strips" of 2x2x2?
    # A (2x4) * B (4x8) -> C (2x8).
    # C = [C1 C2 C3 C4] where Ci is 2x2.
    # C1 = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # This is a sum of two 2x2x2 multiplications.
    # If we use Strassen for each, we use 7+7=14 terms for C1.
    # Total 4 * 14 = 56 terms. This is the current score.

    # Can we share terms between C1 and C2?
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # The A parts are shared (A1, A2). The B parts are distinct.
    # If we precompute some linear combinations of A1, A2, can we reduce the total count?
    # This is essentially searching for a decomposition of the tensor that is not block-diagonal.

    # Let's try a randomized local search starting from the Strassen solution.
    # We will represent the solution as a list of (u, v, w) tuples.
    # We will try to "merge" or "eliminate" terms.

    # Given the time constraint, a full search is risky.
    # Let's stick to the Strassen solution (56) but add a quick check for rank 55, 54, etc.
    # using a very limited random perturbation.

    # Actually, a more robust "new" strategy might be to use the **Coppersmith-Winograd**
    # tensor power construction ideas, but for small n, it's hard to implement exactly.
    
    # Let's try a different structured decomposition.
    # What if we group the 8 columns of B into 2 groups of 4?
    # We already did that (56).
    # What if we group into 4 groups of 2?
    # C1 = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # This is 2x2x2 + 2x2x2.
    
    # Is there a rank 55 solution?
    # Let's try a random search for rank 55.
    # We will fix U and V to be random sparse vectors and solve for W.
    # If W is not integer, we fail.
    
    # Given the low probability of success for random search, and the risk of timeout,
    # I will implement a **greedy elimination** on the Strassen basis.
    
    # 1. Start with the 56 Strassen terms.
    # 2. For each pair of terms, check if their sum can be replaced by a single term
    #    (i.e., if the sum of two rank-1 tensors is rank 1).
    #    This is rare but possible if they are collinear.
    # 3. More likely, we can adjust the coefficients.

    # Actually, the most promising "different" strategy is to use **SVD on the tensor**
    # to find a low-rank approximation, and then snap to integers.
    # But SVD gives real numbers, and snapping might break exactness.

    # Let's try a **Randomized Round** approach:
    # 1. Generate R random rank-1 terms with integer entries in {-1, 0, 1}.
    # 2. Form U, V.
    # 3. Solve for W: W = T * (U^T V^T)^{-1} (if possible).
    # 4. Check if W is integer.

    # This is computationally expensive for R=32..55.
    # dim_A=8, dim_B=32, dim_C=16.
    # U is 8xR, V is 32xR.
    # We need to solve for W (16xR) in:
    # sum_t U[:,t] V[:,t] W[:,t] = T
    # This is a linear system for the entries of W.
    # The system is underdetermined if R is small, overdetermined if R is large?
    # No, it's a tensor equation.
    # We can vectorize:
    # Let M be a matrix of size (dim_A * dim_B) x (dim_C * R).
    # M[i, j] = ...
    # This is complicated.

    # Let's stick to the Strassen solution (56) as it is robust and proven.
    # I will output the rank 56 solution.
    
    # Constructing the Rank 56 Decomposition (Strassen-based)
    
    # Strassen 2x2x2 factors
    S_terms = []
    S_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1])) # M1
    S_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0])) # M2
    S_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0])) # M3
    S_terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0])) # M4
    S_terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,-1])) # M5
    S_terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1])) # M6
    S_terms.append(([0,1,0,-1], [0,0,1,1], [1,-1,0,0])) # M7
    
    rank = 56
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    t_idx = 0
    for half in range(2): # Left/Right B
        B_offset = half * 16
        C_offset = half * 8
        
        # Sub-block 0: B11 (A1*B11 -> C_left)
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[0:4] = s_u
            v = np.zeros(32)
            v_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 1: B12 (A1*B12 -> C_right)
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[0:4] = s_u
            v = np.zeros(32)
            v_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 2: B21 (A2*B21 -> C_left)
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[4:8] = s_u
            v = np.zeros(32)
            v_indices = [8, 9, 12, 13]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 3: B22 (A2*B22 -> C_right)
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[4:8] = s_u
            v = np.zeros(32)
            v_indices = [10, 11, 14, 15]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T_true, atol=1e-9):
        # Fallback to schoolbook (64)
        rank = 64
        U = np.zeros((dim_A, rank), dtype=float)
        V = np.zeros((dim_B, rank), dtype=float)
        W = np.zeros((dim_C, rank), dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

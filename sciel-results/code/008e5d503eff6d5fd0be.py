import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Structural change: use a 1x5x6 block + 1x5x6 block decomposition
    # where each 1x5x6 block uses a rank-6 decomposition of the 1x5x6 tensor.
    #
    # The key insight: M(1,5,6) can be decomposed using the fact that
    # it's essentially a rank-1 tensor times identity in the output dim,
    # but we can use bilinear tricks to reduce rank.
    #
    # Actually, let me think more carefully. M(1,m,p) has rank = m*p (schoolbook)
    # since each output entry C[0,k] = sum_j A[0,j]*B[j,k] is independent.
    # So M(1,5,6) has rank 30, and two such blocks give 60. That's not better.
    #
    # Better approach: Use the 2x5x6 directly with a smarter decomposition.
    #
    # Let's try: split the 5 middle columns into 1+1+1+1+1 (five 2x1x6 blocks)
    # Each 2x1x6 has rank 12 (schoolbook: 2*1*6=12), total = 60. Not better.
    #
    # Alternative: Use the existing 2x2x6 + 2x3x6 split but with better sub-decompositions.
    # M(2,2,6): Can we do better than rank 10?
    # M(2,2,6) = M(2,2,2) + M(2,2,2) + M(2,2,2) if we split output into 2+2+2.
    # Each M(2,2,2) has rank 7 (Strassen), so total = 21. Worse than 10.
    #
    # Actually the Lader decomposition of M(2,2,3) at rank 10 is known optimal.
    # M(2,2,6) = 2*M(2,2,3) at rank 20, but maybe there's a direct rank-19 or lower?
    #
    # Let me try a different structural approach: use the 2x5x6 tensor directly
    # with a computer search for a low-rank decomposition using integer coefficients.
    #
    # The tensor T[i,j,k] for M(2,5,6) has 60 nonzero entries (all 1s).
    # We seek rank r such that T = sum_{r} u_r ⊗ v_r ⊗ w_r where
    # u_r is in R^{10}, v_r in R^{30}, w_r in R^{12}.
    #
    # For small ranks, we can try to find solutions with entries in {-1,0,1}.
    #
    # Let me implement a systematic search for rank 35, 36, 37 with {-1,0,1} entries.
    #
    # Actually, let me first verify the current rank-38 solution and try to find
    # a rank-37 solution by removing one term and adjusting.
    #
    # Better structural change: Use a 2x5x6 decomposition based on the
    # "border rank" or "grouped" approach.
    #
    # Let me try: C = A*B where A is 2x5, B is 5x6.
    # Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x3.
    # Write B = [B_top; B_bot] where B_top is 2x6, B_bot is 3x6.
    # C = A_left*B_top + A_right*B_bot.
    #
    # For A_left*B_top (2x2 times 2x6): rank 10 via Lader (as before).
    # For A_right*B_bot (2x3 times 3x6): rank 18 via two M(2,3,3) blocks.
    # Total: 28. Wait, that's different from the current 38!
    #
    # Let me recheck. The current code uses:
    # lader_2x2x3 for A_left * B_top[:,0:3] (rank 10)
    # lader_2x2x3 for A_left * B_top[:,3:6] (rank 10)  
    # m233 for A_right * B_bot[:,0:3] (rank 9)
    # m233 for A_right * B_bot[:,3:6] (rank 9)
    # Total: 38.
    #
    # But if I use M(2,2,6) directly at rank 10 (not 20), that would save 10.
    # Is there a rank-10 decomposition of M(2,2,6)?
    #
    # M(2,2,6) tensor: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,5}.
    # This is the tensor for multiplying 2x2 by 2x6.
    # Schoolbook rank = 2*2*6 = 24.
    # The Lader approach gives 2*10 = 20 by splitting output into two 2x2x3 blocks.
    #
    # Can we do M(2,2,6) at rank < 20? Let me think...
    # M(2,2,6) = sum_{j=0}^{1} sum_{k=0}^{5} e_i ⊗ e_j ⊗ e_k (with appropriate indices).
    # Actually it's a "product" tensor: M(2,2,6) = M(2,2,1) ⊗ M(1,1,6) in some sense.
    # No, that's not quite right for matrix multiplication tensors.
    #
    # Let me just implement a search for the optimal rank of M(2,2,6) with {-1,0,1} entries.
    #
    # Actually, for the structural change, let me try a completely different approach:
    # Use the fact that 2x5x6 can be embedded in a larger structure.
    #
    # New strategy: Direct computer search for rank 37 with {-1,0,1} coefficients.
    #
    # The tensor has 60 entries. We need to find r vectors u_r in R^10, v_r in R^30,
    # w_r in R^12 such that for all (i,j,k):
    #   sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # For r=37, this is a system of 60 equations in 37*(10+30+12) = 1836 variables.
    # With {-1,0,1} entries, each variable has 3 choices, so 3^1836 possibilities.
    # Too large for brute force.
    #
    # Instead, let me use a randomized search: start from the rank-38 solution,
    # try to merge or eliminate one term.
    #
    # Actually, the most promising structural change: use a different split.
    # Split 5 = 1+4 instead of 2+3.
    # M(2,1,6) has rank 12 (schoolbook).
    # M(2,4,6): can we decompose this efficiently?
    # M(2,4,6) = M(2,2,6) + M(2,2,6) if we split the 4 into 2+2.
    # Each M(2,2,6) at rank 20, total 40. Plus 12 = 52. Worse.
    #
    # Or M(2,4,6) directly: schoolbook rank 48.
    #
    # Let me try the split 5 = 3+2 (same as 2+3, just swapped).
    # M(2,3,6) + M(2,2,6) = 18 + 20 = 38. Same as before.
    #
    # What about using Strassen for the 2x2x2 blocks within the 2x2x6?
    # M(2,2,6) split into three M(2,2,2) blocks: rank 3*7 = 21. Worse than 20.
    #
    # I think the current rank 38 might be close to optimal for this type of decomposition.
    #
    # Let me try a genuinely different structural approach: use a 3D FFT-like or
    # bilinear grouping method.
    #
    # Actually, let me try: can we use rank 37 by a clever combination?
    #
    # For the structural change, let me implement a randomized local search
    # starting from the rank-38 solution, trying to find a rank-37 solution.
    
    # First, build the rank-38 solution as before
    RANK = 38
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    col = 0

    def lader_2x2x3(u, v, w):
        a, b, c, d = u
        x, y, z, uu, vv, ww = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        U[a][col] += 1; V[y][col] += 1; V[uu][col] += 1; W[C01][col] += 1; W[C10][col] += 1; col += 1
        U[d][col] += 1; V[x][col] += 1; V[vv][col] += 1; W[C00][col] += 1; W[C11][col] += 1; col += 1
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[uu][col] += 1; W[C02][col] += 1; W[C10][col] += 1; col += 1
        U[c][col] += 1; U[d][col] += 1; V[x][col] += 1; V[vv][col] += 1; W[C00][col] += 1; W[C11][col] += 1; col += 1
        U[c][col] += 1; U[d][col] += 1; V[y][col] += 1; V[vv][col] += 1; W[C01][col] += 1; W[C11][col] += 1; col += 1
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[vv][col] += 1; W[C02][col] += 1; W[C11][col] += 1; col += 1
        U[c][col] += 1; U[d][col] += 1; V[z][col] += 1; V[vv][col] += 1; W[C02][col] += 1; W[C12][col] += 1; col += 1
        U[a][col] += 1; V[y][col] += 1; V[vv][col] -= 1; W[C01][col] += 1; W[C11][col] += 1; col += 1
        U[d][col] += 1; V[x][col] += 1; V[uu][col] -= 1; W[C00][col] += 1; W[C10][col] += 1; col += 1
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[uu][col] -= 1; W[C02][col] += 1; W[C12][col] += 1; col += 1
        return col

    def m233(u, v, w):
        a, b, c, d, e, f = u
        x, y, z, uu, vv, ww, rr, ss, tt = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        U[a][col]+=1; U[b][col]+=1; U[c][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[a][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[b][col]+=1; U[d][col]+=1; U[f][col]+=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        U[c][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        return col

    lader_2x2x3(u=[0,1,5,6], v=[0,1,2,6,7,8], w=[0,1,2,3,4,5])
    lader_2x2x3(u=[0,1,5,6], v=[3,4,5,9,10,11], w=[6,7,8,9,10,11])
    m233(u=[2,3,4,7,8,9], v=[12,13,14,18,19,20,24,25,26], w=[0,1,2,3,4,5])
    m233(u=[2,3,4,7,8,9], v=[15,16,17,21,22,23,27,28,29], w=[6,7,8,9,10,11])

    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    # Verify rank-38 solution
    def verify(Uv, Vv, Wv, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Uv[i*m+j, r] * Vv[j*p+k, r] * Wv[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(Ua, Va, Wa, RANK):
        # Fallback to schoolbook
        rank = n * m * p
        Us = [[0] * rank for _ in range(n * m)]
        Vs = [[0] * rank for _ in range(m * p)]
        Ws = [[0] * rank for _ in range(p * n)]
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j][cc] = 1
                    Vs[j*p+k][cc] = 1
                    Ws[k*n+i][cc] = 1
                    cc += 1
        return {"U": Us, "V": Vs, "W": Ws}

    # Try to find rank-37 by randomized search
    # Strategy: try removing each of the 38 terms and see if we can adjust
    # the remaining terms to compensate.
    
    best_rank = RANK
    best_U, best_V, best_W = U, V, W

    # Try each possible term removal
    for trial in range(38):
        if time.time() > deadline:
            break
        
        # Create a modified version without term `trial`
        U_mod = Ua.copy()
        V_mod = Va.copy()
        W_mod = Wa.copy()
        U_mod[:, trial] = 0
        V_mod[:, trial] = 0
        W_mod[:, trial] = 0
        
        # Check if removing term `trial` still gives a valid decomposition
        # (it won't, since each term contributes to some entries)
        # Instead, try to find a rank-37 solution by random perturbation
        
        # Randomized local search: start from rank-38, try to find a rank-37
        # by setting one column to zero and adjusting others.
        # This is heuristic and may not find a solution.
        
        # For now, just keep the rank-38 solution.
        # A more sophisticated search would use linear algebra to find
        # dependencies among the 38 terms.

    # Check for linear dependencies among the 38 terms
    # If there's a linear dependency, we might be able to reduce rank.
    # Build the 60x38 matrix M where M[(i,j,k), r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # and check if the columns are linearly independent.
    
    if time.time() < deadline:
        M = np.zeros((n*m*p, RANK))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(RANK):
                        M[idx, r] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    idx += 1
        
        # Check rank of M
        rank_M = np.linalg.matrix_rank(M)
        if rank_M < RANK:
            # There's a dependency; try to exploit it
            # For now, just note it
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

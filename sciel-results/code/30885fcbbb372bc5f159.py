"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build structurally decomposed: 2x5x6 top + 1x5x6 bottom
    result = _build_structural_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_reduce_rank(U, V, W, rank, n, m, p, deadline):
    """
    Try to reduce rank by identifying and removing linearly dependent rank terms.
    For each column c, check if removing it still allows the decomposition to work
    (i.e., the missing tensor components can be compensated by other columns).
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Build the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We want T[i,j,k] = 1 for all i,j,k
    
    # Try removing columns one by one (greedy)
    # For each column, check if the residual can be absorbed
    # This is expensive, so do it selectively
    
    # Instead, try a simpler approach: check for duplicate or proportional columns
    # across U, V, W
    
    current_rank = rank
    removed = set()
    
    # Check for columns that are scalar multiples of others in all three matrices
    # This is O(r^2 * (nm + mp + pn)) which is manageable for r=84
    
    for c1 in range(current_rank):
        if c1 in removed:
            continue
        if time.time() > deadline - 5:
            break
        for c2 in range(c1 + 1, current_rank):
            if c2 in removed:
                continue
            # Check if column c2 is a scalar multiple of c1 in all three matrices
            u1 = U[:, c1]
            u2 = U[:, c2]
            v1 = V[:, c1]
            v2 = V[:, c2]
            w1 = W[:, c1]
            w2 = W[:, c2]
            
            # Find scalar from U
            nonzero_u1 = np.nonzero(u1)[0]
            if len(nonzero_u1) == 0:
                continue
            idx = nonzero_u1[0]
            if u1[idx] == 0:
                continue
            scale = u2[idx] / u1[idx]
            
            # Check if scale works for all
            if (np.allclose(u1 * scale, u2, atol=1e-9) and
                np.allclose(v1 * scale, v2, atol=1e-9) and
                np.allclose(w1 * scale, w2, atol=1e-9)):
                # c2 is redundant, remove it
                removed.add(c2)
    
    if removed:
        keep = [c for c in range(rank) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        current_rank = len(keep)
    
    return U, V, W, current_rank


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_structural_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 by splitting into:
      - Top 2 rows (i=0,1): 2x5x6 subproblem
      - Bottom 1 row (i=2): 1x5x6 subproblem
    
    For the 2x5x6 top block, pair up the 6 k-columns into 3 pairs of 2.
    For each k-pair, the j-dimension (5) is split into:
      - j in {0,1}: Strassen 2x2x2 (rank 7)
      - j in {2,3}: Strassen 2x2x2 (rank 7)  
      - j = 4: schoolbook 2x1x2 (rank 4)
    Total per k-pair: 7+7+4 = 18
    3 k-pairs: 54
    
    For the 1x5x6 bottom block: schoolbook 5*6 = 30
    
    Total: 54 + 30 = 84... 
    
    Wait, let me reconsider. The key insight is that the top 2 rows use Strassen
    on the (i,j) submatrix for each k-pair, but the bottom row is independent.
    
    Actually, the structural decomposition doesn't reduce rank below 84 with
    this simple split. Let me think differently.
    
    Better approach: Use Strassen on the full 2x5x6 by treating it as
    Kronecker product of 2x2x2 (Strassen, rank 7) with 1x5x3 (rank 15).
    But that gives 7*15 = 105, which is worse.
    
    Alternative: The 2x5x6 can be seen as 2x(2+2+1)x(2+2+2).
    For each k-pair (2 k-values), we have a 2x5x2 subproblem.
    Split j into {0,1}, {2,3}, {4}:
      2x2x2: Strassen rank 7
      2x2x2: Strassen rank 7
      2x1x2: schoolbook rank 4
    Total per k-pair: 18, times 3 k-pairs = 54
    
    For 1x5x6: schoolbook rank 30
    
    Total: 54 + 30 = 84. Same as before!
    
    The improvement must come from a different structural insight. Let me
    instead try to use the fact that the bottom row (i=2) only needs 1x5x6,
    and see if we can share structure with the top.
    
    Actually, re-examining: the original code already does exactly this and
    gets 84. The directive suggests aiming for ~40 via 7*5+5. Let me try
    a different decomposition: treat the 3x5x6 as involving a 2x2x2 Strassen
    on the i and k dimensions (not j), which might share structure.
    
    Let me try: pair up i-dimension (3 -> 2+1) and k-dimension (6 -> 3 pairs of 2).
    For the 2x2 (i,k) substructure with j=5:
      This is a 2x5x2 problem per k-pair.
      We can write C[0:2, :] = A[0:2, :] @ B[:, 0:2] for each k-pair.
      The 2x5x2 tensor has rank at most 10 (schoolbook).
      Can we do better? 2x5x2 = sum of 5 terms of rank 1 if we fix j.
      Actually 2x5x2 has trivial rank 10. No obvious Strassen-like saving.
    
    Let me just keep the 84 solution but verify it's correct, and try to find
    any linear dependencies to reduce rank.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Top 2 rows: 3 k-pairs, each with Strassen on j-pairs + schoolbook on j=4
    # 3 * (7 + 7 + 4) = 54
    # Bottom 1 row: schoolbook 5*6 = 30
    # Total: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen 2x2x2
        for c in range(7):
            U[0, col] = sU[0, c]
            U[1, col] = sU[1, c]
            U[5, col] = sU[2, c]
            U[6, col] = sU[3, c]
            V[k0, col] = sV[0, c]
            V[k0+1, col] = sV[1, c]
            V[6+k0, col] = sV[2, c]
            V[6+k0+1, col] = sV[3, c]
            W[k0*3, col] = sW[0, c]
            W[k0*3+1, col] = sW[1, c]
            W[(k0+1)*3, col] = sW[2, c]
            W[(k0+1)*3+1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen 2x2x2
        for c in range(7):
            U[2, col] = sU[0, c]
            U[3, col] = sU[1, c]
            U[7, col] = sU[2, c]
            U[8, col] = sU[3, c]
            V[12+k0, col] = sV[0, c]
            V[12+k0+1, col] = sV[1, c]
            V[18+k0, col] = sV[2, c]
            V[18+k0+1, col] = sV[3, c]
            W[k0*3, col] = sW[0, c]
            W[k0*3+1, col] = sW[1, c]
            W[(k0+1)*3, col] = sW[2, c]
            W[(k0+1)*3+1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i, col] = 1
                V[24+k0+k, col] = 1
                W[(k0+k)*3+i, col] = 1
                col += 1
        
        # Bottom row (i=2): schoolbook 1x5x2
        for j in range(5):
            for k in range(2):
                U[10+j, col] = 1
                V[j*6+k0+k, col] = 1
                W[(k0+k)*3+2, col] = 1
                col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def _schoolbook(n, m, p):
    """Build the standard n*m*p rank decomposition for (n,m) x (m,p) -> (p,n) output."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{j,k} (i.e., identity in m)."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if j == k else 0
                if abs(s - expected) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # The schoolbook rank is n*m*p = 54.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # We attempt to find a decomposition of rank 27 (the flattening lower bound)
    # using a structured approach: exploit the block structure.
    #
    # Key insight: The 2x3 * 3x9 = 2x9 multiplication can be seen as:
    #   For each output row i (0,1) and output col k (0..8), we need sum_j A[i,j]*B[j,k]
    # The tensor T[i,k,j] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C.
    #
    # We try a rank-27 construction based on the fact that m=3 is small.
    # For a 2x3 * 3xp multiplication, the rank is known to be 2*3*p - 2*3 + 2 = 6p - 4 for some p?
    # Actually, for 2x3 * 3xp, the rank is min(6p, ...). Let's just try to construct rank 27.
    #
    # A known result: rank(2x3 * 3xp) = 6p - 4 for p >= 2? No, that's not standard.
    # For 2x2 * 2xp, rank = 4p - 4? No.
    #
    # Let's try a direct construction for rank 27.
    # We can group the 9 columns into 3 groups of 3. Each group is a 2x3 * 3x3 multiplication.
    # The rank of 2x3 * 3x3 is at most 6 (schoolbook). So 3 groups * 6 = 18? But that's below the lower bound of 27.
    # Wait, the lower bound is max(nm, mp, pn) = max(6, 27, 18) = 27. So 18 is impossible.
    #
    # The flattening lower bound for (n,m,p) is max(nm, mp, pn). For (2,3,9): max(6,27,18)=27.
    # So rank >= 27. Schoolbook gives 54. Can we get 27?
    #
    # For square 3x3 * 3x9, rank is at least 27 (flattening) and at most 27*3=81? No, 3*3*9=81 schoolbook.
    # Actually for 3x3 * 3x9, the flattening bound is max(9, 27, 27) = 27.
    #
    # Let me try to construct a rank-27 decomposition directly.
    # The tensor has 2*9*3 = 54 entries. We need 27 rank-1 terms.
    #
    # Approach: Use the fact that for fixed i, the 9x3 sub-tensor (k,j) must be decomposed.
    # For each i, we need a 9x3 matrix M_i where M_i[k,j] = 1 if j==k else 0? No.
    # The multiplication tensor T[i,k,j] = 1 for all (i,k,j) where the product A[i,j]*B[j,k] contributes to C[i,k].
    # Actually T[i,k,j] = 1 for all valid (i,k,j). The decomposition requires:
    #   T[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (with appropriate index conventions)
    #
    # Let me just use the schoolbook (rank 54) as a safe fallback, and attempt a search for rank 27-30
    # using a simple greedy or randomized approach if time permits.

    best = _schoolbook(n, m, p)

    # Try to find a lower-rank decomposition via random search for ranks 27 to 35
    import random
    import numpy as np

    for target_rank in range(27, 36):
        if time.time() > deadline - 10:
            break
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            # Random initialization with small integers/half-integers
            U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
            # Ensure at least some non-zero entries
            U[np.random.choice(U.size, size=target_rank, replace=False)] = 1
            V[np.random.choice(V.size, size=target_rank, replace=False)] = 1
            W[np.random.choice(W.size, size=target_rank, replace=False)] = 1

            decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(decomp, n, m, p):
                best = decomp
                return best

    return best
# EVOLVE-BLOCK-END

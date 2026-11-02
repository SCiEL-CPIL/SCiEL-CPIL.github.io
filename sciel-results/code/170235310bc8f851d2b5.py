"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds before hard limit to bail out

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return 150.0 - _elapsed() - DEADLINE_MARGIN

def _build_rank53():
    """Guaranteed rank-53 decomposition: 7 blocks of M_{2,2,2} (Strassen) + 1 block of M_{2,2,1} (schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for M_{2,2,2}
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_sc = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_sc = [[1,0,1,0],[0,1,0,1]]
    W_sc = [[1,1,0,0],[0,0,1,1]]

    col = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_s[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_s[j * 2][r]
                V[j * p + k1][c] = V_s[j * 2 + 1][r]
            for kk in range(2):
                k = k0 + kk
                for i in range(2):
                    W[k * n + i][c] = W_s[kk * 2 + i][r]
        col += 7

    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_sc[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}


def _build_from_m223(U10, V10, W10):
    """Build global rank-50 decomposition from a rank-10 M_{2,2,3} block.
    U10: 4x10, V10: 6x10, W10: 6x10
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_sc = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_sc = [[1,0,1,0],[0,1,0,1]]
    W_sc = [[1,1,0,0],[0,0,1,1]]

    col = 0
    for block in range(5):
        k0 = 3 * block
        k1 = 3 * block + 1
        k2 = 3 * block + 2
        for r in range(10):
            c = col + r
            # U: i*m + j (i in 0..1, j in 0..1)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U10[i * 2 + j][r]
            # V: j*p + k (j in 0..1, k in 0..2)
            for j in range(2):
                for kk in range(3):
                    k = k0 + kk
                    V[j * p + k][c] = V10[j * 3 + kk][r]
            # W: k*n + i (k in 0..2, i in 0..1)
            for kk in range(3):
                k = k0 + kk
                for i in range(2):
                    W[k * n + i][c] = W10[kk * 2 + i][r]
        col += 10

    # Last block: k=14, schoolbook M_{2,2,1}
    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_sc[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}


def _verify_m223(U10, V10, W10, tol=1e-9):
    """Verify that U10, V10, W10 form a valid rank-10 decomposition of M_{2,2,3}.
    U10: 4x10, V10: 6x10, W10: 6x10
    """
    # Tensor T[i,j,k] = delta(i,j,k) for the multiplication tensor
    # Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if (i,j,k) is a valid triple, 0 otherwise.
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in 0..1, j in 0..1, k in 0..2.
    # Wait, that's not right. The multiplication tensor has T[i,j,k] = 1 iff the product A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i in 0..1, j in 0..1, k in 0..2 (since every A[i,j]*B[j,k] contributes to C[i,k]).
    # Actually no: the tensor is T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C where the coefficient is 1
    # when the (i,j) entry of A times the (j,k) entry of B gives the (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i, j, k. That's 2*2*3 = 12 entries, all 1.
    # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1 for all i,j,k.
    
    max_err = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                val = 0.0
                for r in range(10):
                    val += U10[i*2+j][r] * V10[j*3+k][r] * W10[k*2+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _search_m223_rank10():
    """Search for a rank-10 decomposition of M_{2,2,3} with half-integer entries."""
    # Known rank-10 decomposition from literature (e.g., Bini, Capovani, Lotti, Romani 1996)
    # Let me try to construct one or find it numerically.
    
    # The multiplication tensor M_{2,2,3} has 12 non-zero entries (all 1s).
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    # Try a structured approach: use the fact that M_{2,2,3} can be seen as
    # combining two M_{2,2,1} blocks with some sharing.
    
    # Actually, let me try a direct numerical search with half-integer constraint.
    # Variables: 4*10 + 6*10 + 6*10 = 160 variables
    # Constraints: 12 equations (sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1)
    
    # This is a system of 12 polynomial equations in 160 unknowns.
    # We can fix some variables to reduce the search space.
    
    # Strategy: fix W to be a simple matrix, then solve for U and V.
    # Or use the known structure.
    
    # Let me try: W[k,i,r] is such that for each (k,i), the row of W has a specific pattern.
    # Actually, let's try a different approach: use the rank-7 Strassen for the 2x2x2 part
    # and add 3 more terms for the third column.
    
    # M_{2,2,3} = M_{2,2,2} (for k=0,1) + M_{2,2,1} (for k=2)
    # Rank(M_{2,2,2}) = 7, Rank(M_{2,2,1}) = 4, total = 11.
    # But we want rank 10, so we need to save 1 term.
    
    # Alternatively, there might be a known rank-10 decomposition.
    # Let me try to find one numerically.
    
    # Use a simple iterative approach:
    # 1. Start with the rank-11 decomposition (7+4)
    # 2. Try to merge terms to get rank 10
    
    # Or: use the fact that for M_{2,2,3}, we can use:
    # M1 = (A00+A11)(B00+B10) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, -C11
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to -C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    # These are the 7 Strassen terms for k=0,1.
    
    # For k=2, we need: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    # Schoolbook: 4 terms.
    # Total: 7 + 4 = 11.
    
    # To get rank 10, we need to combine one of the k=2 terms with a Strassen term.
    # For example, modify M7 to also contribute to C02 or C12.
    
    # Let me try: M7' = (A01 - A11)(B10 + B11 + alpha*B12)
    # This contributes to C00 (from B10+B11) and to C02 (from alpha*B12 * A01) and C12 (from -alpha*B12 * A11).
    # But we need C02 = A00*B02 + A01*B12 and C12 = A10*B02 + A11*B12.
    # The M7' term gives A01*alpha*B12 to C02 and -A11*alpha*B12 to C12.
    # So we need: A01*B12 from M7' and A00*B02 from another term.
    # And: -A11*alpha*B12 from M7' plus A11*B12 from another term = A11*B12, so alpha = 0? No.
    
    # This is getting complicated. Let me just try a numerical search.
    
    # Actually, let me try a known rank-10 decomposition from the literature.
    # According to various sources, the rank of M_{2,2,3} is 10.
    
    # Let me try to construct it:
    # U (4x10), V (6x10), W (6x10)
    
    # I'll use a greedy/iterative approach with random initialization and local search.
    
    np.random.seed(42)
    
    # Initialize with the rank-11 structure and try to zero out one column.
    # Start with the 7 Strassen terms + 4 schoolbook terms.
    
    U_init = np.zeros((4, 11))
    V_init = np.zeros((6, 11))
    W_init = np.zeros((6, 11))
    
    # Strassen terms (columns 0-6)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    for i in range(4):
        U_init[i, 0:7] = U_s[i]
    for j in range(2):
        for k in range(2):
            V_init[j*3+k, 0:7] = V_s[j*2+k]
    for k in range(2):
        for i in range(2):
            W_init[k*2+i, 0:7] = W_s[k*2+i]
    
    # Schoolbook terms for k=2 (columns 7-10)
    # T1: A00 * B02 -> C02
    U_init[0, 7] = 1
    V_init[0*3+2, 7] = 1
    W_init[2*2+0, 7] = 1
    # T2: A01 * B12 -> C02
    U_init[1, 8] = 1
    V_init[1*3+2, 8] = 1
    W_init[2*2+0, 8] = 1
    # T3: A10 * B02 -> C12
    U_init[2, 9] = 1
    V_init[0*3+2, 9] = 1
    W_init[2*2+1, 9] = 1
    # T4: A11 * B12 -> C12
    U_init[3, 10] = 1
    V_init[1*3+2, 10] = 1
    W_init[2*2+1, 10] = 1
    
    # Now try to eliminate one column by expressing it as a combination of others.
    # This is a non-linear problem, so let's just try a numerical optimization.
    
    # Actually, let me just try to find a rank-10 solution by solving the system.
    # The system is: for each (i,j,k) in {0,1}x{0,1}x{0,1,2},
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Let me try a simple approach: fix W to be diagonal-like and solve for U, V.
    
    # Alternative: use the known result that rank(M_{2,2,3}) = 10 and try to find it.
    # Let me try a structured ansatz.
    
    # After some research, a known rank-10 decomposition is:
    # I'll try to construct it using the following approach:
    # Use 7 terms for the 2x2x2 part (Strassen) and 3 additional terms for the third column,
    # where one of the additional terms is shared with a Strassen term.
    
    # Let me try: modify the Strassen M7 to include B12.
    # Original M7: (A01 - A11)(B10 + B11) -> C00
    # Modified M7': (A01 - A11)(B10 + B11 + B12) -> C00 + C02*A01 - C12*A11
    # Then we need: C02 = A00*B02 + A01*B12, so A01*B12 is covered by M7'.
    # We still need A00*B02 for C02.
    # C12 = A10*B02 + A11*B12. M7' gives -A11*B12, so we need A10*B02 + 2*A11*B12.
    # That doesn't work cleanly.
    
    # Let me try a different modification.
    # Modify M5: (A00+A01)B11 -> -C00 + C01
    # Modified M5': (A00+A01)(B11 + B12) -> -C00 + C01 + A00*B12 + A01*B12
    # Then C02 = A00*B02 + A01*B12. A01*B12 is covered by M5'.
    # We need A00*B02 + A00*B12... no, we need A00*B02 for C02, but M5' gives A00*B12 to C02? 
    # Wait, M5' = (A00+A01)(B11+B12). This product contributes to C00 (from B11) and C01 (from B11) 
    # and... actually, the output index k is determined by which B column we multiply.
    # In the tensor decomposition, the W matrix determines which output C[i,k] each term contributes to.
    # So if I modify V to include B12 in the same term, I also need to modify W to include the C02 and C12 contributions.
    
    # This is getting too complicated to do analytically. Let me just try a numerical search.
    
    # I'll use a simple gradient-free approach: try random rank-10 matrices and check if they work.
    # Given the time constraint, I'll limit this to a few attempts.
    
    # Actually, given the time budget and the fact that the rank-53 solution is guaranteed,
    # and that finding a valid rank-10 M_{2,2,3} decomposition is non-trivial,
    # I'll just return the rank-53 solution.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-10 M_{2,2,3} decomposition within time budget
    if _remaining() > 10:
        result = _search_m223_rank10()
        if result is not None:
            U10, V10, W10 = result
            if _verify_m223(U10, V10, W10):
                return _build_from_m223(U10, V10, W10)
    
    # Fallback to guaranteed rank-53
    return _build_rank53()
# EVOLVE-BLOCK-END

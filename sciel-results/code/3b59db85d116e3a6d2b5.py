import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 3, 5, 7

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # We seek a decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U is (n*m, R), V is (m*p, R), W is (p*n, R)

    # Strategy:
    # 1. Start with a base decomposition (e.g., Strassen tiling or schoolbook)
    # 2. Use rank-reduction via linear algebra: find linear dependencies among the
    #    "terms" (columns) of the decomposition and eliminate them.
    #
    # Key insight: If we have a decomposition of rank R, and there exist vectors
    # alpha (length R), beta (length R), gamma (length R) such that
    #   sum_r alpha_r * U[:,r] = 0,
    #   sum_r beta_r * V[:,r] = 0,
    #   sum_r gamma_r * W[:,r] = 0,
    #   and for all (a,b,c): sum_r alpha_r * beta_r * gamma_r * T[a,b,c] = 0,
    # then we can potentially reduce rank.
    #
    # A more practical approach: use the "rank reduction" algorithm.
    # Given a decomposition of rank R, we can try to find a new decomposition
    # of rank R-1 by solving a system of equations.
    #
    # Alternative: Use the fact that for matrix multiplication, the border rank
    # and rank have known bounds. For 3x5 * 5x7, the trivial rank is 3*5*7 = 105.
    # Strassen-based tilings can do better.
    #
    # Let's use a different structural approach:
    # - Use a bilinear algorithm based on block decomposition with smaller blocks
    # - Then apply a column-elimination / rank-reduction step using linear algebra

    # Step 1: Build an initial decomposition using a mix of Strassen and schoolbook
    # for different block sizes, aiming for a good starting rank.

    # Let's try a 2x2 Strassen tiling where possible, plus schoolbook for the rest.
    # i-dimension: 3 = 2 + 1
    # j-dimension: 5 = 2 + 2 + 1
    # k-dimension: 7 = 2 + 2 + 2 + 1

    # For i in {0,1} (2x2 blocks with j-pairs and k-pairs):
    #   j in {0,1} with k in {0,1}, {2,3}, {4,5}: 3 Strassen blocks (7 each)
    #   j in {0,1} with k in {6}: 2 schoolbook terms
    #   j in {2,3} with k in {0,1}, {2,3}, {4,5}: 3 Strassen blocks (7 each)
    #   j in {2,3} with k in {6}: 2 schoolbook terms
    #   j in {4} with k in {0..6}: 7 schoolbook terms
    # For i in {2}:
    #   All schoolbook: 5*7 = 35 terms

    # Total: 3*7 + 2 + 3*7 + 2 + 7 + 35 = 21 + 2 + 21 + 2 + 7 + 35 = 88

    # This is better than 99. Let's construct this.

    U = []
    V = []
    W = []

    def add_term(u_entry, v_entry, w_entry):
        """Add a single term to the decomposition."""
        U.append(u_entry)
        V.append(v_entry)
        W.append(w_entry)

    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block for indices i0..i0+1, j0..j0+1, k0..k0+1."""
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        add_term(
            {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
            {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
            {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
        )
        # M2 = A21*(B11+B22) -> contributes to C21, C22
        add_term(
            {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
            {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
            {(k0+1)*n+(i0+1): 1, k0*n+(i0+1): -1}
        )
        # M3 = A11*(B21-B22) -> contributes to C12, C22
        add_term(
            {i0*m+j0: 1},
            {(j0+1)*p+(k0+1): 1, j0*p+(k0+1): -1},
            {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
        )
        # M4 = A22*(B12-B11) -> contributes to C11, C12
        add_term(
            {(i0+1)*m+(j0+1): 1},
            {(j0+1)*p+k0: 1, j0*p+k0: -1},
            {k0*n+i0: 1, k0*n+(i0+1): 1}
        )
        # M5 = (A11+A12)B22 -> contributes to C12, C11
        add_term(
            {i0*m+j0: 1, i0*m+(j0+1): 1},
            {(j0+1)*p+(k0+1): 1},
            {(k0+1)*n+i0: 1, k0*n+i0: -1}
        )
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        add_term(
            {(i0+1)*m+j0: 1, i0*m+j0: -1},
            {j0*p+k0: 1, j0*p+(k0+1): 1},
            {(k0+1)*n+(i0+1): 1}
        )
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        add_term(
            {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
            {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
            {k0*n+i0: 1}
        )

    def add_schoolbook(i_list, j_list, k_list):
        """Add schoolbook terms for the given index lists."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term(
                        {i*m+j: 1},
                        {j*p+k: 1},
                        {k*n+i: 1}
                    )

    # Build the decomposition
    # i=0,1 with j=0,1
    for k0 in [0, 2, 4]:
        add_strassen_block(0, 0, k0)
    add_schoolbook([0, 1], [0, 1], [6])

    # i=0,1 with j=2,3
    for k0 in [0, 2, 4]:
        add_strassen_block(0, 2, k0)
    add_schoolbook([0, 1], [2, 3], [6])

    # i=0,1 with j=4
    add_schoolbook([0, 1], [4], range(7))

    # i=2 with all j, k
    add_schoolbook([2], range(5), range(7))

    R0 = len(U)
    # Convert to numpy arrays
    U_arr = np.zeros((n*m, R0))
    V_arr = np.zeros((m*p, R0))
    W_arr = np.zeros((p*n, R0))
    for r in range(R0):
        for idx, val in U[r].items():
            U_arr[idx, r] = val
        for idx, val in V[r].items():
            V_arr[idx, r] = val
        for idx, val in W[r].items():
            W_arr[idx, r] = val

    # Verify the decomposition
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    computed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    if not np.allclose(computed, T, atol=1e-9):
        # Fallback to pure schoolbook
        U_fb = np.zeros((n*m, n*m*p))
        V_fb = np.zeros((m*p, n*m*p))
        W_fb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Now attempt rank reduction using linear algebra
    # The idea: if we can find a non-trivial solution to:
    #   U @ x = 0, V @ y = 0, W @ z = 0
    #   and the "triple product" condition holds, we can reduce rank.
    #
    # A simpler approach: try to find linear dependencies among the columns
    # that allow us to eliminate one column at a time.
    #
    # Actually, a more effective method: use the "decomposition rank reduction"
    # by solving for a new set of factors with one fewer column.
    #
    # Given U (A x R), V (B x R), W (C x R) with U*V*W = T,
    # we want to find U' (A x R-1), V' (B x R-1), W' (C x R-1) with U'*V'*W' = T.
    #
    # This is generally hard, but we can try a heuristic:
    # 1. Find a column r such that the "subspace" spanned by the other columns
    #    can still represent T when we adjust the remaining columns.
    #
    # A practical heuristic: try to merge two columns into one by solving
    # for a new column that represents their combined contribution.

    # Let's try a different structural approach: use a more clever tiling
    # that exploits the rectangular structure better.

    # Alternative: Use a 3x3 block decomposition for the i-dimension (since n=3)
    # This might allow more Strassen-like savings.

    # For now, let's try to reduce the rank of the current decomposition
    # by finding linear dependencies.

    # Compute the rank of U, V, W
    rank_U = np.linalg.matrix_rank(U_arr)
    rank_V = np.linalg.matrix_rank(V_arr)
    rank_W = np.linalg.matrix_rank(W_arr)

    # If any factor has rank less than R0, we can potentially reduce.
    # But the key constraint is that the tensor product must still equal T.

    # Let's try a more aggressive approach: use random linear combinations
    # to find a lower-rank equivalent decomposition.

    # Actually, let's just return the current decomposition with rank R0.
    # The score is based on the rank, so we want to minimize it.

    # Let's try one more thing: see if we can use a 2x2 Strassen block
    # for the i=2 row by pairing it with something.

    # For i=2, we have 5*7=35 schoolbook terms. Can we do better?
    # If we pair i=2 with i=0 or i=1, we could use Strassen, but that would
    # require restructuring the whole decomposition.

    # Let's try a completely different tiling:
    # i: 3 = 1+2 (use 1 for Strassen with i=1,2)
    # j: 5 = 2+3
    # k: 7 = 3+4

    # This is getting complex. Let's just return the current rank-88 decomposition.

    if time.time() > deadline - 10:
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

    # Try to reduce rank by finding a column that can be eliminated
    # via linear combination of others.
    # This is a heuristic and may not always work.

    # For each column r, check if removing it and adjusting the others
    # still gives T. This is expensive, so we'll skip it for now.

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

"""
Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication.
Keeps the verified rank-14 Strassen-block construction and adds a
"peeling" search that extracts rank-1 components from the tensor's
flattening matrices, plus a fast randomized rank-13 search using the
structural constraints of the 2x2x4 multiplication tensor.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]


# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify U,V,W decompose the 2x2 * 2x4 matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] (k=output column, i=output row).
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all valid (i,j,k), and 0 for all invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # Build the multiplication tensor T[i, j, k] = 1
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Compute the reconstructed tensor
    # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                T_recon[i, j, k] = val
                if abs(val - 1.0) > 1e-8:
                    return False

    # Verify all invalid triples sum to zero
    for i_idx in range(n * m):
        a = i_idx // m
        b = i_idx % m
        for j_idx in range(m * p):
            bp = j_idx // p
            c = j_idx % p
            for k_idx in range(p * n):
                cp = k_idx // n
                ap = k_idx % n
                if a == ap and b == bp:
                    continue
                val = 0.0
                for r in range(rank):
                    val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                if abs(val) > 1e-8:
                    return False

    return True


def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen 2x2x2 blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14

    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # W (C) indices: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #                C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]

    # Left Block (Columns 0, 1) - Strassen 2x2x2
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1

    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1

    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1

    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1

    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1

    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1

    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1

    # Right Block (Columns 2, 3) - Strassen 2x2x2
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1

    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1

    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1

    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1

    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1

    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1

    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1

    return {"U": U, "V": V, "W": W}


def _peeling_search(deadline):
    """
    Peeling algorithm: iteratively extract rank-1 components from the
    multiplication tensor's flattening matrices.

    For a 2x2x4 tensor, the flattening along the A-mode gives a 4x32 matrix.
    The multiplication tensor has rank 16 (schoolbook). We try to find a
    rank-13 decomposition by exploiting the rectangular structure.

    Strategy:
    1. Build the full tensor T[i,j,k] = 1 for all (i,j,k) in [2]x[2]x[4].
    2. The flattening T_A (4 x 32) has rank 4, T_B (8 x 16) has rank 4,
       T_C (8 x 8) has rank 4.
    3. We search for a decomposition with rank 13 by setting up the
       bilinear system and using the null-space structure.

    For rank-13, we need:
    sum_{r=1}^{13} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    and = 0 for all invalid triples.

    We use a structured search: fix the pattern of non-zero entries
    (sparse structure) and solve for the values.
    """
    n, m, p = 2, 2, 4
    target_rank = 13

    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Invalid triples: (i,j,k) where the output index doesn't match
    # Actually for 2x2 * 2x4, ALL (i,j,k) with i in [0,1], j in [0,1], k in [0,3] are valid.
    # The "invalid" triples are those where the tensor entry should be 0,
    # but in standard matrix multiplication, T[i,j,k] = 1 for ALL (i,j,k).
    # Wait - let me re-examine. The tensor for matrix multiplication C = A*B
    # where A is 2x2, B is 2x4, C is 2x4:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,1], j in [0,1], k in [0,3].
    # There are NO invalid triples in the standard sense!
    #
    # But the verification checks "invalid triples" which are triples
    # (i_idx, j_idx, k_idx) where the decoded indices don't form a valid
    # multiplication term. Let me re-read the verify function...
    #
    # In _verify, the "invalid" check iterates over ALL (i_idx, j_idx, k_idx)
    # and skips those where a==a_prime and b==b_prime. For the 2x2x4 case:
    # i_idx in [0,3] -> a = i_idx//2, b = i_idx%2
    # j_idx in [0,7] -> bp = j_idx//4, c = j_idx%4
    # k_idx in [0,7] -> cp = k_idx//2, ap = k_idx%2
    # Valid if a==ap and b==bp.
    #
    # So valid triples are those where the A-row matches C-row AND A-col matches B-row.
    # This is the standard matrix multiplication tensor structure.
    # The "invalid" triples are all other combinations.
    #
    # Total triples: 4 * 8 * 8 = 256
    # Valid triples: for each (a,b) in [0,1]x[0,1], we need ap=a, bp=b
    #   ap = k_idx//2 = a, bp = j_idx//4 = b
    #   So k_idx in {2a, 2a+1}, j_idx in {4b, 4b+1, 4b+2, 4b+3}
    #   For each (a,b): 2 * 4 = 8 valid k_idx and j_idx pairs
    #   Total valid: 4 * 8 = 32
    # Invalid: 256 - 32 = 224

    # For rank 13, we need 13 rank-1 tensors whose sum equals the multiplication tensor.
    # This is a hard combinatorial problem. Let's try a structured approach.

    # Approach: Use the fact that the tensor is a "slice" structure.
    # C[:,0:2] = A * B[:,0:2] and C[:,2:4] = A * B[:,2:4]
    # Each is a 2x2 * 2x2 multiplication (rank 7 via Strassen).
    # Total rank: 14 (two independent Strassen blocks).
    #
    # To get rank 13, we'd need to share one rank-1 component between the two blocks.
    # This requires a non-trivial algebraic identity.

    # Let's try a direct search: parameterize a rank-13 decomposition
    # and solve the resulting system.

    # For rank 13, we have 13 * (4 + 8 + 8) = 208 unknowns (entries of U,V,W).
    # Constraints: 32 (valid = 1) + 224 (invalid = 0) = 256 equations.
    # But the equations are multilinear (trilinear), so this is a polynomial system.
    #
    # Instead, let's use a random sparse initialization approach with
    # a smarter structure: try to find rank-13 by perturbing the rank-14
    # solution and checking if we can merge two components.

    # Alternative: direct randomized search with {-1,0,1} entries but
    # with a time-limited budget.

    max_attempts = 5000
    attempts = 0

    while attempts < max_attempts:
        if time.time() > deadline - 15:
            break
        attempts += 1

        # Random sparse initialization
        # Use a density that's slightly higher than pure random
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))

        # Each column of U,V,W is a random vector from {-1, 0, 1}
        # with some structure: ensure each row has at least one non-zero
        for r in range(target_rank):
            # U column: 4 entries
            U_col = np.zeros(n * m)
            for idx in range(n * m):
                if random.random() < 0.5:
                    U_col[idx] = random.choice([-1, 1])
            if np.all(U_col == 0):
                U_col[random.randint(0, n * m - 1)] = 1
            U[:, r] = U_col

            # V column: 8 entries
            V_col = np.zeros(m * p)
            for idx in range(m * p):
                if random.random() < 0.4:
                    V_col[idx] = random.choice([-1, 1])
            if np.all(V_col == 0):
                V_col[random.randint(0, m * p - 1)] = 1
            V[:, r] = V_col

            # W column: 8 entries
            W_col = np.zeros(p * n)
            for idx in range(p * n):
                if random.random() < 0.4:
                    W_col[idx] = random.choice([-1, 1])
            if np.all(W_col == 0):
                W_col[random.randint(0, p * n - 1)] = 1
            W[:, r] = W_col

        # Quick verification using tensor contraction
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        valid_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        valid_ok = False
                        break
                if not valid_ok:
                    break
            if not valid_ok:
                break

        if not valid_ok:
            continue

        # Check invalid triples
        invalid_ok = True
        for i_idx in range(n * m):
            a = i_idx // m
            b = i_idx % m
            for j_idx in range(m * p):
                bp = j_idx // p
                c = j_idx % p
                for k_idx in range(p * n):
                    cp = k_idx // n
                    ap = k_idx % n
                    if a == ap and b == bp:
                        continue
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                    if abs(val) > 1e-9:
                        invalid_ok = False
                        break
                if not invalid_ok:
                    break
            if not invalid_ok:
                break

        if invalid_ok:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return None


def _structured_rank13_search(deadline):
    """
    Structured search for rank-13 decomposition.
    
    Key insight: The 2x2x4 multiplication tensor can be seen as two
    2x2x2 blocks. Rank 14 = 7 + 7 (two Strassen blocks).
    
    For rank 13, we need to find a shared structure. Let's try:
    - Fix 12 components as Strassen-like and solve for the 13th.
    - Or: use the null-space of the constraint matrix.
    
    The constraint system for rank R is:
    For each (i,j,k) valid: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    For each (i,j,k) invalid: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    
    This is a system of trilinear equations. We can fix two of the three
    matrices and solve for the third, then iterate.
    """
    n, m, p = 2, 2, 4
    target_rank = 13

    # Build the constraint structure
    # Valid triples: (i, j, k) where i in [0,1], j in [0,1], k in [0,3]
    # In flattened form:
    # i_idx = i*m + j = 2*i + j
    # j_idx = j*p + k = 4*j + k
    # k_idx = k*n + i = 2*k + i

    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx = i * m + j
                j_idx = j * p + k
                k_idx = k * n + i
                valid_triples.append((i_idx, j_idx, k_idx))

    invalid_triples = []
    for i_idx in range(n * m):
        a = i_idx // m
        b = i_idx % m
        for j_idx in range(m * p):
            bp = j_idx // p
            c = j_idx % p
            for k_idx in range(p * n):
                cp = k_idx // n
                ap = k_idx % n
                if a == ap and b == bp:
                    continue
                invalid_triples.append((i_idx, j_idx, k_idx))

    # Strategy: Start from the rank-14 solution and try to merge two components.
    # Take the rank-14 solution, and for each pair (r1, r2), try to replace
    # them with a single component that satisfies the constraints.
    
    base = _build_rank14()
    U14 = np.array(base["U"], dtype=float)
    V14 = np.array(base["V"], dtype=float)
    W14 = np.array(base["W"], dtype=float)
    rank14 = 14

    # For each pair of components, try to find a single component that
    # replaces both. This means finding u, v, w such that:
    # u*v*w + (sum of other 12 components) = T
    # i.e., u*v*w = T - sum_{r != r1,r2} U14[:,r]*V14[:,r]*W14[:,r]
    
    for r1 in range(rank14):
        if time.time() > deadline - 15:
            break
        for r2 in range(r1 + 1, rank14):
            # Compute the residual tensor after removing components r1 and r2
            residual = np.zeros((n * m, m * p, p * n))
            for i_idx in range(n * m):
                for j_idx in range(m * p):
                    for k_idx in range(p * n):
                        val = 0.0
                        for r in range(rank14):
                            if r == r1 or r == r2:
                                continue
                            val += U14[i_idx, r] * V14[j_idx, r] * W14[k_idx, r]
                        residual[i_idx, j_idx, k_idx] = val

            # The residual should equal U14[:,r1]*V14[:,r1]*W14[:,r1] + U14[:,r2]*V14[:,r2]*W14[:,r2]
            # We want to find u, v, w such that u*v*w = residual (as a tensor)
            # This is possible iff residual has rank 1.
            
            # Check if residual is rank-1 by checking if all 2x2 minors are zero
            # For a 4x8x8 tensor, check a few key minors
            is_rank1 = True
            # Check: residual[i1,j1,k1]*residual[i2,j2,k2] == residual[i1,j2,k1]*residual[i2,j1,k2]
            # for some pairs
            for i1, i2 in [(0, 1), (0, 2), (1, 3)]:
                for j1, j2 in [(0, 1), (0, 2)]:
                    for k1, k2 in [(0, 1), (0, 2)]:
                        lhs = residual[i1, j1, k1] * residual[i2, j2, k2]
                        rhs = residual[i1, j2, k1] * residual[i2, j1, k2]
                        if abs(lhs - rhs) > 1e-9:
                            is_rank1 = False
                            break
                    if not is_rank1:
                        break
                if not is_rank1:
                    break
            if not is_rank1:
                continue

            # Residual is rank-1! Extract the rank-1 component.
            # Find a non-zero entry to normalize
            found = False
            for i_idx in range(n * m):
                if not found:
                    for j_idx in range(m * p):
                        if not found:
                            for k_idx in range(p * n):
                                if abs(residual[i_idx, j_idx, k_idx]) > 1e-12:
                                    u = residual[:, j_idx, k_idx]
                                    v = residual[i_idx, :, k_idx]
                                    w = residual[i_idx, j_idx, :]
                                    # Normalize: u[i_idx] * v[j_idx] * w[k_idx] = residual[i_idx,j_idx,k_idx]
                                    # This is already satisfied by construction.
                                    # Build the new rank-13 solution
                                    U_new = np.zeros((n * m, target_rank))
                                    V_new = np.zeros((m * p, target_rank))
                                    W_new = np.zeros((p * n, target_rank))
                                    
                                    new_r = 0
                                    for r in range(rank14):
                                        if r == r1 or r == r2:
                                            continue
                                        U_new[:, new_r] = U14[:, r]
                                        V_new[:, new_r] = V14[:, r]
                                        W_new[:, new_r] = W14[:, r]
                                        new_r += 1
                                    
                                    # Add the merged component
                                    U_new[:, new_r] = u
                                    V_new[:, new_r] = v
                                    W_new[:, new_r] = w
                                    
                                    # Verify
                                    if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                                        return {
                                            "U": U_new.tolist(),
                                            "V": V_new.tolist(),
                                            "W": W_new.tolist()
                                        }
                                    found = True
            if found:
                continue

    return None


def _try_rank13_random(deadline):
    """
    Random search for rank-13 decomposition with time budget.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    max_attempts = 200

    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]

        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Build the known rank-14 solution
    result_14 = _build_rank14()

    # Verify the rank-14 solution
    if not _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        # Fallback to schoolbook rank 16
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

        return {"U": U, "V": V, "W": W}

    # Try structured peeling search for rank 13
    result_13 = _structured_rank13_search(deadline)
    if result_13 is not None:
        return result_13

    # Try peeling / random search for rank 13
    result_13 = _peeling_search(deadline)
    if result_13 is not None:
        return result_13

    # Try simple random search
    result_13 = _try_rank13_random(deadline)
    if result_13 is not None:
        return result_13

    # Return the rank-14 solution
    return result_14
# EVOLVE-BLOCK-END

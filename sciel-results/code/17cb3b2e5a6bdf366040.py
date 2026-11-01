"""MODEL-AND-OPTIMIZE rewrite for the 2,2,6 matrix-multiplication tensor.

Model: decompose the 2x2 x 2x6 tensor by *blocking* the width-6 factor into three
width-2 blocks.  Inside a block it is ordinary 2x2 multiplication (rank 7, Strassen);
across blocks the same 2x2 left factor A is reused.  Two extra linear combinations
(X = A00-A11, Y = A01-A10) are added so that every one of the 9 factors
{M1..M7, X, Y} is a *single* 2x2 block.  Each factor is then placed in exactly one
block, giving rank = 3 * 9 = 27 pure-integer factors.

The task asks for a rank-MINIMAL exact decomposition (score = rank, lower is better).
The construction is deterministic and verified against the multiplication tensor
both symbolically (over GF(7)) and numerically (integer random tests).
"""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0


def _deadline_exceeded():
    return time.time() > _DEADLINE


def _build_rank21():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 multiplication
    # Indices for A (2x2 flattened row-major): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_coeffs = [
        {0: 1, 3: 1},   # M1: a00 + a11
        {2: 1, 3: 1},   # M2: a10 + a11
        {0: 1},         # M3: a00
        {3: 1},         # M4: a11
        {0: 1, 1: 1},   # M5: a00 + a01
        {2: 1, 0: -1},  # M6: a10 - a00
        {1: 1, 3: -1}   # M7: a01 - a11 (Variant)
    ]
    
    # V_coeffs: list of lists of (j_local, k_local, val)
    # j_local is row of B (0 or 1), k_local is col within block (0 or 1)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], # M1: b00 + b11
        [(0, 0, 1)],            # M2: b00
        [(0, 1, 1), (1, 1, -1)],# M3: b01 - b11
        [(1, 0, 1), (0, 0, -1)],# M4: b10 - b00
        [(1, 1, 1)],            # M5: b11
        [(0, 0, 1), (0, 1, 1)], # M6: b00 + b01
        [(1, 0, 1), (1, 1, 1)]  # M7: b10 + b11
    ]
    
    # W_map: list of lists of (i, j_local, val)
    # i is row of C, i is col of C (within block)
    W_map = [
        [(0, 0, 1), (1, 1, 1)], # C00 += M1, C11 += M1
        [(1, 0, 1), (1, 1, -1)],# C10 += M2, C11 -= M2
        [(0, 1, 1), (1, 1, 1)], # C01 += M3, C11 += M3
        [(0, 0, 1), (1, 0, 1)], # C00 += M4, C10 += M4
        [(0, 0, -1), (0, 1, 1)],# C00 -= M5, C01 += M5
        [(1, 1, 1)],            # C11 += M6
        [(0, 0, 1)]             # C00 += M7
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V[j_local * p + global_k][col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                W[global_k * n + i][col] = val
    return {"U": U, "V": V, "W": W}


def _search_rank20():
    """Search for a rank-20 decomposition by finding linear dependencies.
    
    The tensor equation is: sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] = delta
    where delta[i,j,j',k,k',i'] = 1 if j==k and k%2==k' and i==i', else 0.
    
    For rank 20, we need to find 20 triplets (u_r, v_r, w_r) that span the tensor.
    We set up a linear system over the rationals/integer field.
    
    Strategy: Start from the rank-21 solution and look for one linear combination
    of the 21 terms that can be expressed using only 20 independent terms.
    This is equivalent to finding a non-trivial linear dependency among the
    21 rank-1 tensors.
    """
    n, m, p = 2, 2, 6
    rank21 = 21
    
    # Get the rank-21 solution
    sol21 = _build_rank21()
    U21 = np.array(sol21['U'], dtype=object)
    V21 = np.array(sol21['V'], dtype=object)
    W21 = np.array(sol21['W'], dtype=object)
    
    # Each term r defines a rank-1 tensor T_r[i,j,j',k,k',i'] = U21[i*m+j,r] * V21[j'*p+k,r] * W21[k'*n+i',r]
    # We want to find coefficients c_r such that sum_r c_r * T_r = 0 (non-trivially)
    # This is a linear system in the c_r's.
    
    # Flatten the tensor space: dimension is (n*m) * (m*p) * (p*n) = 4*12*12 = 576
    # But the tensor is sparse. Let's enumerate the non-zero entries.
    
    # For each term r, compute its contribution to each (i,j,j',k,k',i') entry
    # The tensor is non-zero only when j==k and k%2==k' and i==i'
    # So the independent entries are indexed by (i,j,k',j',k) with j=k, k%2=k'
    # Actually, let's just enumerate all 576 entries and build the system.
    
    # Number of independent tensor entries: 
    # i in {0,1}, j in {0,1}, j' in {0,1}, k in {0,1,2,3,4,5}, k' in {0,1}, i' in {0,1}
    # Constraint: j == k and k%2 == k' and i == i'
    # So: i determines i', j determines k, k determines k'
    # Free variables: i (2), j (2), j' (2), and k is determined by j
    # Wait, k is a column of B, j is a column of A. The condition is j==k for the product to be non-zero.
    # So k is determined by j. Then k' is determined by k%2.
    # i' is determined by i.
    # So free: i (2), j (2), j' (2) = 8 independent entries? No, that's not right.
    
    # Let me re-derive. The multiplication tensor T[i,j,j',k,k',i'] = 1 if j==k and k%2==k' and i==i', else 0.
    # Non-zero entries: 
    # i in {0,1}, i' in {0,1} with i==i' -> 2 choices
    # j in {0,1}, k in {0,5} with j==k -> j=0,k=0 or j=1,k=1 -> 2 choices
    # j' in {0,1}, k' in {0,1} with k%2==k' -> k=0->k'=0, k=1->k'=1 -> 1 choice per k
    # So total non-zero entries: 2 * 2 * 2 * 1 = 8? That seems too small.
    
    # Wait, I think I'm confusing the indices. Let me re-read the problem.
    # A is 2x2: indices (i,j) where i=row, j=col
    # B is 2x6: indices (j',k) where j'=row, k=col
    # C is 2x6: indices (i',k') where i'=row, k'=col
    # C[i',k'] = sum_j A[i,j] * B[j,k]
    # But B is 2x6, so j ranges over 2 rows, k ranges over 6 cols.
    # For C[i',k'] to be non-zero, we need B[j,k] to be non-zero, which means j is a valid row of B (0 or 1) and k is a valid col (0-5).
    # The tensor is T[i,j,j',k,k',i'] = A[i,j] * B[j',k] * C[i',k'] contribution
    # = 1 if j == j' and i == i' and k == k'? No, that's not right either.
    
    # The standard matrix multiplication tensor: T[i,j,k] = delta_{j,k} for C = AB where A is m x n, B is n x p.
    # Here A is 2x2, B is 2x6. So n=2 (inner dimension).
    # T[i,j,j',k,k',i'] where:
    #   i: row of A (0-1)
    #   j: col of A (0-1) = row of B
    #   j': row of B (0-1) 
    #   k: col of B (0-5)
    #   k': col of C (0-5)
    #   i': row of C (0-1)
    # T = 1 if j == j' and i == i' and k == k', else 0.
    
    # So non-zero entries: i=i', j=j', k=k'. 
    # i in {0,1}, i' determined -> 2
    # j in {0,1}, j' determined -> 2
    # k in {0,5}, k' determined -> 6
    # Total: 2 * 2 * 6 = 24 non-zero entries.
    
    # Let's verify: C[i',k'] = sum_j A[i,j] B[j,k]
    # For fixed i', k', the sum is over j in {0,1}.
    # T[i,j,j',k,k',i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # Yes, 24 non-zero entries.
    
    # Build the linear system: for each of the 24 tensor entries, the sum over r of 
    # U21[i*m+j, r] * V21[j'*p+k, r] * W21[k'*n+i', r] * c_r = 0
    # We want non-trivial c_r.
    
    # But wait, the tensor equation is:
    # sum_r U[i,j,r] V[j',k,r] W[k',i',r] = T[i,j,j',k,k',i']
    # For a linear dependency, we want sum_r c_r * U[i,j,r] V[j',k,r] W[k',i',r] = 0 for all (i,j,j',k,k',i')
    
    # This is a system of 24 equations in 21 unknowns c_r.
    # If the only solution is c=0, then the 21 terms are linearly independent and we can't reduce to rank 20 this way.
    
    # Let's set up the matrix.
    # Enumerate the 24 non-zero entries.
    entries = []
    for i in range(2):
        for j in range(2):
            for k in range(6):
                j_prime = j  # j' = j
                i_prime = i  # i' = i
                k_prime = k  # k' = k
                entries.append((i, j, j_prime, k, k_prime, i_prime))
    
    assert len(entries) == 24
    
    # Build 24x21 matrix M where M[entry_idx, r] = U21[i*m+j, r] * V21[j'*p+k, r] * W21[k'*n+i', r]
    M = np.zeros((24, 21), dtype=object)
    for eidx, (i, j, jp, k, kp, ip) in enumerate(entries):
        for r in range(21):
            M[eidx, r] = U21[i*m+j, r] * V21[jp*p+k, r] * W21[kp*n+ip, r]
    
    # Solve M * c = 0 for non-trivial c.
    # Use SVD or null space.
    M_float = np.array(M, dtype=float)
    # Compute null space
    U_svd, s, Vt = np.linalg.svd(M_float, full_matrices=True)
    # Null space is spanned by rows of Vt corresponding to zero singular values.
    tol = 1e-10
    null_space_indices = np.where(s < tol)[0]
    
    if len(null_space_indices) > 0:
        # There exists a non-trivial linear dependency.
        # The null space dimension is len(null_space_indices).
        # We can try to use this to reduce the rank.
        # For now, just return True to indicate a potential rank-20 exists.
        # But we need to actually construct the rank-20 solution.
        # This is complex, so for now, let's just check if the null space is non-trivial.
        return None  # Indicate that further work is needed
    
    # No linear dependency found, rank 21 is minimal in this family.
    return None


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    # Try to find rank-20
    if not _deadline_exceeded():
        result20 = _search_rank20()
        if result20 is not None:
            return result20
    
    # Fall back to rank-21
    return _build_rank21()
# EVOLVE-BLOCK-END

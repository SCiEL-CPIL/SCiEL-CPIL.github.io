"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2):
    """
    Create a 2x2x2 Strassen-like block decomposition that handles a specific
    2x2x2 sub-block of the multiplication. Returns (U, V, W) as sparse-like
    dense arrays of size (n*m, 7), (m*p, 7), (p*n, 7).
    The block computes C[i,k] for i in {i1,i2}, k in {k1,k2} using the 
    7 Strassen terms for A[i,j], B[j,k] with j in {j1,j2}.
    """
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7-term formulas for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M6
    # C12 = M2 + M5
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # Note: indices are 0-based internally.
    # i1, i2 are the two row indices of A (and C)
    # j1, j2 are the two col indices of A (and row indices of B)
    # k1, k2 are the two col indices of B (and C)
    
    # M1
    U[i1*m + j1, 0] = 1.0
    U[i2*m + j2, 0] = 1.0
    V[j1*p + k1, 0] = 1.0
    V[j2*p + k2, 0] = 1.0
    W[k1*n + i1, 0] = 1.0  # C11
    W[k2*n + i2, 0] = 1.0  # C22
    
    # M2
    U[i2*m + j1, 1] = 1.0
    U[i2*m + j2, 1] = 1.0
    V[j1*p + k1, 1] = 1.0
    W[k2*n + i1, 1] = 1.0  # C12
    
    # M3
    U[i1*m + j1, 2] = 1.0
    V[j2*p + k1, 2] = 1.0
    V[j2*p + k2, 2] = -1.0
    W[k1*n + i2, 2] = 1.0  # C21
    
    # M4
    U[i2*m + j2, 3] = 1.0
    V[j1*p + k2, 3] = 1.0
    V[j1*p + k1, 3] = -1.0
    W[k1*n + i1, 3] = 1.0  # C11
    W[k1*n + i2, 3] = 1.0  # C21
    
    # M5
    U[i1*m + j1, 4] = 1.0
    U[i1*m + j2, 4] = 1.0
    V[j2*p + k2, 4] = 1.0
    W[k1*n + i1, 4] = -1.0  # C11
    W[k2*n + i1, 4] = 1.0   # C12
    
    # M6
    U[i2*m + j1, 5] = 1.0
    U[i1*m + j1, 5] = -1.0
    V[j1*p + k1, 5] = 1.0
    V[j1*p + k2, 5] = 1.0
    W[k1*n + i1, 5] = 1.0   # C11
    
    # M7
    U[i1*m + j2, 6] = 1.0
    U[i2*m + j2, 6] = -1.0
    V[j2*p + k1, 6] = 1.0
    V[j2*p + k2, 6] = 1.0
    W[k2*n + i2, 6] = 1.0   # C22
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    # The multiplication tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if we're looking at the term for (i,j,k))
    # Actually, the condition is: for all r, the product U[:,r] ⊗ V[:,r] ⊗ W[:,r] should sum to the identity tensor.
    # More precisely: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k), and 0 otherwise? 
    # No, the matrix multiplication tensor is defined as:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] has T[i,j,k] = 1 for all i,j,k (it's a 3D identity-like tensor where each slice (i,.,k) has a 1 at position j).
    # Actually, T is a 3D array of size (n, m, p) where T[i,j,k] = 1 for all i,j,k? No.
    # Let me think: A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k.
    # Wait, no. The tensor decomposition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    # Because A[i,j] and B[j,k] are free variables, the coefficient of A[i,j]B[j,k] in C[i',k'] should be delta_{i,i'}delta_{k,k'}.
    # So: sum_r U[i,j,r] V[j,k,r] W[k,i',r] = delta_{i,i'} delta_{k,k'} for all i,k,i',j.
    # In our W indexing: W is indexed by (k, i) where the output index is (i, k) but stored as k*n+i.
    # So W[k*n + i, r] corresponds to the (k,i) output.
    # Let me re-read the task: "W[k*n+i,r]" - so W is p*n x rank, indexed by (k, i) where k is row of B / col of C, i is col of A / row of C.
    # So C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right. The standard form is:
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] * A[i,j] * B[j,k] summed over j and r.
    # So: C[i,k] = sum_j A[i,j] B[j,k] * (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    # And for the off-diagonal terms to vanish:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 0 when (i',k') != (i,k)
    
    # Let's verify: compute the 3D array T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be all 1s.
    # And for off-diagonals: T[i,j,k] for the output (i',k') should be 0.
    # Actually, the full condition is that the matrix M_{(i,k), (i',k')} = sum_j,k T[i,j,k'] ... no.
    
    # Let's just check: for all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    # And for all i, j, k, i', k' with (i',k') != (i,k): 
    #   The coefficient of A[i,j]B[j,k] in C[i',k'] is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i', r]
    #   This should be 0.
    
    # Efficient check: form the tensor and verify.
    # Create a 4D array S[i', k, j, i] where S[i',k,j,i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i', r]
    # This should be delta_{i',i} * 1 (i.e., S[i',k,j,i] = 1 if i'==i, 0 otherwise)
    
    # Actually, let's be more direct. The condition is:
    # For all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # For all i,j,k,i',k' with i'!=i or k'!=k: sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 0
    
    # Let's check the diagonal first:
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    
    # Diagonal: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is an einsum: 'ijr,jkr,kir->ijk'
    diag = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat, optimize=True)
    if not np.allclose(diag, 1.0, atol=tol):
        return False
    
    # Off-diagonal: for each (i',k') != (i,k), sum_r U[i,j,r] V[j,k,r] W[k',i',r] = 0
    # This is equivalent to checking that the 4D tensor S[i',k,j,i] = einsum('ijr,jkr,k'r->... ') is zero off-diagonal.
    # But that's expensive. Instead, note that the full condition is that the matrix of coefficients is identity.
    # Let's form: for each r, the matrix A_r of size (n*p) x (n*p) where A_r[(i',k'), (i,j)] = U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
    # This is not quite right.
    
    # Simpler: check that for all i,j,k and all (i',k') != (i,k):
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 0
    # We can loop over a few random (i',k') to check, or do it fully.
    # For n=4, m=5, p=6, the full check is 4*5*6*4*6 = 2880 terms, each of rank r.
    # If r is ~100, that's 288000 operations, which is fast.
    
    # Let's do it fully:
    # S[i', k', j, i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r]
    # We want S[i', k', j, i] = 0 for all (i',k') != (i,k)
    # And S[i, k, j, i] = 1 for all i,j,k (already checked)
    
    # Actually, let's just check: for all i', k', j, i: 
    # if i' == i and k' == k: should be 1
    # else: should be 0
    
    # This is a 4D array of size (n, p, m, n) = 4*6*5*4 = 480
    S = np.einsum('ijr,jkr,kir->kji', U_mat, V_mat, W_mat, optimize=True)
    # S[k, j, i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, I want S[i', k', j, i] but the einsum above gives S[k, j, i] where k is from V and W.
    # Let me redo: I want for each (i', k', j, i):
    # sum_r U[i*m+j, r] * V[j*p+k', r] * W[k'*n+i', r]
    # = einsum('ijr,jk'r,k'r i'r -> ...') but the indices are mixed.
    
    # Let's use: S[i', k', j, i] = sum_r U[i*m+j, r] V[j*p+k', r] W[k'*n+i', r]
    # = einsum('ijr,jkr,kir->...') but i' and k' are the output indices, i and k are the input.
    # Actually, i' is the row of C, k' is the col of C. i is row of A, j is col of A, k is col of B.
    # So: S[i', k', i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r]
    # This should be 1 if (i',k') == (i,k) and j is the same j (it always is), else 0.
    # Wait, j is fixed in the sum. So for each (i,j,k) and (i',k'), we have a coefficient.
    
    # Let's just verify: C[i',k'] = sum_j A[i',j] B[j,k'] * (sum_r U[i'*m+j, r] V[j*p+k', r] W[k'*n+i', r])
    # So the coefficient of A[i',j]B[j,k'] in C[i',k'] is:
    # coeff[i', k', i', j, k'] = sum_r U[i'*m+j, r] V[j*p+k', r] W[k'*n+i', r]
    # This should be 1 for all i',j,k'.
    # And for A[i,j]B[j,k] to appear in C[i',k'] with i'!=i or k'!=k, we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 0
    
    # So let's check:
    # 1. For all i', j, k': sum_r U[i'*m+j, r] V[j*p+k', r] W[k'*n+i', r] == 1
    # 2. For all i, j, k, i', k' with (i',k') != (i,k): sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] == 0
    
    # Check 1:
    # diag2 = einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat) but with i',j,k'
    # Actually, check 1 is: for all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    # This is exactly what I computed as 'diag' above.
    
    # Check 2: off-diagonal
    # For each (i,j,k) and (i',k') != (i,k):
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] == 0
    
    # Let's do a spot check with random samples:
    rng = np.random.default_rng(42)
    for _ in range(100):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        ip = rng.integers(0, n)
        kp = rng.integers(0, p)
        if ip == i and kp == k:
            continue
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+ip, :])
        if abs(val) > tol:
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U_best, V_best, W_best = _make_schoolbook(n, m, p)
    rank_best = n * m * p
    
    # Try to find a better decomposition using Strassen-like blocks
    # Strategy: for each 2x2x2 block, try to replace the 8 schoolbook terms with 7 Strassen terms
    # This saves 1 term per block.
    
    # But we need to be careful: the Strassen block only works if the 2x2x2 sub-block is "isolated"
    # i.e., no other terms contribute to the same (i,k) outputs from different (j) indices.
    
    # For n=4, m=5, p=6, we can partition into blocks:
    # i: 0-1, 2-3
    # j: 0-1, 2-3, 4 (single)
    # k: 0-1, 2-3, 4-5
    
    # Let's try a different approach: use a known rank-120 decomposition if it exists.
    # Actually, let's just try to apply Strassen to disjoint 2x2x2 blocks.
    
    # Partition:
    # i blocks: (0,1), (2,3)
    # j blocks: (0,1), (2,3), (4,4) -- but j=4 is single, can't do 2x2
    # k blocks: (0,1), (2,3), (4,5)
    
    # For each combination of i-block, j-block, k-block where all have size 2:
    # i: (0,1), (2,3)
    # j: (0,1), (2,3)
    # k: (0,1), (2,3), (4,5)
    
    # So 2 * 2 * 3 = 12 blocks of size 2x2x2.
    # Each block: 8 schoolbook terms -> 7 Strassen terms, saving 1.
    # Total savings: 12.
    # New rank: 120 - 12 = 108.
    
    # But we need to make sure the blocks are disjoint in the (i,k) output space.
    # Each block (i1,i2, j1,j2, k1,k2) affects outputs C[i,k] for i in {i1,i2}, k in {k1,k2}.
    # Since the i-blocks and k-blocks are disjoint, the output regions are disjoint.
    # The j-index is only internal to the block, so no conflict.
    
    # Let's build the decomposition:
    U_new = np.zeros((n * m, rank_best), dtype=float)
    V_new = np.zeros((m * p, rank_best), dtype=float)
    W_new = np.zeros((p * n, rank_best), dtype=float)
    
    # Copy schoolbook
    U_new[:, :] = U_best
    V_new[:, :] = V_best
    W_new[:, :] = W_best
    
    # Now replace each 2x2x2 block with Strassen
    # But we need to remove the 8 schoolbook terms and add 7 Strassen terms.
    # This changes the rank, so we need to build from scratch.
    
    # Let's build a new decomposition:
    # For each (i,j,k) triple:
    #   If (i,j,k) is in a 2x2x2 block, use Strassen
    #   Else, use schoolbook
    
    # Actually, it's easier to build the Strassen decomposition directly.
    
    # Define blocks:
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # For each combination, create a Strassen block
    # And for the remaining (i,j,k) triples, use schoolbook
    
    # Let's build the full decomposition:
    rank_new = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                # Create Strassen block
                U_block, V_block, W_block = _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2)
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)
                rank_new += 7
                # Mark covered triples
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_col = np.zeros(n * m)
                    V_col = np.zeros(m * p)
                    W_col = np.zeros(p * n)
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    U_list.append(U_col.reshape(-1, 1))
                    V_list.append(V_col.reshape(-1, 1))
                    W_list.append(W_col.reshape(-1, 1))
                    rank_new += 1
    
    if rank_new < rank_best:
        U_new = np.hstack(U_list)
        V_new = np.hstack(V_list)
        W_new = np.hstack(W_list)
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p, tol=1e-9):
            U_best = U_new
            V_best = V_new
            W_best = W_new
            rank_best = rank_new
    
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END

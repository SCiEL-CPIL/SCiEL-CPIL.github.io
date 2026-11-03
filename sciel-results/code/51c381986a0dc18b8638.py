import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 150.0
SAFETY = 12.0

def _deadline_ok():
    return time.time() < (DEADLINE - SAFETY)

def _column_tensor(n, m, p, j):
    """Return the (n*p, m) matrix T where T[k*n+i, l] = 1 if l==j else 0.
    (This is the column-j slice of the multiplication tensor viewed as a matricization.)"""
    T = np.zeros((n * p, m), dtype=np.int64)
    for i in range(n):
        for k in range(p):
            T[k * n + i, j] = 1
    return T

def _rank_of(mat):
    return int(np.linalg.matrix_rank(mat))

def _search_column(n, m, p, j, max_rank, budget_s, rng):
    """Try to find an exact rank decomposition of the column-j slice.
    Returns (U, V) with U shape (n*p, r), V shape (m, r), r <= max_rank, or None."""
    T = _column_tensor(n, m, p, j)
    Tmat = T.astype(np.float64)
    start = time.time()
    r0 = _rank_of(Tmat)  # should be 1
    # We need U (n*p x r) and V (m x r) with U @ V.T == Tmat.
    # Since Tmat has rank 1, we can try r from 1 upward.
    for r in range(1, max_rank + 1):
        if not _deadline_ok():
            return None
        # Try several random restarts at this rank
        restarts = 200
        for it in range(restarts):
            if not _deadline_ok():
                return None
            if time.time() - start > budget_s:
                return None
            # Random U with small integer/half-integer entries
            scale = rng.choice([1, 2, 3])
            U = (rng.integers(0, 2 * scale + 1, size=(n * p, r)) - scale)
            U = U / 2.0  # half-integer
            # Solve V from U V^T = T  =>  V^T = (U^T U)^{-1} U^T T  if U has full col rank
            try:
                Umat = U.astype(np.float64)
                if _rank_of(Umat) < r:
                    continue
                # least squares for V
                Vt, *_ = np.linalg.lstsq(Umat.T, Tmat, rcond=None)
                V = Vt.T  # shape (m, r)
                # Check exactness with tolerance
                prod = Umat @ V.astype(np.float64).T
                if np.allclose(prod, Tmat, atol=1e-9, rtol=0):
                    # Convert to half-integers: multiply by 2 and round
                    U2 = np.round(U * 2)
                    V2 = np.round(V * 2)
                    # Verify exact integer check
                    prod2 = (U2.astype(np.float64) @ V2.astype(np.float64).T) / 4.0
                    if np.allclose(prod2, Tmat, atol=1e-9, rtol=0):
                        return (U2.astype(np.int64), V2.astype(np.int64))
            except np.linalg.LinAlgError:
                continue
    return None

def _build_full_from_column(col_decomp, n, m, p):
    """Given (U, V) for column j (U: (n*p, r), V: (m, r)), build full U,V,W of rank m*r.
    W is the standard: W[k*n+i, c] = 1 where c encodes (i,k).
    Actually we need consistency. Let's build directly."""
    Ucol, Vcol = col_decomp
    r = Ucol.shape[1]
    full_rank = m * r
    U = np.zeros((n * m, full_rank), dtype=np.int64)
    V = np.zeros((m * p, full_rank), dtype=np.int64)
    W = np.zeros((p * n, full_rank), dtype=np.int64)
    # For each column j of A, we use a block of r factors.
    # The decomposition says: for fixed j, sum_{c=0}^{r-1} Ucol[k*n+i, c] * Vcol[j, c] = 1 if j==j else 0.
    # Wait, that's not right. Let me reconsider.
    #
    # The multiplication tensor: T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k]).
    # The tensor is T[i,j,k] = delta (it's 1 for every (i,j,k) because the multiplication
    # tensor has T_{i,j,k} = 1 for all indices). Actually no:
    #
    # The matrix multiplication tensor is defined as:
    #   T_{(i,k), (j)} ... let me use the standard form.
    #   We want C = A B, so C[i,k] = sum_j A[i,j] B[j,k].
    #   The tensor T has T[i, j, k] = 1 for all i, j, k? No.
    #   Actually the multiplication tensor is the sum over j of e_i ⊗ e_j ⊗ e_k,
    #   i.e., T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k? That's not right either.
    #
    # Let me re-derive. C[i,k] = sum_j A[i,j] B[j,k].
    # As a tensor: T = sum_{i,j,k} T_{i,j,k} (e_i ⊗ e_j ⊗ e_k) where T_{i,j,k} = 1 for all i,j,k?
    # No! The tensor is T_{i,j,k} = 1 for all i,j,k only if we're summing over j.
    # Actually: the multiplication tensor is the sum over j of (e_i ⊗ e_j ⊗ e_k) for all i,k,
    # but with the constraint that the middle index is summed. So:
    #   T = sum_{i=0}^{n-1} sum_{k=0}^{p-1} sum_{j=0}^{m-1} e_i ⊗ e_j ⊗ e_k
    # Wait, that gives T_{i,j,k} = 1 for all i,j,k, which is the all-ones tensor.
    # But that's not the multiplication tensor. The multiplication tensor has
    # T_{i,j,k} = 1 for all i,j,k? Let me check with n=m=p=1: T_{0,0,0}=1, rank 1. Correct.
    # For n=2,m=2,p=2: T_{i,j,k} = 1 for all i,j,k? That's the all-ones 2x2x2 tensor,
    # which has rank 1 (it's e_0⊗e_0⊗e_0 + ... no, all-ones is rank 1? No, all-ones 2x2x2
    # has tensor rank 1? Let's see: ones = (1,1)⊗(1,1)⊗(1,1). Yes! Rank 1.
    # But 2x2x2 matrix multiplication has rank 7, not 1. So I'm confusing something.
    #
    # The correct multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1 for all (i,j,k)?
    # No. The tensor is: for each triple (i,j,k), the coefficient is 1. So T_{i,j,k}=1 for all i,j,k.
    # But that's the all-ones tensor, which has rank 1. Contradiction with known rank 7 for 2x2x2.
    #
    # I think I'm wrong. Let me look it up mentally. The matrix multiplication tensor for
    # m_{n×m, m×p} is: T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k? No.
    #
    # Actually the correct definition: the multiplication tensor is
    #   T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k
    # where the sum is over ALL i,j,k. This gives T_{i,j,k} = 1 for all i,j,k.
    # But this tensor has rank 1 (it's the outer product of all-ones vectors).
    #
    # I'm clearly confusing the multiplication tensor with something else. Let me think again.
    #
    # The bilinear map (A, B) -> C = A B. The tensor for this bilinear map is:
    #   T = sum_{i,j,k} (E_{i,j} ⊗ E_{j,k}) ⊗ E_{i,k}
    # where E_{i,j} is the matrix with 1 at (i,j) and 0 elsewhere.
    # So T_{(i,j),(j,k),(i,k)} = 1. In other words, T has nonzero entries only when
    # the "middle" indices match in a specific way.
    #
    # Let me use the standard tensor notation. The multiplication tensor T ∈ R^{(n×m) × (m×p) × (n×p)}
    # where T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    # So T[(i,j),(j,k),(i,k)] = 1 for all i,j,k, and 0 otherwise.
    #
    # So the tensor is: for each (i,j,k), there's a 1 at position ((i,j),(j,k),(i,k)).
    # This is NOT the all-ones tensor. It's a structured tensor.
    #
    # Now, the schoolbook decomposition: for each (i,j,k), one term:
    #   U[(i,j), c] = 1, V[(j,k), c] = 1, W[(i,k), c] = 1, where c = index of (i,j,k).
    # This gives rank n*m*p.
    #
    # Now, can we do better? The key observation: for a fixed j, the terms are
    #   U[(i,j), c], V[(j,k), c], W[(i,k), c] for all i,k.
    # This is a sub-tensor of size n×p×(n×p) = (n*p) × (n*p) with rank at most n*p.
    # Actually for fixed j, we have n*p terms (one for each (i,k)), each contributing
    # U[(i,j), c] * V[(j,k), c] * W[(i,k), c] = 1*1*1 = 1 at position ((i,j),(j,k),(i,k)).
    # So the sub-tensor for fixed j is the all-ones tensor of size (n*p) × (n*p)?
    # No. For fixed j, the nonzero entries are at ((i,j),(j,k),(i,k)) for all i,k.
    # This is a tensor in R^{(n) × (p) × (n×p)} where the first two indices are i,k (from U and V)
    # and the third is (i,k) (from W). So it's the "identity-like" tensor:
    #   T_j[i, k, (i,k)] = 1 for all i,k.
    # This is the tensor that maps (i,k) to (i,k), i.e., it's the identity tensor of size n*p.
    # Its rank is n*p (it's the sum of n*p rank-1 terms, and they're linearly independent).
    #
    # So for fixed j, the sub-tensor has rank exactly n*p = 15.
    # And the full tensor is the sum over j=0..4 of these sub-tensors, but they live in
    # different "slices" of the U and V indices (different j values). So the full tensor
    # is a direct sum of 5 sub-tensors, each of rank 15. Total rank = 5*15 = 75.
    #
    # Wait, but that's the same as the schoolbook rank! Because n*m*p = 3*5*5 = 75.
    #
    # Hmm, so the block structure doesn't help here because each block already has
    # rank n*p = 15, and there are m=5 blocks, giving 5*15 = 75 = n*m*p.
    #
    # So the schoolbook decomposition is already optimal for this block structure?
    # Not necessarily. There might be cross-block cancellations or shared factors.
    #
    # Let me reconsider. The full tensor T has T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'.
    # A rank-1 term is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{n*p}.
    # The term contributes u[(i,j)] * v[(j',k)] * w[(i',k')] to T[(i,j),(j',k),(i',k')].
    #
    # For the sum to equal T, we need:
    #   sum_c u_c[(i,j)] * v_c[(j',k)] * w_c[(i',k')] = 1 if i=i', j=j', k=k', else 0.
    #
    # This is a system of equations. The question is whether we can find a small set of
    # rank-1 terms that satisfy this.
    #
    # One approach: use the fact that the tensor is "separable" in some sense.
    #
    # Actually, let me try a different approach. I'll set up the problem as a system of
    # polynomial equations and try to solve it for small rank using numerical methods.
    #
    # But given the time constraints, let me try a more practical approach:
    # use a randomized search with integer/half-integer entries, trying to find a
    # decomposition of rank less than 75.
    #
    # Given the complexity, let me just return the schoolbook decomposition for now,
    # but with a small search to see if we can do better.
    #
    # For now, let's just return the schoolbook decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    rng = np.random.default_rng(42)
    
    # Try to find a better decomposition via search
    best_rank = n * m * p
    best_decomp = None
    
    # Schoolbook as baseline
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_decomp = (U, V, W)
    best_rank = rank
    
    # Try search for better rank (limited time)
    budget = min(100, DEADLINE - SAFETY - (time.time() - start))
    if budget > 10:
        # Try to find a decomposition of rank < best_rank
        # This is very hard, so we'll just do a quick randomized search
        for target_rank in range(best_rank - 1, max(best_rank - 20, 0), -1):
            if not _deadline_ok():
                break
            if time.time() - start > budget:
                break
            # Random search
            found = False
            for it in range(50):
                if not _deadline_ok() or time.time() - start > budget:
                    break
                U_r = (rng.integers(0, 4, size=(n * m, target_rank)) - 1)
                V_r = (rng.integers(0, 4, size=(m * p, target_rank)) - 1)
                W_r = (rng.integers(0, 4, size=(p * n, target_rank)) - 1)
                # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
                # This is a tensor reconstruction check
                # Build the tensor from the decomposition
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We need T[i,j,k] = 1 for all i,j,k
                # Let's check a few random entries
                ok = True
                for _ in range(100):
                    i = rng.integers(0, n)
                    j = rng.integers(0, m)
                    k = rng.integers(0, p)
                    s = 0
                    for c in range(target_rank):
                        s += U_r[i * m + j, c] * V_r[j * p + k, c] * W_r[k * n + i, c]
                    if s != 1:
                        ok = False
                        break
                if ok:
                    # Verify all entries
                    all_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(target_rank):
                                    s += U_r[i * m + j, c] * V_r[j * p + k, c] * W_r[k * n + i, c]
                                if s != 1:
                                    all_ok = False
                                    break
                            if not all_ok:
                                break
                        if not all_ok:
                            break
                    if all_ok:
                        best_rank = target_rank
                        best_decomp = (U_r, V_r, W_r)
                        found = True
                        break
            if found:
                break
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

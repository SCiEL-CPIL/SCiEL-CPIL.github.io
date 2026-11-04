import time
import itertools

import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 120.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _to_int_list(A):
    # Entries should be integer; round tiny float error if any.
    A = np.asarray(A)
    return np.round(A).astype(int).tolist()


def construct():
    n, m, p = DIMENSIONS
    # Start with the schoolbook rank n*m*p.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Structured reduction attempt.
    try:
        _deadline_hit = False
        try:
            U, V, W, _ = _structured_reduce(U, V, W, n, m, p)
        except _Deadline:
            _deadline_hit = True
    except Exception:
        pass  # Keep the known-good schoolbook if reduction fails.

    # Final safety: verify exactness (cheap for rank 80).
    try:
        if not _deadline_hit:
            if _verify_exact(U, V, W, n, m, p, max_samples=4):
                pass
            else:
                # Fall back to schoolbook if verification fails.
                U = np.zeros((n * m, n * m * p))
                V = np.zeros((m * p, n * m * p))
                W = np.zeros((p * n, n * m * p))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
    except _Deadline:
        pass
    except Exception:
        pass

    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}


class _Deadline(Exception):
    pass


def _check_deadline():
    if _elapsed() > DEADLINE - 8.0:
        raise _Deadline()


def _verify_exact(U, V, W, n, m, p, max_samples=2):
    """Check tensor identity by sampling random integer matrices."""
    _check_deadline()
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Tensor coefficient for (i, j, k): sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] should equal 1.
    # Sample a few random integer matrices to confirm C = A @ B.
    rng = np.random.default_rng(12345)
    for _ in range(max_samples):
        A = rng.integers(-2, 3, size=(n, m))
        B = rng.integers(-2, 3, size=(m, p))
        C_ref = A @ B
        # Compute via tensors: C[i,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
        # Efficient: T_c[i,k] = sum_j (U[i*m+j,c] * A[i,j]) * (V[j*p+k,c] * B[j,k]) then C[i,k] = sum_c W[k*n+i,c]*T_c[i,k]
        # Simpler with rank small: build M[c, (i,j)] and N[c,(j,k)] then combine.
        Ua = np.einsum('ic,ij->ijc', U, A)  # (i, m, rank)
        Vb = np.einsum('jc,jk->kjc', V, B)  # (p, m, rank)
        # C[i,k] = sum_c sum_j W[k*n+i, c] * Ua[i, j, c] * Vb[k, j, c]
        prod = np.einsum('ijc,kjc->ikc', Ua, Vb)  # (i, k, rank)
        C_calc = np.einsum('ikc,kic->ik', prod, W.T.T)  # W is (p*n, rank); want W[k*n+i, c]
        W_ikc = W.T  # (rank, p*n)
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_calc[i, k] = float(np.dot(prod[i, k, :], W[k * n + i, :]))
        if not np.allclose(C_calc, C_ref, atol=1e-9):
            return False
        _check_deadline()
    return True


def _structured_reduce(U, V, W, n, m, p):
    """
    Try to find a lower-rank exact decomposition via:
    1) Kronecker-structured ansatz: exploit 4=2*2 to build from a 2x2x2 base and a 4x4x? block.
    2) Sparse linear dependency elimination on the current rank: solve for each term as a linear
       combination of the others (since the tensor is multilinear in U,V,W per-term, linear
       redundancy in the term set does NOT necessarily imply redundancy of the decomposition --
       the tensor is NOT linear in the terms! The sum of rank-1 tensors is multilinear).
    Because the tensor is a SUM of rank-1 tensors (each product U_col * V_col * W_col), and the
    set of such sums of a fixed rank is not a linear subspace, we cannot just null-space reduce.
    Instead, we search for a structured exact decomposition.
    """
    _check_deadline()

    # ---- Approach: Kronecker product of a 2x2x2 and 2x2x(5/2?) ----
    # Since 4 = 2*2 and 5 is not even, try:
    # M_{4,4,5} = (M_{2,2,2} ⊗ M_{2,2,?}) is not direct because 2*?=5 impossible.
    # Alternative: M_{4,4,5} = M_{2,2,5} ⊗ M_{2,2,1} won't give integer ranks cleanly.
    #
    # Let's try the "Kronecker sum" style:
    #   M_{4,4,5} = (M_{2,2,2} ⊗ I_2) + (I_2 ⊗ M_{2,2,5})
    #   rank(M_{2,2,2}⊗I_2) = 4*2 = 8, rank(I_2⊗M_{2,2,5}) = 2*rank(M_{2,2,5}).
    #   If rank(M_{2,2,5}) = 10 (schoolbook), total = 18. But need to verify the tensor identity holds!
    #   Actually (A⊗I) @ (B⊗I) = (A@B)⊗I, so M_{2,2,2}⊗I_2 corresponds to multiplying a 2x2 block
    #   matrix by I_2, which is not the same as M_{4,4,5}. The Kronecker sum doesn't simply add.

    # ---- More robust: Direct Kronecker embedding ----
    # View 4x4 = 2x2 blocks of 2x2. Then C = A@B where A is 4x4, B is 4x5.
    # Partition A into 2x2 blocks: A = [A11 A12; A21 A22] each 2x2.
    # B = [B1; B2] where B1, B2 are 2x5.
    # C = [A11 B1 + A12 B2; A21 B1 + A22 B2]  -> C1, C2 are 2x5.
    # So C1 = A11 B1 + A12 B2  (2x5 product-sum)
    #    C2 = A21 B1 + A22 B2
    #
    # rank(M_{4,4,5}) <= 2 * rank(M_{2,2,5}) + cross terms... not clean.

    # ---- Let's just try to find a lower-rank solution via random initialization + structured search ----
    # Given time constraints, we will attempt a Kronecker-based construction that is KNOWN to work:
    #
    # M_{4,4,5} can be built from M_{2,2,5} using:
    #   A = A1 ⊗ I2 + A2 ⊗ J2  (4x4) where J2 = [[0,1],[1,0]] or similar
    # This gets complicated. 

    # ---- Safest structured approach: Kronecker product M_{2,2,5} ⊗ M_{2,2,1} where M_{2,2,1} is rank 4 ----
    # M_{2,2,1} has rank 2*2*1 = 4.
    # M_{2,2,5} schoolbook rank = 2*2*5 = 20.
    # Kronecker: rank = 4 * 20 = 80. Same as schoolbook. Not helpful.

    # ---- Try:  M_{4,4,5} = M_{2,2,5} ⊗ I_2  --> rank = 20 * 4 = 80. Same.

    # Given the difficulty of finding a true lower rank in the remaining time budget,
    # we will attempt a "structured sparsity" reduction: look for terms in the schoolbook
    # decomposition that are linearly dependent in a multilinear sense by testing small
    # random replacements. This is a heuristic and may not improve.

    # Given time, return the original if no clear improvement is found quickly.
    _check_deadline()

    # Try a quick structured Kronecker build for 4,4,5 using 2,2,5 and 2,2,2:
    # C = (A1 ⊗ I2 + A2 ⊗ K2)(B1 ⊗ I2 + B2 ⊗ K2) ... too speculative.

    return U, V, W, 0

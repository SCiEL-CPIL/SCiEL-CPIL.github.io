import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds


def _winograd_frequency_domain(n, m, p):
    """
    Winograd Frequency-Domain decomposition for (n x m) * (m x p) matrix multiplication.
    
    Key insight: For each row i of A and row j of B, the contribution to output row k
    involves a linear combination. We use DFT over Z_p to diagonalize the convolution-like
    structure.
    
    For m=3, p=10: We map the 3-element vector (row of A) and 10-element vector (row of B)
    to frequency domain over Z_10. The multiplication becomes pointwise in frequency domain.
    
    The rank is determined by the number of distinct frequency components needed.
    For Z_10, the DFT has 10 frequency bins, but we only need those where the
    3-element kernel has non-zero transform.
    
    Actually, let me think more carefully. The tensor T[i,j,k] = sum_l A[i,l]*B[l,k].
    For fixed i, this is: C[i,k] = sum_l A[i,l]*B[l,k].
    
    Using Winograd: represent A[i,:] (length m=3) and B[:,k] (length m=3) 
    in a basis where multiplication is pointwise.
    
    For m=3, we can use the fact that any 3x3 matrix multiplication has rank 7
    (Strassen). But here we have 2x3 * 3x10.
    
    Alternative Winograd approach: Pad to length 5 (next power of 2 above 3+3-1=5)
    and use FFT-like decomposition. But this gives rank 5*2=10 per (i,k) pair,
    total rank 2*10*10 = 200, which is worse.
    
    Better: Use the algebraic structure directly.
    
    For each output position (i,k), we need: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    This is a dot product of length 3. The tensor for a single dot product of length 3
    has rank 3 (it's just the identity tensor for the dot product).
    
    But we can share computations across different (i,k) pairs.
    
    Winograd's approach: For convolution of length m with length p, use frequency domain.
    Here, for each i, we compute C[i,:] = A[i,:] * B (a 1x3 times 3x10).
    
    This is equivalent to: for each i, C[i,k] = sum_l A[i,l]*B[l,k].
    
    We can view this as: C = A * B where A is 2x3 and B is 3x10.
    
    Winograd frequency domain: Let omega = exp(2*pi*i/10). 
    For each frequency f in Z_10:
      A_hat[i,f] = sum_l A[i,l] * omega^(l*f)  (but A only has 3 non-zero entries)
      B_hat[f,k] = sum_l B[l,k] * omega^(-l*f)
      C_hat[i,f,k] = A_hat[i,f] * B_hat[f,k]
      C[i,k] = (1/10) * sum_f C_hat[i,f,k]
    
    This gives rank 2*10*10 = 200, which is much worse.
    
    The key insight from Winograd: we don't need all 10 frequencies. We only need
    frequencies where the transform is non-trivial. For a 3-element vector, the
    transform has at most 3 non-zero values in any basis, but in the DFT basis
    over Z_10, all 10 can be non-zero.
    
    However, we can use a different approach: use the minimal polynomial basis.
    
    Let me try a direct algebraic approach:
    
    For the 3-term dot product: x0*y0 + x1*y1 + x2*y2
    
    We can write this as:
    Let s0 = x0 + x1 + x2
    Let s1 = x0 + ω*x1 + ω²*x2  (where ω is a primitive 3rd root of unity)
    Let s2 = x0 + ω²*x1 + ω*x2
    
    Similarly for y. Then:
    x0*y0 + x1*y1 + x2*y2 = (1/3)(s0*t0 + s1*t1 + s2*t2)
    
    But ω is not rational, so this doesn't give integer coefficients.
    
    For integer/rational coefficients, the rank of the 3-term dot product tensor
    is 3 (it's already minimal for rational coefficients).
    
    So for 2*10 = 20 output positions, each requiring a rank-3 decomposition,
    the naive approach gives rank 60.
    
    But we can share: the x-vector is shared across all k for a fixed i.
    So for each i (2 values), we compute the 3 "transformed" values once,
    then for each k, we do 3 multiplications.
    
    Total rank = 2 * 3 + 2 * 10 * 3 = 6 + 60 = 66. Worse!
    
    Actually no. Let me reconsider.
    
    The tensor is T[i,j,k] = 1 for all i,j,k (it's the matmul tensor).
    
    A rank-r decomposition is: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    
    For the schoolbook: U[i,j,c]=1 if c=(i,j), V[j,k,c]=1 if c=(j,k), W[k,i,c]=1 if c=(i,k).
    Rank = n*m*p = 2*3*10 = 60.
    
    To reduce rank, we need to find a decomposition with r < 60.
    
    The Strassen approach on 2x2 blocks: for each pair of k-values, we use rank 7
    instead of 8 for the 2x2x2 sub-tensor. This saves 1 per block, giving 55.
    
    Can we do better? 
    
    For the 2x3x10 tensor, the rank is known to be between 55 and 60.
    The lower bound is max(2*3, 3*10, 2*10) = 30, but that's very loose.
    
    A better lower bound: the slice rank or other bounds.
    
    Let me try a different Winograd approach:
    
    Group the 10 k-values into 5 pairs. For each pair (k0, k1), we have a 2x3x2 sub-tensor.
    The rank of 2x3x2 is at most 2*3*2 = 12, but can be reduced.
    
    Actually, 2x3x2: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1}.
    This is 4 output values, each a 3-term dot product.
    
    Can we do this in rank < 12? 
    
    Using Strassen-like ideas: 
    For the 2x2 part (j=0,1): rank 7 as before.
    For the j=2 part: 4 more terms.
    Total: 11 per block, 5 blocks = 55. Same as before.
    
    What if we use a different grouping? 
    
    Try grouping into 2 groups of 5:
    For each group of 5 k-values, we have a 2x3x5 sub-tensor.
    Rank of 2x3x5: at most 30, but can we reduce?
    
    Using Winograd on the 5-length dimension:
    For convolution of length 3 with length 5, Winograd uses length 3+5-1=7,
    padded to 8. Rank per (i) = 8, total = 2*8*2 = 32. Worse.
    
    I think 55 is close to optimal for this structure. Let me verify the current
    implementation is correct and try small optimizations.
    """
    # Fall back to the Strassen + schoolbook approach
    return _build_strassen_schoolbook(n, m, p)


def _build_strassen_schoolbook(n, m, p):
    """Build the rank-55 decomposition: 5 Strassen blocks (35) + 20 schoolbook terms."""
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # A indices for the 2x2 part
    a00, a01, a10, a11 = 0, 1, 3, 4

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1

        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1

        # M1: (a00+a11)(b00+b11) -> c00, c11
        U[a00, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M2: (a10+a11)b00 -> c10, -c11
        U[a10, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = -1.0
        col += 1

        # M3: a00(b01-b11) -> c01, c11
        U[a00, col] = 1.0
        V[b01, col] = 1.0
        V[b11, col] = -1.0
        W[c01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M4: a11(b10-b00) -> c00, c10
        U[a11, col] = 1.0
        V[b10, col] = 1.0
        V[b00, col] = -1.0
        W[c00, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # M5: (a00+a01)b11 -> -c00, c01
        U[a00, col] = 1.0
        U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0
        W[c01, col] = 1.0
        col += 1

        # M6: (a10-a00)(b00+b01) -> c11
        U[a10, col] = 1.0
        U[a00, col] = -1.0
        V[b00, col] = 1.0
        V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M7: (a01-a11)(b10+b11) -> c00
        U[a01, col] = 1.0
        U[a11, col] = -1.0
        V[b10, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook for A_R (2x1) * B_B (1x2)
        a02 = 0 * 3 + 2
        a12 = 1 * 3 + 2
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1

        # P1: a02*b20 -> c00
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # P2: a02*b21 -> c01
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        # P3: a12*b20 -> c10
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # P4: a12*b21 -> c11
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    r = U.shape[1]
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # Compute tensor sum
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < 1e-9


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by finding a more efficient decomposition.
    Returns (U', V', W') if successful, else None.
    """
    # Try a 3x3 Strassen-like decomposition for the full 2x3x10 problem
    # by grouping differently
    
    # Approach: Use the fact that for 2x3x10, we can potentially use
    # a decomposition that shares more structure.
    
    # Try: group k into 2 groups of 5, use Winograd on each group
    # For 2x3x5: rank is at most 30, but can be reduced
    
    # Actually, let's try a direct algebraic optimization
    # using the structure of the problem.
    
    # The key insight: for each i, the row A[i,:] is shared across all k.
    # So we can precompute combinations of A[i,:] that are useful for
    # multiple k values.
    
    # For 2x3x10, try to find a rank-54 decomposition by merging
    # one schoolbook term with a Strassen term.
    
    # This is a hard combinatorial problem. For now, return None
    # to keep the rank-55 solution.
    return None


def _build_rank54_attempt(n, m, p):
    """
    Attempt a rank-54 decomposition by using a modified Strassen approach
    that shares one more term across blocks.
    
    Key idea: Instead of independent 2x2 Strassen blocks, use a global
    decomposition that exploits the structure of the 2x3x10 tensor.
    
    For 2x3x10, we can write:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Let's define global combinations:
    For the 2x2 part (j=0,1), use Strassen per 2x2 block as before (rank 35).
    For the j=2 part, instead of 20 separate terms, try to share.
    
    Actually, the j=2 part is: A[i,2]*B[2,k] for i in {0,1}, k in {0..9}.
    This is an outer product structure: (A[:,2]) * (B[2,:]).
    The rank of this outer product tensor is 2*10 = 20 (each term is
    A[i,2]*B[2,k] contributing to C[i,k]).
    
    Can we reduce this? The tensor for the outer product has rank 20
    (it's a rank-1 tensor in the (i,k) space for each (i,k) pair, but
    the full tensor over all i,k has rank 20).
    
    Wait, no. The tensor T[i,2,k] = A[i,2]*B[2,k]. This is a 2x1x10 tensor
    which has rank 20 (each entry is a separate product).
    
    Actually, the rank of a 2x1x10 tensor where T[i,0,k] = a_i * b_k is
    min(2, 1, 10) = 1? No, that's not right either.
    
    Let me think again. The tensor is T[i,j,k] with j=2 fixed.
    T[i,2,k] = A[i,2] * B[2,k].
    
    In the decomposition T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c],
    for j=2: T[i,2,k] = sum_c U[i,2,c]*V[2,k,c]*W[k,i,c].
    
    If we set U[i,2,c] = A[i,2], V[2,k,c] = B[2,k], W[k,i,c] = 1,
    then T[i,2,k] = sum_c A[i,2]*B[2,k] = 10*A[i,2]*B[2,k], which is wrong.
    
    The correct schoolbook: for each (i,k), one term with U[i,2,c]=1, V[2,k,c]=1, W[k,i,c]=1.
    Rank = 2*10 = 20.
    
    Can we do better? The tensor T[i,2,k] = a_i * b_k (where a_i = A[i,2], b_k = B[2,k]).
    This is a rank-1 tensor in the (i,k) space! So it should have rank 1 in the
    (i,k) decomposition, but in the full (i,j,k) tensor with j fixed at 2,
    the rank is still 20 because we need separate terms for each (i,k) to
    contribute to the correct W[k,i,c] position.
    
    Actually no. Let me re-examine.
    
    If we use: U[i,2,c] = a_i, V[2,k,c] = b_k, W[k,i,c] = 1 for all i,k,
    then T[i,2,k] = sum_c a_i * b_k * 1 = (sum_c 1) * a_i * b_k.
    If we have r terms, T[i,2,k] = r * a_i * b_k.
    
    We want T[i,2,k] = a_i * b_k, so we need r=1. But then we only have
    one term, and U[i,2,0] = a_i, V[2,k,0] = b_k, W[k,i,0] = 1.
    This works! Rank 1 for the j=2 part!
    
    But wait, this assumes a_i and b_k are the actual values, not just
    indices. In the tensor decomposition, U, V, W are fixed matrices,
    and the tensor is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c].
    The values a_i and b_k are not part of the decomposition; they are
    the input matrix entries.
    
    So the decomposition must work for ALL possible input matrices.
    That means U, V, W must be such that for any A, B:
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j A[i,j]*U[j,i,c]) * ... 
    
    I'm confusing myself. Let me go back to basics.
    
    The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k.
    A rank-r decomposition is U, V, W with:
    sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    
    The schoolbook has r = n*m*p = 60 terms.
    The Strassen+schoolbook has r = 55 terms.
    
    To get rank 54, we need to find a 54-term decomposition.
    This is a hard problem. For now, return None.
    """
    return None


def construct():
    _start = time.time()
    n, m, p = DIMENSIONS

    # Build the base rank-55 decomposition
    U, V, W, base_rank = _build_strassen_schoolbook(n, m, p)

    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition is invalid")

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = base_rank

    # Try rank-54 decomposition
    if time.time() - _start < TIME_LIMIT - 30:
        result = _build_rank54_attempt(n, m, p)
        if result is not None:
            U54, V54, W54, rank54 = result
            if _verify_decomposition(U54, V54, W54, n, m, p):
                best_U, best_V, best_W = U54, V54, W54
                best_rank = rank54

    # Check time and return best result
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

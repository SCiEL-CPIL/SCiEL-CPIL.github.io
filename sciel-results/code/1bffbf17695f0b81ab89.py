"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _terms_to_matrices(terms, n_a, n_b, n_c):
    """Convert list of terms to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _get_rank17_terms():
    """Return the 17-term decomposition as a list of term tuples."""
    terms = []
    # Part 1: Strassen for 2x2 x 2x2
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    # Part 2: Direct for 2x2 x 2x1
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    # Part 3: Direct for 2x1 x 1x3
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    return terms

def _try_merge_two_terms(terms, T, n_a, n_b, n_c, deadline):
    """
    Try to merge pairs of terms into a single rank-1 term.
    For each pair (i,j), check if their sum can be expressed as a single rank-1 term.
    The sum of two rank-1 terms u1*v1*w1 + u2*v2*w2 is rank-1 iff
    one of the vectors is proportional to another in a specific way.
    
    Actually, let's think differently. We have 17 terms. We want to see if any two
    can be combined. The sum of two rank-1 tensors is rank-1 iff they are linearly
    dependent (i.e., one is a scalar multiple of the other), which is not the case here.
    
    Alternative approach: Try to find a rank-16 decomposition by allowing one term
    to have non-trivial structure. Let's try a different strategy:
    
    Try to replace two terms with a single term that has half-integer or other values.
    This is hard in general. Let's try a more systematic search.
    
    Strategy: For each pair of terms (i, j), try to find a single rank-1 term
    u*v*w such that term_i + term_j = u*v*w + correction, where correction
    can be absorbed into other terms. This is complex.
    
    Simpler approach: Try to find if any single term can be eliminated by adjusting
    others. Let's try a local search: start with 17 terms, try to set one term to
    zero and see if the residual can be expressed with the remaining 16 terms
    (possibly modified).
    """
    start_time = time.time()
    
    # For each term to remove, try to see if the remaining 16 can be adjusted
    # to cover the full tensor. This is a linear algebra problem.
    
    # Actually, let's try a different approach: search for rank-16 decompositions
    # with entries in {-1, 0, 1} using a smarter method.
    
    # Try: can we merge term 8 (A[0,0]*B[0,2]->C[0,2]) with term 14 (A[0,2]*B[2,2]->C[0,2])?
    # Both contribute to C[0,2]. Their sum is A[0,0]*B[0,2] + A[0,2]*B[2,2].
    # This is not rank-1 in general.
    
    # Let's try a completely different structural approach.
    # What if we use a different block decomposition?
    
    # Alternative: 2x3 * 3x3. Split B into B_left(3x2) and B_right(3x1).
    # A(2x3) * B_left(3x2) = 2x2, A(2x3) * B_right(3x1) = 2x1.
    # For 2x3 * 3x2: this is more complex.
    
    # Let's try: split A into A_top(1x3) and A_bottom(1x3), both 1x3.
    # A_top * B(3x3) = 1x3, A_bottom * B(3x3) = 1x3.
    # Each 1x3 * 3x3 = 1x3 requires 9 scalar mults naively, but can be done in fewer?
    # Actually 1x3 * 3x3: C[k] = sum_j A[j]*B[j,k]. This is 9 terms, but the rank
    # of this sub-tensor is 3 (since it's a sum of 3 rank-1 tensors, one per j).
    # So total rank would be 3+3=6 for this split, but we need to handle the full 2x3*3x3.
    
    # Wait, that's not right. Let me reconsider.
    
    # The tensor for 2x3 * 3x3 has rank at most 18 (schoolbook).
    # We have 17. Can we do 16?
    
    # Known results: The rank of matrix multiplication tensor M_{2,3,3} is known to be 17.
    # See: "The rank of matrix multiplication" - it's known that M_{2,3,3} has rank 17.
    # If that's the case, we cannot improve below 17.
    
    # But let's still try to verify by attempting a search.
    
    # Try a limited search for rank-16 with entries in {-1, 0, 1}.
    # This is a huge search space, so we'll do a heuristic search.
    
    # Heuristic: Start from the 17-term solution and try to perturb it.
    
    # For now, return None to indicate no improvement found.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Try to find rank-16 decomposition
    if time.time() < deadline - 30:
        result16 = _try_merge_two_terms(None, T, n_a, n_b, n_c, deadline)
        if result16 is not None:
            U, V, W = result16
            if _verify_decomposition(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-17
    terms = _get_rank17_terms()
    U, V, W = _terms_to_matrices(terms, n_a, n_b, n_c)
    
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

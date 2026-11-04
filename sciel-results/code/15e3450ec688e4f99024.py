"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Try to find a rank-16 decomposition by exploiting block structure
    # Strategy: Use the known rank-17 decomposition as a starting point,
    # then try to reduce by one more rank through linear dependencies.
    
    # First, build the rank-17 decomposition (Strassen + direct)
    rank17_terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    # M1 = (a0+a4)*(b0+b4)
    rank17_terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    # M2 = (a3+a4)*b0
    rank17_terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    # M3 = a0*(b1-b4)
    rank17_terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    # M4 = a4*(b3-b0)
    rank17_terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    # M5 = (a0+a1)*b4
    rank17_terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    # M6 = (a3-a0)*(b0+b1)
    rank17_terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # M7 = (a1-a4)*(b3+b4)
    rank17_terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    rank17_terms.append(([0], [1], [2], [1], [4], [1]))
    rank17_terms.append(([1], [1], [5], [1], [4], [1]))
    rank17_terms.append(([3], [1], [2], [1], [5], [1]))
    rank17_terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    rank17_terms.append(([2], [1], [6], [1], [0], [1]))
    rank17_terms.append(([2], [1], [7], [1], [2], [1]))
    rank17_terms.append(([2], [1], [8], [1], [4], [1]))
    rank17_terms.append(([5], [1], [6], [1], [1], [1]))
    rank17_terms.append(([5], [1], [7], [1], [3], [1]))
    rank17_terms.append(([5], [1], [8], [1], [5], [1]))
    
    # Build rank-17 matrices
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(rank17_terms):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
    
    # Verify rank-17
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook rank 18
        U_fb = np.zeros((n_a, 18))
        V_fb = np.zeros((n_b, 18))
        W_fb = np.zeros((n_c, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now try to find a rank-16 decomposition
    # Idea: The 17 rank-1 terms span a 17-dimensional space. We want to see if
    # we can express the same tensor with only 16 rank-1 terms.
    # 
    # One approach: Check if one of the 17 terms can be expressed as a linear
    # combination of the others in a way that preserves the tensor structure.
    # This is generally hard, but we can try a specific structural approach.
    #
    # Alternative: Use the fact that the two rows of A are somewhat independent.
    # Row 0 of A: [a0, a1, a2] -> multiplies with B to give row 0 of C
    # Row 1 of A: [a3, a4, a5] -> multiplies with B to give row 1 of C
    #
    # Each row is a 1x3 * 3x3 multiplication, which has rank 9 (schoolbook) or
    # potentially lower. But the full 2x3*3x3 tensor has more structure.
    #
    # Let's try a different approach: use bilinear complexity results.
    # For 2x3 * 3x3, the known bounds are:
    # Lower bound: max(2*3, 3*3, 2*3) = 9 (trivial)
    # Upper bound: 18 (schoolbook)
    # The Strassen-based approach gives 17.
    #
    # Let's try to find if rank 16 is achievable by a more clever decomposition.
    # We'll attempt to combine some of the direct multiplication terms with
    # the Strassen terms.
    
    # Try rank 16 by checking if we can merge two of the simple terms
    # The terms involving a2 and a5 (the last column of A) are:
    # Term 12: a2 * b6 -> c0
    # Term 13: a2 * b7 -> c2
    # Term 14: a2 * b8 -> c4
    # Term 15: a5 * b6 -> c1
    # Term 16: a5 * b7 -> c3
    # Term 17: a5 * b8 -> c5
    #
    # These 6 terms could potentially be reduced if we can find a shared structure.
    # But each (a, b) pair maps to a unique c, so they're all independent.
    
    # Let's try another approach: modify the Strassen decomposition to incorporate
    # some of the direct terms.
    
    # Actually, let's just return the rank-17 decomposition since finding rank 16
    # is non-trivial and we don't have a known construction.
    
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, rank, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_laderman_rank46(n, m, p):
    """
    Rank-46 decomposition of 3x3 * 3x6 using Laderman's rank-23 for 3x3*3x3.
    B is split into B_left (cols 0..2) and B_right (cols 3..5).
    U shared across both halves; V has 23 cols for left + 23 for right.
    W has 23 cols for left (output cols 0..2) + 23 for right (output cols 3..5).
    """
    # Laderman's 23-term decomposition for 3x3 * 3x3.
    # Each term t: u_t (9-vector over A), v_t (9-vector over B), w_t (9-vector over C).
    # C[i,k] = sum_t u_t[i,j] A[i,j] * v_t[j,k] B[j,k] * w_t[i,k]
    # Actually the standard form: C[i,k] = sum_t (sum_{a,b} u_{i,a,b,t} A[a,b]) * (sum_{c,d} v_{c,d,i,k,t} B[c,d])
    # But the simpler bilinear form: C[i,k] += x_t * y_t * w_{i,k,t}
    # where x_t = linear combo of A, y_t = linear combo of B.
    #
    # I'll use a known explicit 23-term set. The coefficients are:
    # Terms 1-9: "diagonal" / schoolbook-like
    # Terms 10-23: correction terms
    #
    # Let me use the standard Laderman coefficients.
    # u_t: 9 values for A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2],A[2,0],A[2,1],A[2,2]
    # v_t: 9 values for B[0,0],B[0,1],B[0,2],B[1,0],B[1,1],B[1,2],B[2,0],B[2,1],B[2,2]
    # w_t: 9 values for C[0,0],C[0,1],C[0,2],C[1,0],C[1,1],C[1,2],C[2,0],C[2,1],C[2,2]

    # Laderman 23 coefficients (one standard representation from literature)
    # I'll define them as a list of (u, v, w) tuples.
    # This is a well-known decomposition; let me use a verified set.

    # Using the formulation where:
    # C[i,k] = sum_t w_{i,k,t} * (sum_{a,b} u_{a,b,t} * A[a,b]) * (sum_{c,d} v_{c,d,t} * B[c,d])

    laderman_terms = [
        # (u, v, w) each of length 9
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2: A[0,2]*B[2,0] -> C[0,0]
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 3: A[1,0]*B[0,1] -> C[1,1]
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 5: A[1,2]*B[2,1] -> C[1,1]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 6: A[2,0]*B[0,2] -> C[2,2]
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 7: A[2,1]*B[1,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 8: A[2,2]*B[2,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # Terms 9-22: correction terms (Laderman's additional 14 terms)
        # These handle the off-diagonal cross terms.
        # I'll use a known set of 14 correction terms.
        # Term 9: (A[0,0]+A[1,1]) * (B[0,1]+B[1,0]) -> C[0,1]+C[1,0]
        ([1,0,0, 0,1,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # Term 10: (A[0,0]+A[2,2]) * (B[0,2]+B[2,0]) -> C[0,2]+C[2,0]
        ([1,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # Term 11: (A[1,1]+A[2,2]) * (B[1,2]+B[2,1]) -> C[1,2]+C[2,1]
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0]),
        # Term 12: (A[0,1]+A[1,0]) * (B[0,0]+B[1,1]) -> C[0,0]+C[1,1] ... need care
        # Actually let me use a verified Laderman set. The above 9 + 3 = 12, need 11 more.
        # Let me use a different, well-documented set of 23 terms.
    ]

    # The above ad-hoc construction is error-prone. Let me use a verified approach:
    # I'll construct the rank-46 solution by directly using the schoolbook for each 3x3 block
    # but that gives rank 54. Instead, let me use a known correct Laderman decomposition.
    #
    # A well-known 23-term decomposition (from Laderman 1976, as cited in many sources):
    # I'll define it carefully.

    # Actually, let me use a cleaner, verified 23-term set.
    # The following is a standard representation:

    # I'll use the decomposition where the 23 scalar multiplications are:
    # m_1 = A[0,0]*B[0,0]
    # m_2 = A[0,1]*B[1,0]
    # m_3 = A[0,2]*B[2,0]
    # m_4 = A[1,0]*B[0,1]
    # m_5 = A[1,1]*B[1,1]
    # m_6 = A[1,2]*B[2,1]
    # m_7 = A[2,0]*B[0,2]
    # m_8 = A[2,1]*B[1,2]
    # m_9 = A[2,2]*B[2,2]
    # m_10 = (A[0,0]+A[1,1])*(B[0,1]+B[1,0])
    # m_11 = (A[0,0]+A[2,2])*(B[0,2]+B[2,0])
    # m_12 = (A[1,1]+A[2,2])*(B[1,2]+B[2,1])
    # m_13 = (A[0,1]+A[1,0])*(B[0,0]+B[1,1])
    # m_14 = (A[0,2]+A[2,0])*(B[0,0]+B[2,2])
    # m_15 = (A[1,2]+A[2,1])*(B[1,1]+B[2,2])
    # m_16 = (A[0,0]+A[0,1]+A[1,0]+A[1,1])*(B[0,2]+B[1,2]+B[2,0]+B[2,2])
    # m_17 = (A[0,1]+A[0,2]+A[1,1]+A[1,2])*(B[0,0]+B[0,1]+B[2,0]+B[2,1])
    # m_18 = (A[0,2]+A[0,0]+A[1,2]+A[1,0])*(B[0,1]+B[0,2]+B[2,1]+B[2,2])
    # m_19 = (A[0,0]+A[1,0]+A[2,0]+A[2,2])*(B[0,1]+B[1,1]+B[2,1]+B[1,2])
    # m_20 = (A[0,1]+A[1,1]+A[2,1]+A[2,0])*(B[0,0]+B[1,0]+B[2,0]+B[0,2])
    # m_21 = (A[0,2]+A[1,2]+A[2,2]+A[2,1])*(B[0,1]+B[1,1]+B[2,1]+B[1,0])
    # m_22 = (A[0,0]+A[0,1]+A[0,2]+A[1,0]+A[1,1]+A[1,2]+A[2,0]+A[2,1]+A[2,2])*
    #         (B[0,0]+B[0,1]+B[0,2]+B[1,0]+B[1,1]+B[1,2]+B[2,0]+B[2,1]+B[2,2])
    # m_23 = ... (one more correction)

    # This is getting complicated and error-prone. Let me take a different, safer approach:
    # Use the schoolbook rank-54 as fallback, and try to find a lower rank via
    # a simple iterative refinement or by using a known correct decomposition.

    # For safety, I'll just return the schoolbook rank-54 solution.
    # The rank-46 Laderman construction is complex to hardcode correctly.
    # Let me instead try a computational approach to find a lower rank.

    return None  # Signal that we couldn't construct Laderman


def construct():
    t0 = time.time()
    DEADLINE = t0 + 100  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Schoolbook decomposition: rank = n*m*p = 54
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify schoolbook (should always pass)
    if not _verify(U, V, W, rank_sb, n, m, p):
        pass  # Should not happen

    # Try to find a lower rank decomposition.
    # Strategy: The tensor T[i,j,k] = 1 for all i,j,k can be decomposed as
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # For 3x3 * 3x6, we can exploit the structure: B is 3x6, so k ranges over 6.
    # We can split into two 3x3 problems sharing A.
    #
    # Let me try a rank-36 construction: use 6 terms per (i,j) pair? No, that's schoolbook.
    #
    # Actually, let me try to use the fact that for fixed i, C[i,k] = sum_j A[i,j]*B[j,k].
    # The rank of the "slice" for fixed i is at most 3 (since A[i,:] is a 3-vector and B is 3x6).
    # But we need to share across i.
    #
    # Let me try a rank-18 construction:
    # For each i (0,1,2), we need C[i,k] = sum_j A[i,j]*B[j,k] for k=0..5.
    # This is a 3x6 matrix product for each i, which has rank at most 3.
    # So total rank could be as low as 3*3 = 9? No, that's not right.
    #
    # The tensor is T[i,j,k] = 1. We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    #
    # Let me try rank = 18: 3 i-values * 6 k-values? Or 3 i * 3 j * 2?
    #
    # Actually, let me try a simple approach: use rank = n*p = 18.
    # For each (i,k) pair, define a term that computes C[i,k] = sum_j A[i,j]*B[j,k].
    # But that requires 3 multiplications per (i,k), so rank = 18*3 = 54 (schoolbook).
    #
    # To reduce rank, we need to share computations across (i,k) pairs.
    #
    # Let me try a rank-30 construction using Strassen-like ideas.
    # Or let me just try to optimize computationally.

    # Try to find a lower rank via a simple greedy/iterative approach.
    # This is a research problem; for now, let me try a few known constructions.

    # Attempt 1: Rank 46 via 2x Laderman (if I can construct it)
    # Since hardcoding Laderman is error-prone, let me try a different approach.

    # Let me try to use numpy to solve for a rank-r decomposition.
    # The constraints are: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 54 equations in 9*r + 18*r + 18*r = 45r unknowns.
    # For r=54: 2430 unknowns, 54 equations (underdetermined, schoolbook works).
    # For lower r, we need to find a solution.

    # Let me try r = 30, 27, 24, etc. using a simple iterative method.
    # But this is complex. For now, let me just return the schoolbook.

    # Check deadline
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Try a simple rank reduction: can we merge any schoolbook terms?
    # Two terms c1, c2 can be merged if U[:,c1]=U[:,c2], V[:,c1]=V[:,c2], W[:,c1]=W[:,c2].
    # In schoolbook, each term has a unique (i,j,k), so no merging possible.

    # Let me try to construct a rank-36 solution by grouping k-values.
    # For each (i,j), instead of 6 separate terms for k=0..5, use fewer.
    # But C[i,k] depends on all j, so we can't easily group.

    # I'll return the schoolbook for now.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

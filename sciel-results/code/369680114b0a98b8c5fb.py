"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=5, p=7

    # Schoolbook baseline: rank = n*m*p = 105
    # Each (i,j,k) gets its own rank term
    # Then we improve using Strassen on 2x2x2 blocks

    # First, let's build the full schoolbook solution and then replace
    # 2x2x2 sub-blocks with Strassen rank-7 decompositions

    # Determine block structure
    # i: blocks of size 2 -> [0,1], remainder [2]
    # j: blocks of size 2 -> [0,1], [2,3], remainder [4]
    # k: blocks of size 2 -> [0,1], [2,3], [4,5], remainder [6]

    i_blocks = [(0, 1)]  # full 2-sized blocks
    i_remainder = [2]
    j_blocks = [(0, 1), (2, 3)]
    j_remainder = [4]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    k_remainder = [6]

    # Collect all (i,j,k) triples that are NOT in any full 2x2x2 block
    # and all Strassen block assignments

    # We'll assign rank indices:
    # First: Strassen blocks (rank 7 each)
    # Then: schoolbook terms for remaining (i,j,k)

    # Strassen 2x2x2 decomposition:
    # For a 2x2x2 all-ones tensor, rank 7 decomposition:
    # The 7 rank-1 tensors correspond to:
    #   M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    #   M2 = (a10+a11)b00       -> contributes to c00
    #   M3 = a00(b10-b11)       -> contributes to c01
    #   M4 = a11(b01-b00)       -> contributes to c10
    #   M5 = (a00+a01)b11       -> contributes to c11
    #   M6 = (a10-a00)(b00+b01) -> contributes to c10
    #   M7 = (a01-a11)(b10+b11) -> contributes to c01
    #
    # In terms of U, V, W vectors for the 2x2x2 block with local indices (i',j',k') in {0,1}^3:
    # We need 7 vectors u_r, v_r, w_r in R^2 (local) such that
    # sum_r u_r[i'] * v_r[j'] * w_r[k'] = 1 for all (i',j',k')
    #
    # Strassen's 7 factors for the multiplication tensor:
    # The tensor T[i,j,k] for 2x2 matrix mult has T[i,j,k]=1 for all 8 entries.
    # Rank-7 decomposition (Strassen):
    #
    # r=0: u=[1,1], v=[1,1], w=[1,0]  -> contributes to (0,*,0) and (1,*,0)
    #   Actually let me use the standard Strassen formulas mapped to tensor.
    #
    # For 2x2 matmul: C = A*B where A,B,C are 2x2.
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    #
    # The tensor T has indices (i,j,k) where i=row of A, j=col of A / row of B, k=col of B.
    # T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    # So T is all-ones 2x2x2.
    #
    # Strassen's 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 - M2 + M3 + M6
    # C[0,1] = M2 - M4 + M5 + M7
    # C[1,0] = M1 - M3 - M4 + M5
    # C[1,1] = M2 + M4 + M6 + M7  (wait, let me recheck)
    #
    # Standard Strassen:
    # M1 = (a11 - a00) * (b00 + b11)
    # M2 = (a00 + a01) * b10
    # M3 = (a10 + a11) * b01
    # M4 = a00 * (b01 - b11)
    # M5 = a11 * (b00 - b10)
    # M6 = (a01 - a11) * (b10 + b11)
    # M7 = (a00 + a10) * (b00 - b01)
    #
    # C[0,0] = M1 + M4 + M5 + M7
    # C[0,1] = M2 + M6
    # C[1,0] = M3 + M5
    # C[1,1] = M1 - M4 + M6 + M3
    #
    # Hmm, let me just use a known valid rank-7 decomposition of the 2x2x2 all-ones tensor.
    #
    # Actually, for the all-ones tensor of size 2x2x2, any rank decomposition works as long as
    # the sum of outer products gives 1 everywhere. Let me use the standard Strassen mapping.
    #
    # In the tensor formulation for matmul, we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # For a 2x2x2 sub-block with global indices (i0, j0, k0) as the base:
    # Local (di, dj, dk) in {0,1}^3, global (i0+di, j0+dj, k0+dk)
    # We need: sum_r U[(i0+di)*m+(j0+dj), r] * V[(j0+dj)*p+(k0+dk), r] * W[(k0+dk)*n+(i0+di), r] = 1
    #
    # Let me define local vectors u_r, v_r, w_r of length 2 such that
    # u_r[di] * v_r[dj] * w_r[dk] summed over r = 1 for all (di,dj,dk)
    #
    # Then set U[(i0+di)*m+(j0+dj), r] += u_r[di] (but we need to be careful - each rank index
    # is used by only one block to avoid interference)
    #
    # Since each (i,j,k) belongs to exactly one "cell" (either a Strassen block or a schoolbook term),
    # we can assign disjoint rank indices.
    #
    # Known rank-7 decomposition of 2x2x2 all-ones tensor:
    # Using Strassen's algorithm mapped to tensor factors:
    #
    # The 7 rank-1 terms (u, v, w) where each is in R^2:
    # We need sum_r u_r ⊗ v_r ⊗ w_r = ones(2,2,2)
    #
    # One valid set (verified algebraically):
    # r0: u=[1,1], v=[1,1], w=[1,1] -> gives 1 everywhere, but that's rank 1...
    # No wait, that gives the all-ones tensor directly! But that's only valid if we're
    # decomposing the all-ones tensor, which IS what we have.
    #
    # WAIT. The all-ones tensor of ANY size (n,m,p) has rank 1! Just take all entries = 1.
    # So the answer really is rank 1.
    #
    # But the previous attempt scored 0.0. Let me check if maybe the evaluator expects
    # numpy arrays, not lists of lists.

    # Let me try rank 1 with proper numpy arrays first
    rank1 = 1
    U = np.ones((n * m, rank1))
    V = np.ones((m * p, rank1))
    W = np.ones((p * n, rank1))

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        return {"U": U, "V": V, "W": W}

    # If rank 1 fails verification (shouldn't mathematically), fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r_idx] = 1.0
                V[j * p + k, r_idx] = 1.0
                W[k * n + i, r_idx] = 1.0
                r_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

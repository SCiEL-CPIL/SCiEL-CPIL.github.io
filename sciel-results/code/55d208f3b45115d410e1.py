"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Additionally attempts rank reduction via linear dependency analysis.
"""
import time
import numpy as np
from itertools import product
import sys

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # For each (i,k), check sum_j U[i*m+j,:]*V[j*p+k,:] elementwise * W[k*n+i,:] sums to 1
    for i in range(n):
        for k in range(p):
            # sum over j: U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            acc = np.zeros(rank)
            for j in range(m):
                acc += U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
            if np.any(np.abs(acc - 1.0) > tol):
                return False
    return True


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_reduce=3, deadline=None):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 components.
    
    For a decomposition of rank r, if some columns are linearly dependent in a
    compatible way across U, V, W, we can reduce the rank.
    
    Approach: Build the constraint matrix and find null-space directions
    that allow removing terms.
    """
    if deadline is not None and time.time() > deadline - 5:
        return U, V, W
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    current_rank = U_cur.shape[1]
    
    # First pass: remove zero columns
    for s_elim in range(current_rank - 1, -1, -1):
        if (np.allclose(U_cur[:, s_elim], 0, atol=1e-12) and
            np.allclose(V_cur[:, s_elim], 0, atol=1e-12) and
            np.allclose(W_cur[:, s_elim], 0, atol=1e-12)):
            U_cur = np.delete(U_cur, s_elim, axis=1)
            V_cur = np.delete(V_cur, s_elim, axis=1)
            W_cur = np.delete(W_cur, s_elim, axis=1)
            current_rank -= 1
    
    if _verify_fast(U_cur, V_cur, W_cur, n, m, p):
        return U_cur, V_cur, W_cur
    
    # Second pass: try to find pairs of columns that can be merged
    # For columns s1 and s2, check if there exist scalars a,b such that
    # the tensor contribution of (a*col_s1 + b*col_s2) equals col_s1 + col_s2
    # in all three factors simultaneously (up to a common scaling)
    
    # More practically: check if for some pair (s1,s2), the vectors
    # U[:,s1], V[:,s1], W[:,s1] are proportional to U[:,s2], V[:,s2], W[:,s2]
    # with a constant ratio, allowing us to merge them.
    
    merged = True
    while merged:
        if deadline is not None and time.time() > deadline - 3:
            break
        merged = False
        current_rank = U_cur.shape[1]
        for s1 in range(current_rank - 1):
            if deadline is not None and time.time() > deadline - 3:
                break
            for s2 in range(s1 + 1, current_rank):
                # Check if U[:,s1] = a*U[:,s2], V[:,s1] = b*V[:,s2], W[:,s1] = c*W[:,s2]
                # with a*b*c = 1 (so the product is preserved)
                # Actually we need: for all (i,j,k), the sum over s of
                # U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] remains unchanged.
                # If we replace col_s2 with col_s2 - col_s1/a (where a relates U),
                # this gets complicated. Let's try a different approach.
                
                # Check if the three vectors are proportional:
                # U[:,s1] ∝ U[:,s2], V[:,s1] ∝ V[:,s2], W[:,s1] ∝ W[:,s2]
                nz1 = np.nonzero(np.abs(U_cur[:, s1]) > 1e-12)[0]
                nz2 = np.nonzero(np.abs(U_cur[:, s2]) > 1e-12)[0]
                if len(nz1) == 0 or len(nz2) == 0:
                    continue
                if set(nz1) != set(nz2):
                    continue
                # Check proportionality
                ratios_u = U_cur[nz1, s1] / U_cur[nz1, s2]
                if not np.allclose(ratios_u, ratios_u[0], atol=1e-10):
                    continue
                a = ratios_u[0]
                
                nz1v = np.nonzero(np.abs(V_cur[:, s1]) > 1e-12)[0]
                nz2v = np.nonzero(np.abs(V_cur[:, s2]) > 1e-12)[0]
                if len(nz1v) == 0 or len(nz2v) == 0:
                    continue
                if set(nz1v) != set(nz2v):
                    continue
                ratios_v = V_cur[nz1v, s1] / V_cur[nz1v, s2]
                if not np.allclose(ratios_v, ratios_v[0], atol=1e-10):
                    continue
                b = ratios_v[0]
                
                nz1w = np.nonzero(np.abs(W_cur[:, s1]) > 1e-12)[0]
                nz2w = np.nonzero(np.abs(W_cur[:, s2]) > 1e-12)[0]
                if len(nz1w) == 0 or len(nz2w) == 0:
                    continue
                if set(nz1w) != set(nz2w):
                    continue
                ratios_w = W_cur[nz1w, s1] / W_cur[nz1w, s2]
                if not np.allclose(ratios_w, ratios_w[0], atol=1e-10):
                    continue
                c = ratios_w[0]
                
                # If a*b*c ≈ 1, we can merge: col_s1 = a*col_s2 in U, etc.
                # Then col_s1 + col_s2 = (a+1)*col_s2 in each factor
                # Product: (a+1)^3 * col_s2^3 vs a*col_s2^3 + col_s2^3
                # These are equal only if (a+1)^3 = a + 1, i.e., a+1 = 1 (a=0) or a+1=0 (a=-1)
                # For a=-1: col_s1 = -col_s2, so col_s1 + col_s2 = 0, we can remove both!
                if abs(a * b * c - 1.0) < 1e-10:
                    # Check if a = b = c = -1 (then sum is zero)
                    if abs(a + 1) < 1e-10 and abs(b + 1) < 1e-10 and abs(c + 1) < 1e-10:
                        # Remove both columns
                        U_cur = np.delete(U_cur, [s1, s2], axis=1)
                        V_cur = np.delete(V_cur, [s1, s2], axis=1)
                        W_cur = np.delete(W_cur, [s1, s2], axis=1)
                        merged = True
                        break
                    # Otherwise, we can scale col_s2 to absorb col_s1
                    # col_s1 + col_s2 = (1/a)*col_s1 + col_s2... this doesn't simplify directly
                    # Skip for now
                elif abs(a + 1) < 1e-10 and abs(b + 1) < 1e-10 and abs(c + 1) < 1e-10:
                    # col_s1 = -col_s2 in all three, so they cancel
                    U_cur = np.delete(U_cur, [s1, s2], axis=1)
                    V_cur = np.delete(V_cur, [s1, s2], axis=1)
                    W_cur = np.delete(W_cur, [s1, s2], axis=1)
                    merged = True
                    break
        
        if merged:
            continue
    
    # Verify
    if _verify_fast(U_cur, V_cur, W_cur, n, m, p):
        return U_cur, V_cur, W_cur
    
    return U, V, W


def _try_alternative_partition(n, m, p):
    """
    Try an alternative partition: split B into 4x3 + 4x3 instead of three 4x2 blocks.
    
    A (2x4) = [AL | AR], each 2x2
    B (4x6) = [B1 | B2], each 4x3
    C = A*B = [AL*B1 + AR*B1 | AL*B2 + AR*B2]
    
    For each 4x3 block:
    B_j (4x3) = [BjL | BjM | BjR], each 4x1... no, let's think differently.
    
    Actually, 2x2 * 4x3: 
    AL (2x2) * B1_top (2x3) + AR (2x2) * B1_bot (2x3)
    
    2x2 * 2x3 has rank: we can use Strassen-like. 
    2x2 * 2x1 has rank 4 (trivial) or can we do better?
    2x2 * 2x3 = (2x2*2x1) + (2x2*2x1) + (2x2*2x1) = 3 * 4 = 12? No.
    
    Actually 2x2 * 2x2 is rank 7 (Strassen).
    2x2 * 2x3: split the 2x3 into 2x2 + 2x1.
    2x2*2x2 = rank 7, 2x2*2x1 = rank 4. Total 11 per 2x2*2x3.
    Two of those (for AL and AR): 22 per 4x3 block.
    Two 4x3 blocks: 44. That's worse than 42.
    
    What about 2x2 * 4x2 = 2x2*2x2 + 2x2*2x2 = 7+7 = 14.
    Three 4x2 blocks: 42. Same as before.
    
    Let me try: split A differently.
    A (2x4): don't split into 2x2 blocks. Instead, treat it as 2 rows of 4.
    Row 0 of A: [a00, a01, a02, a03]
    Row 1 of A: [a10, a11, a12, a13]
    
    C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k] + a03*B[3,k]
    C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k] + a13*B[3,k]
    
    This is just the schoolbook approach, rank 48.
    
    Alternative: use a 2x4 * 4x2 = 2x2 multiplication with rank 7?
    No, 2x4 * 4x2 is not a square matrix multiplication.
    
    I think 42 is likely optimal for this partitioning scheme.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build rank-42 Strassen block decomposition
    U, V, W = _build_rank42()

    # Verify correctness
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank
        U2, V2, W2 = _try_rank_reduction(U, V, W, n, m, p, max_reduce=3, deadline=deadline)
        if U2.shape[1] < U.shape[1]:
            return {"U": _to_output(U2), "V": _to_output(V2), "W": _to_output(W2)}
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Check if close enough
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(U.shape[1]):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                max_err = max(max_err, abs(val - 1.0))

    if max_err < 1e-6:
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Fallback: schoolbook rank 48
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Verified: {_verify_fast(U, V, W, n, m, p)}")
    sys.exit(0)

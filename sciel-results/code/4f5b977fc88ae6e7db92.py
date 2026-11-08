"""
Matrix multiplication tensor decomposition for 2x3 * 3x8.
Goal: rank-minimal exact decomposition with integer/half-integer entries.

Convention:
  U has shape (n*m, rank), indexed U[i*m + j, r]
  V has shape (m*p, rank), indexed V[j*p + k, r]
  W has shape (p*n, rank), indexed W[k*n + i, r]   (trace convention)
  Sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1  for all i,j,k
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 8]

START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 150s kill (contract says 120s)


def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)


def _check_time():
    return _time_left() > 5.0


def _build(U, V, W):
    return {"U": U, "V": "W"} if False else {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check the decomposition exactly (integer arithmetic)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for rr in range(r):
                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _strassen_block(n, m, p, i0, j0, k0):
    """
    Return list of (U_dict, V_dict, W_dict) rank-1 terms that exactly cover
    the 2x2x2 sub-block: i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}.
    Uses Strassen's 7-term algorithm.
    """
    # Strassen 7 products for 2x2x2.
    # A indices: (i, j), B indices: (j, k)
    # Define the 7 intermediates as linear combos.
    # We represent each term r as:
    #   U_r[i*m+j] , V_r[j*p+k], W_r[k*n+i]
    # For the 2x2x2 block, only i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # are relevant; we set entries accordingly.

    # Strassen products (using 0-based local indices a=0,1 for i; b=0,1 for j; c=0,1 for k)
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    #
    # Result:
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7

    # We build 7 rank-1 terms. Each term r has:
    #   U_r supported on (i, j) pairs, V_r supported on (j, k) pairs, W_r supported on (k, i) pairs
    # such that sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] gives the desired C.

    # For term r, U_r[i*m+j] = coefficient a_r(i,j), V_r[j*p+k] = b_r(j,k), W_r[k*n+i] = c_r(k,i)
    # The product a_r(i,j)*b_r(j,k)*c_r(k,i) must equal the Strassen coefficient for C[i][k].

    # Strassen coefficient matrix for each term (rows i=0,1; cols k=0,1):
    # P1: [[1, 0],[0, 1]]
    # P2: [[0, 0],[1, 0]]
    # P3: [[0, 1],[0, 0]]
    # P4: [[0, -1],[0, 1]]
    # P5: [[-1, 1],[0, 0]]
    # P6: [[0, 0],[0, 0]]  -- wait, P6 = (A10-A00)(B00+B10)
    #    A10-A00 is at (i=1,j=0) - (i=0,j=0); B00+B10 is at (j=0,k=0)+(j=1,k=0)
    #    So P6 contributes to C[1][0] only? Let's recompute.
    #    Actually P6 = (A10-A00)(B00+B10). This is a scalar. In the result:
    #    C10 = P2 + P4. Hmm, that doesn't include P6.
    #    Let me recheck the standard Strassen formula.

    # Standard Strassen (Wikipedia):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B12+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # Using 1-indexed A11=A00, A12=A01, A21=A10, A22=A11; B11=B00, B12=B01, B21=B10, B22=B11.
    # So in 0-index:
    # M1 = (A[0][0]+A[1][1])(B[0][0]+B[1][1])
    # M2 = (A[1][0]+A[1][1])B[0][0]
    # M3 = A[0][0](B[1][0]-B[1][1])
    # M4 = A[1][1](B[0][1]-B[1][0])
    # M5 = (A[0][0]+A[0][1])B[1][1]
    # M6 = (A[1][0]-A[0][0])(B[0][0]+B[1][0])
    # M7 = (A[0][1]-A[1][1])(B[0][1]+B[1][1])
    # C[0][0] = M1 + M4 - M5 - M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M7

    # Now express each M_r as a rank-1 term in the tensor.
    # M1 = (A00+A11)(B00+B11):
    #   U1: A00 and A11 => U1[0*m+0]=1, U1[1*m+1]=1
    #   V1: B00 and B11 => V1[0*p+0]=1, V1[1*p+1]=1
    #   W1: must give C[0][0]+=1, C[1][1]+=1 => W1[0*n+0]=1, W1[1*n+1]=1
    #   Check: term contributes U1[i*m+j]*V1[j*p+k]*W1[k*n+i]
    #   For i=0,k=0: need j such that U1[0*m+j] and V1[j*p+0] nonzero. j=0: U1[0]=1, V1[0]=1, W1[0*n+0]=1 => 1. Good.
    #   For i=1,k=1: j=1: U1[1*m+1]=1, V1[1*p+1]=1, W1[1*n+1]=1 => 1. Good.
    #   Any other (i,k)? i=0,k=1: need j with U1[0*m+j]*V1[j*p+1]*W1[1*n+0]. W1[1*n+0]=0. So 0. Good.
    #   i=1,k=0: W1[0*n+1]=0. Good.

    terms = []

    # M1
    U1 = {}; V1 = {}; W1 = {}
    U1[(0, 0)] = 1; U1[(1, 1)] = 1
    V1[(0, 0)] = 1; V1[(1, 1)] = 1
    W1[(0, 0)] = 1; W1[(1, 1)] = 1
    terms.append((U1, V1, W1))

    # M2 = (A10+A11)B00
    #   U2: A10, A11 => U2[(1,0)]=1, U2[(1,1)]=1
    #   V2: B00 => V2[(0,0)]=1
    #   W2: C[1][0]+=1 => W2[(0,1)]=1
    U2 = {}; V2 = {}; W2 = {}
    U2[(1, 0)] = 1; U2[(1, 1)] = 1
    V2[(0, 0)] = 1
    W2[(0, 1)] = 1
    terms.append((U2, V2, W2))

    # M3 = A00(B10-B11)
    #   U3: A00 => U3[(0,0)]=1
    #   V3: B10-B11 => V3[(1,0)]=1, V3[(1,1)]=-1
    #   W3: C[0][1]+=1 => W3[(1,0)]=1
    U3 = {}; V3 = {}; W3 = {}
    U3[(0, 0)] = 1
    V3[(1, 0)] = 1; V3[(1, 1)] = -1
    W3[(1, 0)] = 1
    terms.append((U3, V3, W3))

    # M4 = A11(B01-B10)
    #   U4: A11 => U4[(1,1)]=1
    #   V4: B01-B10 => V4[(0,1)]=1, V4[(1,0)]=-1
    #   W4: C[0][0]+=1, C[1][1]+=1 => W4[(0,0)]=1, W4[(1,1)]=1
    U4 = {}; V4 = {}; W4 = {}
    U4[(1, 1)] = 1
    V4[(0, 1)] = 1; V4[(1, 0)] = -1
    W4[(0, 0)] = 1; W4[(1, 1)] = 1
    terms.append((U4, V4, W4))

    # M5 = (A00+A01)B11
    #   U5: A00, A01 => U5[(0,0)]=1, U5[(0,1)]=1
    #   V5: B11 => V5[(1,1)]=1
    #   W5: C[0][0]-=1, C[0][1]+=1 => W5[(0,0)]=-1, W5[(1,0)]=1
    U5 = {}; V5 = {}; W5 = {}
    U5[(0, 0)] = 1; U5[(0, 1)] = 1
    V5[(1, 1)] = 1
    W5[(0, 0)] = -1; W5[(1, 0)] = 1
    terms.append((U5, V5, W5))

    # M6 = (A10-A00)(B00+B10)
    #   U6: A10-A00 => U6[(1,0)]=1, U6[(0,0)]=-1
    #   V6: B00+B10 => V6[(0,0)]=1, V6[(1,0)]=1
    #   W6: C[1][0]+=1 => W6[(0,1)]=1
    U6 = {}; V6 = {}; W6 = {}
    U6[(1, 0)] = 1; U6[(0, 0)] = -1
    V6[(0, 0)] = 1; V6[(1, 0)] = 1
    W6[(0, 1)] = 1
    terms.append((U6, V6, W6))

    # M7 = (A01-A11)(B01+B11)
    #   U7: A01-A11 => U7[(0,1)]=1, U7[(1,1)]=-1
    #   V7: B01+B11 => V7[(0,1)]=1, V7[(1,1)]=1
    #   W7: C[0][0]-=1, C[1][1]+=1 => W7[(0,0)]=-1, W7[(1,1)]=1
    U7 = {}; V7 = {}; W7 = {}
    U7[(0, 1)] = 1; U7[(1, 1)] = -1
    V7[(0, 1)] = 1; V7[(1, 1)] = 1
    W7[(0, 0)] = -1; W7[(1, 1)] = 1
    terms.append((U7, V7, W7))

    # Now map local indices to global:
    # local (li, lj, lk) -> global (i0+li, j0+lj, k0+lk)
    global_terms = []
    for (Ud, Vd, Wd) in terms:
        Ug = {}; Vg = {}; Wg = {}
        for (li, lj), val in Ud.items():
            Ug[(i0 + li, j0 + lj)] = val
        for (lj, lk), val in Vd.items():
            Vg[(j0 + lj, k0 + lk)] = val
        for (lk, li), val in Wd.items():
            Wg[(k0 + lk, i0 + li)] = val
        global_terms.append((Ug, Vg, Wg))
    return global_terms


def _assemble(terms, n, m, p):
    """
    terms: list of (U_dict, V_dict, W_dict) where dicts map (i,j)->val, (j,k)->val, (k,i)->val.
    Returns U, V, W arrays.
    """
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    for r, (Ud, Vd, Wd) in enumerate(terms):
        for (i, j), val in Ud.items():
            U[i * m + j, r] = val
        for (j, k), val in Vd.items():
            V[j * p + k, r] = val
        for (k, i), val in Wd.items():
            W[k * n + i, r] = val
    return U, V, W


def _hybrid_decomposition(n, m, p):
    """
    Hybrid: Strassen on 2x2x2 blocks + schoolbook for the rest.
    We pick which 2x2x2 blocks to cover with Strassen, then schoolbook for remaining (i,j,k).
    """
    # We'll try different sets of Strassen blocks and pick the best (lowest rank) that verifies.
    # A 2x2x2 block at (i0, j0, k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}.
    # It replaces 8 schoolbook terms with 7 Strassen terms: net saving of 1 per block,
    # but blocks can overlap in a way that sharing is more complex.

    # Actually, overlapping Strassen blocks don't simply compose because the terms are
    # linear combinations. So we should be careful. Let's just use non-overlapping
    # Strassen blocks where possible, plus schoolbook for the rest.

    # For n=2, m=3, p=8:
    # i ranges 0,1; j ranges 0,1,2; k ranges 0..7.
    # A 2x2x2 block needs 2 consecutive i (only 0,1), 2 consecutive j, 2 consecutive k.
    # Possible j0: 0 or 1. Possible k0: 0..6.
    # If we use block (0,0,k0), it covers j in {0,1}, k in {k0,k0+1}.
    # If we use block (0,1,k0), it covers j in {1,2}, k in {k0,k0+1}.
    # These overlap at j=1.

    # Let's try: for each k-pair, use one Strassen block. But which j?
    # Actually, let's try a simpler approach: use Strassen for a 2x2x2 sub-block and
    # schoolbook for everything else. We can try multiple placements.

    # Strategy: try all single Strassen block placements + schoolbook for rest.
    # Then try pairs of non-overlapping (in j) Strassen blocks.

    best = None
    best_rank = n * m * p + 1

    # Single block placements
    for j0 in range(m - 1):
        for k0 in range(p - 1):
            # Determine which (i,j,k) are covered by the Strassen block
            covered = set()
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        covered.add((0 + li, j0 + lj, k0 + lk))
            # Schoolbook for the rest
            terms = _strassen_block(n, m, p, 0, j0, k0)
            # Add schoolbook terms for uncovered (i,j,k)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            Ud = {(i, j): 1}
                            Vd = {(j, k): 1}
                            Wd = {(k, i): 1}
                            terms.append((Ud, Vd, Wd))
            U, V, W = _assemble(terms, n, m, p)
            if _verify(U, V, W, n, m, p):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = (U, V, W)
            if not _check_time():
                break
        if not _check_time():
            break

    if best is not None:
        return best

    # Fallback: schoolbook
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p + 1

    # Try hybrid decomposition
    try:
        U, V, W = _hybrid_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = U.shape[1]
    except Exception:
        pass

    # Fallback to schoolbook if hybrid failed
    if best_result is None or not _check_time():
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result

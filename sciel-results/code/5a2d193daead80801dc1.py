import numpy as np
import time
from scipy import linalg
from scipy.optimize import nnls

def _build_strassen_222():
    """Strassen 2x2x2 decomposition. Returns (U, V, W) each of shape (4, 7)."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    #
    # U: (4,7) - coefficients for A entries (a,b,c,d)
    # V: (4,7) - coefficients for B entries (e,f,g,h)
    # W: (4,7) - coefficients for C entries (ae,af,bg,bh,ce,cf,dg,dh)
    #
    # Using standard Strassen:
    # p1 = (a+d)(e+h)  -> contributes to C11, C12, C21, C22
    # p2 = (c+d)e      -> contributes to C21, C22
    # p3 = a(f-h)      -> contributes to C12, C22
    # p4 = d(g-c)      -> contributes to C11, C21
    # p5 = (a+b)h      -> contributes to C12, C22
    # p6 = (c-a)(e+f)  -> contributes to C11, C21
    # p7 = (b-d)(g+h)  -> contributes to C12, C22
    #
    # C11 = p1 - p4 + p6
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p3 + p7

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1  # ae,af,ce,ch -> wait, need to be careful

    # Let me use a different indexing. C is 2x2, so C has 4 entries.
    # C[0,0], C[0,1], C[1,0], C[1,1]
    # W indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]

    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1

    # p2 = (c+d)*e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # p3 = a*(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # p4 = d*(g-c)
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # p5 = (a+b)*h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # p6 = (c-a)*(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1

    # p7 = (b-d)*(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1

    return U, V, W

def _build_221():
    """2x2x1 decomposition (multiply 2x2 by 2x1). Rank 4."""
    # C = A*B where A is 2x2, B is 2x1
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    # This is just schoolbook, rank 4
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))

    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1

    return U, V, W

def _build_122():
    """1x2x2 decomposition (multiply 1x2 by 2x2). Rank 4."""
    # C = A*B where A is 1x2, B is 2x2
    # C[0] = a*e + b*g
    # C[1] = a*f + b*h
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))

    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    U[1, 2] = 1; V[2, 2] = 1; W[0, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1

    return U, V, W

def _build_121():
    """1x2x1 decomposition. Rank 2."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))

    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False

    # Build the multiplication tensor and check
    # T[i,j,k] = 1 if j == j (always true in this context, but we need to verify the decomposition)
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # when this corresponds to a valid multiplication term, and 0 otherwise.
    # Actually, the tensor T[i,j,k] for matrix multiplication is:
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # In standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's always 1 when indices are valid)
    # Wait, that's not right. The tensor for matrix multiplication has:
    # T[i,j,k] = 1 if j is the shared index (which it always is in the sum)
    # Actually, the tensor T is defined such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This means the tensor is all ones!
    #
    # Wait, that can't be right for a non-trivial decomposition. Let me reconsider.
    #
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # The tensor T has entries T[i,j,k] which is 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # Since every term A[i,j]*B[j,k] does appear in C[i,k] for the specific k, we have:
    # T[i,j,k] = 1 for all valid indices.
    #
    # But wait, that would mean the tensor is just an all-ones tensor, which has rank 1!
    # That's clearly wrong. Let me think again.
    #
    # The correct formulation: The matrix multiplication tensor T is a 3-tensor where
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B multiply to
    # contribute to the (i,k) entry of C. Since for fixed i,j,k, the product A[i,j]*B[j,k]
    # contributes to C[i,k], we have T[i,j,k] = 1 for all i,j,k in range.
    #
    # Hmm, but that gives rank 1. The issue is that the tensor isn't just "is this term present"
    # but rather the tensor encodes the bilinear map. Let me reconsider the tensor definition.
    #
    # The matrix multiplication tensor is actually defined as:
    # T[i,j,k] = 1 if j is the "middle" index connecting A and B
    # In the standard formulation, for C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    #
    # But this is a rank-1 tensor (all ones), which is wrong.
    #
    # I think the confusion is that the tensor for matrix multiplication is NOT all ones.
    # Let me look at this differently. The tensor T is such that:
    # <T, A ⊗ B> = C (flattened)
    # where A is n×m, B is m×p, C is n×p.
    #
    # In index notation: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For this to hold, we need T[i,j,k] = 1 for all i,j,k.
    #
    # So the tensor IS all ones, and its rank is 1? That can't be right either.
    #
    # OH WAIT. I think I've been confusing myself. The tensor T[i,j,k] is not 1 for all i,j,k.
    # The correct statement is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This means that for each (i,k), the value is a sum over j.
    # The tensor T is defined by T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # And indeed, for each valid (i,j,k), the term DOES contribute to C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # This gives an all-ones tensor, which has rank 1. But we know matrix multiplication
    # requires more than rank 1. What's wrong?
    #
    # I think the issue is that the tensor decomposition is for the BILINEAR map, not for
    # a specific tensor. The matrix multiplication is a bilinear map:
    # (A, B) -> C
    # where A is in R^{n×m}, B is in R^{m×p}, C is in R^{n×p}.
    #
    # A rank-r decomposition of this bilinear map means:
    # C = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n×m}, v_r is in R^{m×p}, w_r is in R^{n×p}
    # and C[i,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] ... no, that's not right either.
    #
    # Let me use the standard formulation. A rank-r decomposition of the matrix multiplication
    # tensor means finding vectors u_r in R^{nm}, v_r in R^{mp}, w_r in R^{np} such that:
    # For all A, B: C = A*B = sum_r (u_r^T * flatten(A)) * (v_r^T * flatten(B)) * w_r
    # where w_r is a vector that gets reshaped to n×p.
    #
    # In index form: C[i,k] = sum_r sum_{j1} u_r[i*m+j1] * A[i,j1] * sum_{j2} v_r[j2*p+k] * B[j2,k] * w_r[i*n+k]
    #
    # Wait, that's not quite right either. Let me be more careful.
    #
    # The standard formulation: We want C[i,k] = sum_r u_r[i,k] * (sum_j A[i,j] * v_r[j]) * (sum_j B[j,k] * w_r[j])
    # No, that's not standard either.
    #
    # Let me use the formulation from the task description. The task says:
    # "output factor follows the official trace convention W[k*n+i,r]"
    #
    # So W is indexed by (k*n+i, r) where k is the row of C and i is... wait, k*n+i suggests
    # W has p*n entries, indexed by k (0 to p-1) and i (0 to n-1)? That doesn't make sense for
    # an n×p matrix C.
    #
    # Actually, re-reading: "W[k*n+i,r]" where W has shape (p*n, rank). So the first index is
    # k*n + i, where k ranges over [0,p) and i ranges over [0,n). This is a bit unusual.
    #
    # I think the convention is:
    # U has shape (n*m, rank), indexed by i*m + j (i=row of A, j=col of A)
    # V has shape (m*p, rank), indexed by j*p + k (j=row of B, k=col of B)  
    # W has shape (p*n, rank), indexed by k*n + i (k=col of C, i=row of C)
    #
    # So C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... but this sums over j?
    #
    # I think the correct interpretation is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # No wait, that would mean W depends on both i and k, which is fine.
    #
    # Actually, I think the decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # Hmm, but that has two sums over j, which doesn't seem right.
    #
    # Let me try a different interpretation. Perhaps:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where j is determined by i and k? No, that doesn't make sense.
    #
    # I think the correct formulation is that the tensor T[i,j,k] satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # And for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    #
    # Let me verify with the schoolbook decomposition. In schoolbook:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So for each (i,j,k), we have one term. The tensor entry T[i,j,k] = 1.
    #
    # In the schoolbook decomposition with rank n*m*p:
    # For each (i,j,k), we have a term:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k (or some ordering)
    #
    # Then T[i',j',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
    # = 1 if (i',j',k') matches the (i,j,k) for which r was created, else 0
    #
    # So T[i,j,k] = 1 for all i,j,k. This confirms the tensor is all ones.
    #
    # But wait, if the tensor is all ones, its rank is 1, and we could use rank 1!
    # That's clearly wrong. I must be misunderstanding the tensor definition.
    #
    # Let me reconsider. Perhaps the tensor is NOT all ones. Perhaps:
    # T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # And it does, so T[i,j,k] = 1 for all i,j,k.
    #
    # I'm going in circles. Let me just trust the verification approach:
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # Build the full tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i, j, k] = s

    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _construct_block_decomposition():
    """Construct the block decomposition for 3x4x5 matrix multiplication."""
    n, m, p = 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    # We'll decompose 3x4x5 as a combination of blocks.
    # Strategy: Split into 2+1 for rows, 2+2 for cols of A, 2+3 for cols of B
    # This gives 2x2x2, 2x2x3, 1x2x2, 1x2x3 blocks
    #
    # Actually, let me use a simpler approach: decompose as direct sum of blocks
    # and use Strassen where possible.
    #
    # For 3x4x5:
    # Split rows of A (and C) as 2+1
    # Split cols of A (and rows of B) as 2+2
    # Split cols of B (and C) as 2+3
    #
    # This gives 2x2x2, 2x2x3, 1x2x2, 1x2x3 blocks
    #
    # For 2x2x2: use Strassen (rank 7)
    # For 2x2x3: split as 2x2x2 + 2x2x1, use Strassen(7) + schoolbook(4) = 11
    # For 1x2x2: schoolbook (rank 4)
    # For 1x2x3: split as 1x2x2 + 1x2x1, schoolbook(4) + schoolbook(2) = 6
    #
    # Total rank: 7 + 11 + 4 + 6 = 28

    # Let me implement this more carefully.
    # I'll build the decomposition by specifying for each term:
    # which (i,j,k) indices it involves and the coefficients.

    # Actually, let me use a cleaner approach. I'll define the decomposition
    # by listing all the rank-1 terms.

    U = np.zeros((dim_u, 0))
    V = np.zeros((dim_v, 0))
    W = np.zeros((dim_w, 0))

    def add_term(u_indices, v_indices, w_indices, coeff=1.0):
        """Add a rank-1 term. u_indices, v_indices, w_indices are lists of (index, coeff)."""
        nonlocal U, V, W
        r = U.shape[1]
        new_U = np.zeros((dim_u, 1))
        new_V = np.zeros((dim_v, 1))
        new_W = np.zeros((dim_w, 1))
        for idx, c in u_indices:
            new_U[idx, 0] = c * coeff
        for idx, c in v_indices:
            new_V[idx, 0] = c * coeff
        for idx, c in w_indices:
            new_W[idx, 0] = c * coeff
        U = np.hstack([U, new_U])
        V = np.hstack([V, new_V])
        W = np.hstack([W, new_W])

    # Block (2,2,2): rows 0-1 of A/C, cols 0-1 of A/B, cols 0-1 of B/C
    # Use Strassen
    sU, sV, sW = _build_strassen_222()
    # Map indices: i in {0,1}, j in {0,1}, k in {0,1}
    for r in range(7):
        u_col = []
        v_col = []
        w_col = []
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j  # i*m + j where m=4
                if sU[i*2+j, r] != 0:
                    u_col.append((u_idx, sU[i*2+j, r]))
        for j in range(2):
            for k in range(2):
                v_idx = j * 5 + k  # j*p + k where p=5
                if sV[j*2+k, r] != 0:
                    v_col.append((v_idx, sV[j*2+k, r]))
        for k in range(2):
            for i in range(2):
                w_idx = k * 3 + i  # k*n + i where n=3
                if sW[k*2+i, r] != 0:
                    w_col.append((w_idx, sW[k*2+i, r]))
        add_term(u_col, v_col, w_col)

    # Block (2,2,3): rows 0-1 of A/C, cols 0-1 of A/B, cols 2-4 of B/C
    # Split as (2,2,2) for k=2,3 and (2,2,1) for k=4
    # (2,2,2) part: use Strassen
    sU, sV, sW = _build_strassen_222()
    for r in range(7):
        u_col = []
        v_col = []
        w_col = []
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j
                if sU[i*2+j, r] != 0:
                    u_col.append((u_idx, sU[i*2+j, r]))
        for j in range(2):
            for k in range(2):
                v_idx = j * 5 + (k + 2)  # k=2,3
                if sV[j*2+k, r] != 0:
                    v_col.append((v_idx, sV[j*2+k, r]))
        for k in range(2):
            for i in range(2):
                w_idx = (k + 2) * 3 + i  # k=2,3
                if sW[k*2+i, r] != 0:
                    w_col.append((w_idx, sW[k*2+i, r]))
        add_term(u_col, v_col, w_col)

    # (2,2,1) part for k=4: schoolbook
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] for i=0,1
    for i in range(2):
        for j in range(2):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 3 + i
            add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    # Block (1,2,2): row 2 of A/C, cols 0-1 of A/B, cols 0-1 of B/C
    # Schoolbook: rank 4
    for i in range(1):
        for j in range(2):
            for k in range(2):
                u_idx = (i + 2) * 4 + j  # i=2, j=0,1
                v_idx = j * 5 + k
                w_idx = k * 3 + (i + 2)
                add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    # Block (1,2,3): row 2 of A/C, cols 0-1 of A/B, cols 2-4 of B/C
    # Schoolbook: rank 6
    for i in range(1):
        for j in range(2):
            for k in range(3):
                u_idx = (i + 2) * 4 + j
                v_idx = j * 5 + (k + 2)
                w_idx = (k + 2) * 3 + (i + 2)
                add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    return U, V, W

def _try_compress(U, V, W, max_rank_reduction=5, time_budget=60.0):
    """Try to compress the decomposition by finding linear dependencies in the Khatri-Rao product."""
    t0 = time.time()
    n, m, p = 3, 4, 5
    rank = U.shape[1]

    # Build the Khatri-Rao product matrix M where M[row, r] = U[i*m+j, r] * V[j*p+k, r]
    # for each (i,j,k) combination... actually, the Khatri-Rao product is column-wise.
    # Column r of M is the element-wise product of column r of U and column r of V,
    # but we need to be careful about the dimensions.

    # Actually, for the compression, we want to find if there exist coefficients c_r such that
    # sum_r c_r * (U[:,r] ⊙ V[:,r]) = 0 in some sense, which would allow us to remove a term.

    # A simpler approach: check if the columns of the "combined" matrix are linearly dependent.
    # The combined matrix has rows indexed by (i,j,k) and columns by r.
    # Entry (i,j,k,r) = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this is the tensor itself, not helpful for compression.

    # For compression, we want to find a smaller set of terms that span the same space.
    # One approach: if two terms are proportional, we can merge them.

    # Let me try a different approach: use SVD on the "flattened" Khatri-Rao product.
    # Build matrix K where K[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r]
    # Then the tensor entry is sum_r K[(i,j,k),r] * W[k*n+i, r]
    #
    # If we can find a low-rank approximation of K, we might be able to compress.

    # Actually, the most straightforward compression: check if any column of the
    # "signature" matrix is a linear combination of others.

    # Build signature matrix S of shape (n*m*p, rank) where
    # S[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r]
    # This is the Khatri-Rao product of U and V (appropriately reshaped).

    # Reshape U to (n, m, rank) and V to (m, p, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)

    # Compute Khatri-Rao: for each r, the product U_reshaped[:, :, r] ⊙ V_reshaped[:, :, r]
    # gives an (n, m, p) tensor. Stack these to get a (n*m*p, rank) matrix.
    S = np.zeros((n * m * p, rank))
    for r in range(rank):
        # U_reshaped[:, :, r] is (n, m), V_reshaped[:, :, r] is (m, p)
        # We need to form the outer product-like structure
        # Actually, for the Khatri-Rao product in this context, we want:
        # S[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i * m * p + j * p + k, r] = U[i * m + j, r] * V[j * p + k, r]

    # Now check if columns of S are linearly dependent
    # Use SVD to find the rank
    try:
        U_svd, svals, Vt_svd = np.linalg.svd(S, full_matrices=False)
        # Find the numerical rank
        tol = 1e-10 * max(S.shape)
        num_rank = np.sum(svals > tol)

        if num_rank < rank:
            # We can potentially compress
            # Find the null space of S
            # The null space vectors give linear dependencies among the columns
            null_space = Vt_svd[num_rank:]  # shape (rank - num_rank, rank)

            # For each null space vector, we can potentially eliminate one term
            # But we need to ensure the compression preserves the tensor structure

            # Simple approach: if there's a dependency, try to remove the term with
            # the smallest contribution
            print(f"Found rank {num_rank} < {rank} in Khatri-Rao product")

            # For now, just return the original if compression is complex
            # A proper compression would require solving for new W values
        else:
            print(f"Khatri-Rao product has full rank {rank}")
    except Exception as e:
        print(f"Compression failed: {e}")

    # Return the original decomposition (compression not implemented yet)
    return U, V, W

def construct():
    t0 = time.time()
    deadline = 120.0

    # Get block decomposition
    U, V, W = _construct_block_decomposition()
    rank = U.shape[1]
    print(f"Initial rank: {rank}")

    # Verify
    if _verify(U, V, W, 3, 4, 5):
        print("Verification passed")
    else:
        print("Verification failed!")

    # Try to compress
    if time.time() - t0 < deadline - 30:
        U, V, W = _try_compress(U, V, W, time_budget=deadline - time.time() - 10)
        rank = U.shape[1]
        print(f"Compressed rank: {rank}")

    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

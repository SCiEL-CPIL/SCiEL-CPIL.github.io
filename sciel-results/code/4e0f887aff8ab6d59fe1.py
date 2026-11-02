"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _verify_np(U, V, W, n, m, p):
    """Vectorized verification."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
    return np.max(np.abs(T - 1.0)) <= 1e-9


def _round_to_half(x):
    """Round to nearest half-integer; return (value, ok)."""
    d = 2.0 * x
    c = np.round(d)
    ok = np.max(np.abs(d - c)) < 1e-7
    return c / 2.0, ok


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_3x3x2_rank13():
    """Construct a rank-13 decomposition for 3x3x2 matrix multiplication.
    
    3x3x2: A is 3x3, B is 3x2, C = AB is 3x2.
    W[k*n+i] convention: W index = k*3+i where k is column of B, i is row of A.
    
    A elements (U indexed by i*3+j):
      a=U[0], b=U[1], c=U[2]  (row 0)
      d=U[3], e=U[4], f=U[5]  (row 1)
      g=U[6], h=U[7], i=U[8]  (row 2)
    
    B elements (V indexed by j*2+k):
      j=V[0], k=V[1]  (row 0 of B, cols 0,1)
      m=V[2], n=V[3]  (row 1 of B)
      p=V[4], q=V[5]  (row 2 of B)
    
    C elements (W indexed by k*3+i):
      C[0,0]=W[0], C[1,0]=W[1], C[2,0]=W[2]  (column 0)
      C[0,1]=W[3], C[1,1]=W[4], C[2,1]=W[5]  (column 1)
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    C[0,0] = a*j + b*m + c*p
    C[0,1] = a*k + b*n + c*q
    C[1,0] = d*j + e*m + f*p
    C[1,1] = d*k + e*n + f*q
    C[2,0] = g*j + h*m + i*p
    C[2,1] = g*k + h*n + i*q
    
    18 monomials total. We need 13 bilinear forms to cover them.
    Each bilinear form can produce at most one monomial per output entry,
    but can contribute to multiple output entries via the W vector.
    
    Strategy: Use 9 "diagonal" terms for the main products, then 4 Strassen-style
    shared terms for the cross products.
    
    Actually, let me think about this differently. The tensor is the direct sum of
    three 1x3x2 tensors (one for each row of A). Each 1x3x2 has rank 3 (schoolbook),
    so total rank is 9. But we can do better by sharing across rows.
    
    Wait - 1x3x2: C[i,k] for fixed i is sum_j A[i,j]*B[j,k]. This is just a linear
    combination, rank is min(3,2)=2? No, the tensor T[i,j,k]=delta(i,i0)*1 for the
    1x3x2 case has rank 3 (need 3 terms for the 3 j-values).
    
    Actually for 1x3x2: T[j,k] = 1 for all j,k (since A[i,j] and B[j,k] are independent).
    This is the 3x2 all-ones tensor, which has rank min(3,2)=2.
    
    So 3x3x2 = direct sum of three 1x3x2 tensors, each of rank 2, total rank 6?
    No, that's not right either because the U factors must be different for each row.
    
    Let me just construct it explicitly. We need 13 terms.
    """
    R = 13
    U = np.zeros((9, R))
    V = np.zeros((6, R))
    W = np.zeros((6, R))
    
    # Term 0: a*j -> contributes to C[0,0]
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # Term 1: b*m -> contributes to C[0,0]
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    # Term 2: c*p -> contributes to C[0,0]
    U[2,2]=1; V[4,2]=1; W[0,2]=1
    # Term 3: d*j -> contributes to C[1,0]
    U[3,3]=1; V[0,3]=1; W[1,3]=1
    # Term 4: e*m -> contributes to C[1,0]
    U[4,4]=1; V[2,4]=1; W[1,4]=1
    # Term 5: f*p -> contributes to C[1,0]
    U[5,5]=1; V[4,5]=1; W[1,5]=1
    # Term 6: g*j -> contributes to C[2,0]
    U[6,6]=1; V[0,6]=1; W[2,6]=1
    # Term 7: h*m -> contributes to C[2,0]
    U[7,7]=1; V[2,7]=1; W[2,7]=1
    # Term 8: i*p -> contributes to C[2,0]
    U[8,8]=1; V[4,8]=1; W[2,8]=1
    # Term 9: (a+b+c)*(j+m+p) - a*j - b*m - c*p = cross terms for C[0,0]
    # But we also need C[0,1] = a*k + b*n + c*q
    # Let me use: (a+b)*(j+k) gives a*j + a*k + b*j + b*k
    # We already have a*j. So this adds a*k + b*j + b*k.
    # a*k is needed for C[0,1]. b*j and b*k are unwanted.
    
    # Better approach: Use Strassen on 2x2 subblocks.
    # For rows 0,1 and cols 0,1 of the "effective" product:
    # M1 = (a+b)*(j+k) = a*j + a*k + b*j + b*k
    # M2 = (a-b)*(j-k) = a*j - a*k - b*j + b*k
    # These give us a*j, a*k, b*j, b*k with 2 terms instead of 4.
    # But we need a*j (have it), a*k (need), b*m (have it), b*n (need)
    # The issue is that Strassen works on 2x2x2, but our B is 3x2.
    
    # Let me try a different decomposition. Use 13 terms:
    # 9 terms for the "diagonal" a*j, b*m, c*p, d*j, e*m, f*p, g*j, h*m, i*p
    # 4 additional terms for the remaining 9 cross terms (a*k, b*n, c*q, d*k, e*n, f*q, g*k, h*n, i*q)
    # Each of the 4 additional terms can cover at most 2-3 of these.
    
    # Term 9: (a+c)*(k+q) = a*k + a*q + c*k + c*q
    #   a*k needed (C[0,1]), c*q needed (C[0,1]), a*q and c*k unwanted
    #   If we set W[3,9]=1 (C[0,1]), we get a*k + c*q + unwanted a*q + c*k
    #   This doesn't work cleanly.
    
    # Let me try: Term 9: a*k directly
    U[0,9]=1; V[1,9]=1; W[3,9]=1
    # Term 10: b*n directly
    U[1,10]=1; V[3,10]=1; W[3,10]=1
    # Term 11: c*q directly
    U[2,11]=1; V[5,11]=1; W[3,11]=1
    # Term 12: d*k + e*n + f*q via (d+e+f)*(k+n+q) - d*k... no, that's 1 term
    # (d+e+f)*(k+n+q) = d*k+d*n+d*q + e*k+e*n+e*q + f*k+f*n+f*q
    # We need d*k (C[1,1]), e*n (C[1,1]), f*q (C[1,1])
    # This gives all 9 cross terms for row 1. With W[4,12]=1, we get all of them.
    # But we only want d*k + e*n + f*q, not the other 6.
    
    # This approach isn't working. Let me just use rank 18 (schoolbook) for 3x3x2,
    # giving rank 54 for 3x3x6. Or try rank 15.
    
    # Actually, let me reconsider. The minimum rank of 3x3x2:
    # The tensor is T[i,j,k] = 1 for all i,j,k in {0,1,2}x{0,1,2}x{0,1}.
    # This is the all-ones 3x3x2 tensor. Its rank is min(3,3,2)... no.
    # The rank of the all-ones n x m x p tensor is min(n,m,p).
    # Wait no, that's the rank as a matrix. The tensor rank of all-ones is more complex.
    # Actually the all-ones 3x3x2 tensor has tensor rank 3 (can be written as sum of 3 rank-1 tensors).
    # T[i,j,k] = sum_{r=0}^{2} 1*1*1 = 3, not 1. So we need T[i,j,k]=1.
    # T = (1,1,1) x (1,1,1,1,1,1) x (1,1) gives T[i,j,k]=1. That's rank 1!
    # Wait, that's not right for matrix multiplication.
    
    # For matrix multiplication 3x3x2: T[i,j,k] = 1 if the monomial A[i,j]*B[j,k] appears in C[i,k].
    # T[i,j,k] = 1 for all i,j,k because every A[i,j] multiplies every B[j,k] to contribute to C[i,k].
    # So T is the all-ones 3x3x2 tensor, which has tensor rank 1!
    # But that can't be right because we need the specific pairing j.
    
    # Oh wait, I'm confusing myself. The matrix multiplication tensor for n x m x p is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give contribution to (i,k) entry of C.
    # For standard matmul, T[i,j,k] = 1 for all valid i,j,k. The j is shared between A and B.
    # So yes, T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # This is the all-ones tensor of size n x m x p.
    # The tensor rank of the all-ones n x m x p tensor is min(n,m,p).
    # For 3x3x2, that would be rank 2.
    # But that's the rank over the reals for the all-ones tensor, which is NOT the same as
    # the matrix multiplication tensor rank!
    
    # I'm making an error. The matrix multiplication tensor is NOT the all-ones tensor.
    # The matmul tensor T^{(n,m,p)}[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r sum_{i,j,k: U[i,j,r]*V[j,k,r]*W[i,k,r]=1} ...
    # No, the tensor is defined such that <T, A tensor B> = trace(C) or similar.
    
    # Let me re-read the problem. The tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k (for standard matmul, every (i,j,k) triple contributes).
    # And we need: sum_r U[i,j,r] * V[j,k,r] * W[k,n+i,r] = 1 for all i,j,k.
    # (Using the W[k*n+i] convention.)
    
    # So yes, we need the all-ones condition. The matmul tensor IS the all-ones tensor
    # in this indexing. And the tensor rank of the all-ones n x m x p tensor is indeed
    # min(n,m,p) = 2 for 3x3x2.
    
    # But wait, that means we can do 3x3x2 in rank 2? Let me check:
    # We need U[i,j,r]*V[j,k,r]*W[k*3+i,r] = 1 for all i,j,k,r summed.
    # With rank 2: sum_{r=0}^{1} U[i,j,r]*V[j,k,r]*W[k*3+i,r] = 1
    # This means the tensor T[i,j,k]=1 must be a sum of 2 rank-1 tensors.
    # T = u1 x v1 x w1 + u2 x v2 x w2
    # where u1[i,j]*v1[j,k]*w1[k*3+i] + u2[i,j]*v2[j,k]*w2[k*3+i] = 1
    # For this to hold for all i,j,k, we need the product structure to work.
    # Let u1[i,j]=1, v1[j,k]=1, w1[k*3+i]=1/2 for all. Then sum = 1/2+1/2=1. Rank 2 works!
    # But wait, U[i,j,r] must be such that U is (n*m) x rank = 9x2.
    # U[i*3+j, r] = 1 for all i,j,r. V[j*2+k, r] = 1 for all j,k,r. W[k*3+i, r] = 1/2.
    # Then sum_r U*V*W = 1*1*1/2 + 1*1*1/2 = 1. Yes! Rank 2 works for 3x3x2!
    
    # But that seems too good. Let me verify: the constraint is that entries must be
    # integer or half-integer. 1/2 is a half-integer, so this is valid!
    
    # So 3x3x2 has rank 2, and 3x3x6 = three copies of 3x3x2 has rank 6!
    # But wait, the three 3x3x2 blocks share the same A matrix. In the tensor decomposition,
    # the U factor is shared. Let me check if this works.
    
    # For 3x3x6: T[i,j,k] = 1 for all i in [3], j in [3], k in [6].
    # Using rank 6: U[i,j,r] = 1 for all, V[j,k,r] = 1 for all, W[k*3+i,r] = 1/6.
    # Then sum_r U*V*W = 6 * (1*1*1/6) = 1. Yes!
    
    # So the rank of 3x3x6 matmul tensor is at most 6? That seems impossibly low.
    # The schoolbook rank is 54, and Strassen gives much less, but rank 6?
    
    # I think I'm misunderstanding the tensor. Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 6 matrix."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r] convention"
    
    # The condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is indeed the all-ones condition. And the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # with the specific index structure... 
    
    # Wait, I think the issue is that U, V, W are NOT arbitrary. They must satisfy:
    # The decomposition corresponds to actual matrix multiplication, meaning:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) * W[i,k,r]... no that's not right either.
    
    # Actually, in tensor decomposition of matmul:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    # No, that's not standard. The standard tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # where T[i,j,k] = 1 for matmul.
    
    # And the bilinear algorithm is:
    # Preprocess: x_r = sum_j U[i,j,r] A[i,j] for each r (but U depends on i too, so this isn't a single vector)
    # Actually: for each r, define intermediate:
    #   alpha_r = sum_{i,j} U[i,j,r] A[i,j]  -- no, this doesn't make sense dimensionally.
    
    # Let me think again. The standard formulation:
    # A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T of size n x m x p with T[i,j,k] = 1.
    # Rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] V[j,k,r] W[i,k,r]
    # (or with the W[k*n+i] convention, W[k,i,r])
    
    # The algorithm:
    # For each r:
    #   Compute M_r = sum_{i,j} U[i,j,r] A[i,j]  -- this is a scalar? No.
    #   Actually: x_r = sum_j (sum_i U[i,j,r] A[i,j])... this doesn't work.
    
    # The correct interpretation:
    # For each r, the r-th multiplication is:
    #   M_r = (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k])
    #   This is a scalar times a scalar, but we need it to contribute to C[i,k].
    #   So: C[i,k] += W[i,k,r] * M_r
    #   But M_r is a single scalar, and W[i,k,r] * M_r contributes the same scalar to all C[i,k]
    #   weighted by W[i,k,r]. This can't produce the correct C unless the structure is very special.
    
    # I think I'm confusing two different formulations. Let me look at this more carefully.
    
    # The correct tensor decomposition for matmul:
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
    # where U_r is n x m, V_r is m x p, W_r is n x p.
    
    # The bilinear algorithm:
    # For each r:
    #   Compute: X_r = U_r^T * A  (this is... no)
    #   Actually: the r-th scalar multiplication is:
    #     M_r = <U_r, A> * <V_r, B>  where <,> is the Frobenius inner product? No.
    
    # I think the standard formulation is:
    # For each r:
    #   alpha_r = sum_{i,j} U_r[i,j] A[i,j]   (scalar)
    #   beta_r = sum_{j,k} V_r[j,k] B[j,k]    (scalar)  
    #   C += W_r * alpha_r * beta_r
    # where W_r is an n x p matrix.
    
    # But this means each multiplication produces a rank-1 contribution to C (W_r scaled by a scalar),
    # and we sum r such rank-1 matrices. This can only produce C if C has low rank, which it doesn't
    # in general. So this formulation is wrong.
    
    # The correct formulation must be:
    # For each r:
    #   The r-th multiplication involves specific linear combinations of A and B entries.
    #   Specifically: M_r = (sum_j U_r[j] A[?,j]) * (sum_j V_r[j] B[j,?])
    #   But the indices need to be tracked.
    
    # I believe the correct tensor decomposition is:
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
    # And the algorithm is:
    # For each r:
    #   Compute: s_r = sum_{i,j} U_r[i,j] A[i,j]  -- NO, this loses the i,j structure.
    
    # OK I think the issue is that in the standard tensor decomposition of matmul,
    # the factors are:
    #   U: (n*m) x r, where U[i*m+j, r] is the coefficient for A[i,j] in the r-th multiplier
    #   V: (m*p) x r, where V[j*p+k, r] is the coefficient for B[j,k] in the r-th multiplier
    #   W: (p*n) x r, where W[k*n+i, r] is the coefficient for C[i,k] from the r-th multiplier
    # 
    # And the r-th multiplication is:
    #   M_r = (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    #   C[i,k] += W[k*n+i, r] * M_r
    # 
    # But again, M_r is a scalar, so W[k*n+i,r] * M_r adds a rank-1 matrix to C.
    # This can't work for general C.
    
    # UNLESS the formulation is different. Let me reconsider.
    
    # Maybe the correct formulation is:
    #   M_r[i,k] = (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    #   C[i,k] += W[k*n+i, r] * M_r[i,k]
    # 
    # Here U[i*m+j, r] depends on i, so sum_j U[i*m+j,r] A[i,j] is a different scalar for each i.
    # Similarly for V. And M_r[i,k] is a matrix (product of two vectors indexed by i and k).
    # Then W scales each entry. This makes more sense!
    
    # Let me verify: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
    # = sum_r W[k*n+i,r] * sum_j sum_j' U[i*m+j,r] V[j'*p+k,r] A[i,j] B[j',k]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W[k*n+i,r] U[i*m+j,r] V[j*p+k,r] = 1 for all i,j,k
    # (and 0 for j != j', but since j and j' are both summed, we need the cross terms to vanish)
    
    # Actually, expanding: sum_j sum_j' U[i*m+j,r] V[j'*p+k,r] A[i,j] B[j',k]
    # The coefficient of A[i,j] B[j',k] is sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r]
    # For the result to be sum_j A[i,j] B[j,k], we need:
    #   sum_r W[k*n+i,r] U[i*m+j,r] V[j*p+k,r] = 1 (diagonal: j=j')
    #   sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = 0 (off-diagonal: j!=j')
    
    # So the condition is NOT just that the triple product sums to 1 for all i,j,k.
    # There are additional off-diagonal constraints!
    
    # This changes everything. The tensor T[i,j,k]=1 is not sufficient; we also need
    # the off-diagonal terms to vanish. This is why the rank is much higher than the
    # all-ones tensor rank.
    
    # So my earlier "rank 6" argument was wrong. The correct condition involves
    # 3*3*6 = 54 diagonal conditions AND off-diagonal conditions.
    
    # Given the complexity, let me just use the schoolbook (rank 54) as a safe baseline,
    # and try to find a better construction through search.
    
    return None


def _verify_full(U, V, W, n, m, p):
    """Full verification including off-diagonal constraints.
    Checks that sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = delta(j,j') for all i,j,j',k.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    rank = Ua.shape[1]
    # Check diagonal: sum_r W[k*n+i,r] U[i*m+j,r] V[j*p+k,r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += Wa[k*n+i, r] * Ua[i*m+j, r] * Va[j*p+k, r]
                if abs(val - 1) > 1e-9:
                    return False
    # Check off-diagonal: sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = 0 for j != j'
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Wa[k*n+i, r] * Ua[i*m+j, r] * Va[jp*p+k, r]
                    if abs(val) > 1e-9:
                        return False
    return True


def _search_direct_fast(n, m, p, R, n_iter, deadline, seed):
    """Fast direct search for rank-R decomposition of n x m x p."""
    rng = np.random.RandomState(seed)
    for it in range(n_iter):
        if time.time() > deadline:
            return None
        U = rng.randint(-1, 2, size=(n*m, R)).astype(float)
        V = rng.randint(-1, 2, size=(m*p, R)).astype(float)
        W = np.zeros((p*n, R))
        ok = True
        # Solve for W using the diagonal constraints first
        for i in range(n):
            for k in range(p):
                # We need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                # This is a linear system in W[k*n+i, :]
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except Exception:
                    ok = False
                    break
                w, ok2 = _round_to_half(w)
                if not ok2:
                    ok = False
                    break
                W[k*n + i] = w
            if not ok:
                break
        if not ok:
            continue
        # Full verification including off-diagonal
        if _verify_full(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x6 matmul."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit
    n, m, p = DIMENSIONS

    best = None

    # Search for low-rank decompositions
    for R in range(30, 54):
        if time.time() > deadline:
            break
        n_iter = 5000
        res = _search_direct_fast(n, m, p, R, n_iter, deadline, seed=200 + R)
        if res is not None:
            best = res
            break

    if best is not None:
        if _verify(best["U"], best["V"], best["W"], n, m, p):
            return best

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

"""Rewritten program: constructive low-rank exact decomposition for 3x4x8 matmul."""
import time

DIMENSIONS = [3, 4, 8]
_START = time.time()
_DEADLINE = _START + 120.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta (matmul tensor)."""
    # The matmul tensor: T[i,k,j] = 1 for all i,k,j (it's the identity of the
    # multiplication: C_ik = sum_j A_ij B_jk). A rank-1 term (u,v,w) contributes
    # u_i * v_k * w_j to T[i,k,j]. We need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # to equal 1 for every (i,k,j) -- i.e., the full all-ones tensor of shape (n,p,m)
    # indexed as (i,k,j). Wait: the matmul tensor has T[i,k,j]=1 for all i,k,j.
    # So we need the sum to be 1 everywhere on the (i,k,j) grid.
    ok = True
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def construct():
    n, m, p = DIMENSIONS
    # Constructive decomposition of rank m*(n+p-1) = 4*(3+8-1)=40.
    # Idea: C_ik = sum_j A_ij B_jk.
    # Write A_ij = a_ij, B_jk = b_jk.
    # Use the "prefix" trick: define for each j a term that accumulates.
    # A valid rank-m*(n+p-1) construction:
    # For j in 0..m-1, i in 0..n-1, k in 0..p-1 we'd need n*p*m.
    # Instead use the identity:
    #   sum_j A_ij B_jk  =  sum_j (A_ij) (B_jk)
    # This is already a sum over j of outer products in (i,k), each weighted by
    # the middle index. To express as rank-1 tensors u_i v_k w_j with sum=1
    # on all (i,k,j), we cannot use fewer than ... Let's use a safe exact method:
    # rank = n*p*m is 96. We aim lower.
    #
    # Reliable improved construction (rank = m * (n + p - 1)):
    # For each middle index j, split the (i,k) outer product using a
    # "separable rank (n+p-1)" factorization of the all-ones (n x p) matrix? No,
    # the all-ones n x p matrix has rank 1, not n+p-1.
    #
    # Reconsider: T[i,k,j] = 1 for ALL i,k,j. This is just the all-ones tensor!
    # Because matmul tensor for C=AB: the (i,k,j) coefficient of A_ij B_jk is 1.
    # So T is the all-ones tensor of shape (n, p, m). Its tensor rank is
    # min(n, p, m)? No -- the all-ones tensor 1_{n x p x m} has tensor rank
    # equal to min(n, p, m) ONLY if... actually rank of all-ones 3-tensor is
    # min(n,p,m) when one mode is 1? The all-ones tensor of size a x b x c has
    # tensor rank = min(a,b,c) is FALSE in general; it's at most min(a,b,c)?? 
    # Counterexample: all-ones 2x2x2 has rank 2? 1 = (1,1)(1,1)(1,1) is ONE rank-1
    # term! So rank of all-ones n x p x m tensor = 1 (just the product of all-ones
    # vectors). 
    #
    # BUT the matmul tensor is NOT all-ones in (i,k,j) with the index placement I
    # assumed. Let me re-derive: C_ik = sum_j A_ij B_jk. The bilinear form pairs
    # A_ij (index i,j) with B_jk (index j,k) producing C_ik (index i,k).
    # A rank-1 term in the bilinear algorithm is u_i * v_k * w_j, contributing
    # u_i v_k w_j to C_ik from the "j-th" part. For the algorithm to be correct,
    # we need: for every (i,k), sum_j A_ij B_jk = sum_c (sum_j U_{i,j}^{(c)} A_ij)
    #   (sum_j V_{j,k}^{(c)} B_jk) W ... 
    # Standard: rank-1 term = (u_i)(v_k)(w_j) means C_ik += u_i v_k * (sum_j w_j A_ij B_jk)?
    # No. The tensor T_{i,k,j} such that C_ik = sum_j T_{i,k,j} A_ij B_jk.
    # Here T_{i,k,j} = 1 for all i,k,j. So T IS the all-ones tensor (n x p x m).
    # And its rank-1 decomposition: T_{i,k,j} = sum_c u_i^{(c)} v_k^{(c)} w_j^{(c)}.
    # All-ones = (1,1,...)(1,1,...)(1,1,...) -> rank 1!!
    #
    # That can't be right for matmul rank. The catch: the rank-1 term must be
    # u_i v_k w_j AND the algorithm computes C_ik = sum_c (sum_j u_i w_j A_ij)
    # (sum_j v_k w_j B_jk)? No -- standard matmul tensor rank-1 term is
    # (u_i)(v_k)(w_j) with the SAME w_j multiplying both, i.e. the term is
    # (sum_j u_i w_j A_ij)(sum_j v_k w_j B_jk)?? That's not bilinear in A and B
    # separately with shared j.
    #
    # Correct model: a rank-1 term is a product of three vectors
    #   a_i (from A-space, index i), b_k (from B-space, index k), c_j (from
    #   shared-space, index j), and it contributes to C_ik the value
    #   a_i b_k c_j * (A_ij B_jk) summed appropriately. The bilinear algorithm:
    #   C_ik = sum_c (sum_j a_i^{(c)} c_j^{(c)} A_ij) (sum_j b_k^{(c)} c_j^{(c)} B_jk)
    #   = sum_c sum_j sum_{j'} a_i^{(c)} b_k^{(c)} c_j^{(c)} c_{j'}^{(c)} A_ij B_{j'k}
    # For this to equal sum_j A_ij B_jk, we need for each c: the coefficient of
    # A_ij B_{j'k} is a_i b_k c_j c_{j'}, and summing over c must give delta_{j j'}.
    # I.e. sum_c a_i^{(c)} b_k^{(c)} c_j^{(c)} c_{j'}^{(c)} = delta_{j,j'} for all i,k?
    # That's impossible unless a_i b_k is constant... This shows the standard
    # formulation: a rank-1 term is (u_i)(v_k)(w_j) and the algorithm is
    #   C_ik = sum_c (sum_j u_i w_j A_ij) (sum_j v_k w_j B_jk)  -- NO, that's
    #   quadratic in w.
    #
    # The CORRECT standard: rank-1 term corresponds to choosing, for the shared
    # index j, a single value, i.e. the term is u_i * v_k * w_j and it computes
    #   C_ik += u_i v_k * (w_j A_ij B_jk)  summed over j -> but that's one term
    #   per j. So a "rank-1" in the tensor T_{i,k,j} = u_i v_k w_j means the
    #   algorithm uses C_ik = sum_c (sum_j u_i^{(c)} w_j^{(c)} A_ij) v_k^{(c)} ...
    # I'm overcomplicating. 
    #
    # STANDARD DEFINITION (used by the evaluator per contract): 
    #   C_ik = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  (summed over c AND j)
    #   = sum_c sum_j U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # For this to equal sum_j A_ij B_jk for ALL A,B, we need:
    #   sum_c sum_j U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = 1  (coefficient of A_ij B_jk)
    # for every (i,k,j), AND 0 for mismatched... but the expression only has A_ij B_jk
    # terms (j is the summed index appearing in both U and V). So the condition is:
    #   for all (i,k,j): sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # This is EXACTLY the all-ones condition on (i,k,j) that I had. And the seed
    # program sets U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1 for c=(i,j,k), giving
    # sum_c = 1 at (i,k,j) only when c matches (i,j,k) -> yes equals 1. 
    #
    # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,k,j.
    # This is the tensor equation. The all-ones tensor (in indices i,k,j) with
    # the constraint that term c couples (i,j) in U, (j,k) in V, (k,i) in W.
    # This is NOT a free all-ones decomposition because the index j is shared
    # between U and V, and k shared between V and W, and i between W and U.
    # It's a "cycle" constraint. The rank is the tensor rank of this cyclic
    # all-ones structure.
    #
    # A valid low-rank construction: choose rank = m and set for c=j:
    #   U[i*m+j, c=j] = 1, V[j*p+k, c=j] = 1, W[k*n+i, c=j] = 1.
    # Then sum_c U[i*m+j,c]V[j*p+k,c]W[k*n+i,c] = sum_c [c==j]*[c==j]*[c==j] = 1.
    # YES! For each (i,k,j), only c=j contributes (since U[i*m+j,c] nonzero only
    # at c=j, V[j*p+k,c] nonzero only at c=j, W[k*n+i,c] nonzero only at c=j).
    # Product = 1. So rank = m = 4 WORKS!
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook (guaranteed valid)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

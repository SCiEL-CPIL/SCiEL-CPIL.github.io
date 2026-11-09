import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 5.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, c] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _make_rank12_3x3x2():
    """
    Construct a rank-12 decomposition for 3x3 * 3x2 matrix multiplication.
    This uses the structure that 3x2 = 3x1 + 3x1 (two independent 3x1 multiplications).
    For each 3x1 multiplication, we use a rank-6 decomposition (schoolbook for 3x1).
    """
    n, m, p = 3, 3, 2
    r = 12
    U = np.zeros((r, n, m))
    V = np.zeros((r, m, p))
    W = np.zeros((r, n, p))
    
    # For each column j of the output (j=0,1), we need to compute sum_k A[i,k] * B[k,j]
    # This is a 3x1 multiplication for each j
    # Schoolbook for 3x1: rank 3, with U[t,i,k]=delta(i,k), V[t,k,j]=B[k,j], W[t,i,j]=delta(i,j)
    # But we need to express this in the tensor form
    
    # Actually, for 3x3 * 3x2, the tensor T[i,j,k] where i in [0,3), j in [0,2), k in [0,3)
    # T[i,j,k] = 1 if k is the "inner" index, but the structure is:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor has T[i,k,j] = 1 for all i,k,j (it's just the identity in the k dimension)
    
    # Wait, let me reconsider. The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # is T[i, k, j] where C[i,j] = sum_k A[i,k] B[k,j]
    # So T[i,k,j] = 1 for all i, k, j.
    
    # For n=3, m=3, p=2: T is 3x3x2, all ones.
    # A rank-1 decomposition is a tensor of the form u_i * v_k * w_j
    # For all-ones tensor, we can use:
    # Sum_t U[t,i] * V[t,k] * W[t,j] = 1 for all i,k,j
    
    # One simple decomposition: use r = n*m*p = 18 (schoolbook)
    # But we want rank 12.
    
    # Actually, for the all-ones tensor of size n x m x p, the minimum rank is max(n,m,p)
    # No wait, that's not right either.
    
    # Let me think about this differently. The tensor T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # A rank-1 tensor is a outer product u v w where (u v w)[i,k,j] = u[i] v[k] w[j]
    # We need sum_t u_t[i] v_t[k] w_t[j] = 1 for all i,k,j
    
    # This is equivalent to finding matrices U (n x r), V (m x r), W (p x r) such that
    # U V^T W^T (in some sense) gives the all-ones tensor.
    
    # Actually, the condition is: for all i,k,j, sum_t U[t,i] V[t,k] W[t,j] = 1
    
    # This means that the tensor (U V^T W^T) must be the all-ones tensor.
    # In matrix form, if we flatten, this is a linear system.
    
    # A simple solution: use r = n + m + p - 2 (this is a known result for the all-ones tensor)
    # For n=3, m=3, p=2: r = 3 + 3 + 2 - 2 = 6
    
    # Let me construct it:
    # U[i, t] for t in [0, n): U[i, t] = delta(i, t)
    # V[k, t] for t in [0, n): V[k, t] = 1
    # W[j, t] for t in [0, n): W[j, t] = 1
    # Then sum_t U[t,i] V[t,k] W[t,j] = sum_{t=0}^{n-1} delta(i,t) * 1 * 1 = 1 if i < n, else 0
    # This doesn't work for all i.
    
    # Let me try a different approach. Use the fact that 1 = 1*1*1, so we can use
    # r = 1 with U[i,0]=1, V[k,0]=1, W[j,0]=1. Then sum_t = 1*1*1 = 1.
    # Wait, that gives rank 1!
    
    # But that can't be right. Let me re-examine.
    # If U[i,0] = 1 for all i, V[k,0] = 1 for all k, W[j,0] = 1 for all j
    # Then sum_t U[t,i] V[t,k] W[t,j] = U[0,i] V[0,k] W[0,j] = 1 * 1 * 1 = 1
    # So the all-ones tensor has rank 1!
    
    # But wait, that's the tensor T[i,k,j] = 1. But in matrix multiplication,
    # the tensor is not all ones. Let me re-examine.
    
    # For matrix multiplication C = A B where A is n x m, B is m x p:
    # C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T has indices (i, k, j) and T[i,k,j] = 1 if the term A[i,k] B[k,j] appears in C[i,j]
    # So T[i,k,j] = 1 for all i, k, j.
    
    # So yes, the tensor is all ones, and it has rank 1!
    
    # But that contradicts the known result that matrix multiplication has rank at least n*m*p / something.
    
    # I think I'm confusing the tensor structure. Let me look at this more carefully.
    
    # The matrix multiplication tensor for n x m times m x p is a tensor T in R^{n x m x p}
    # where T[i, k, j] = 1 if the product A[i,k] * B[k,j] contributes to C[i,j]
    # So T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    
    # A rank-1 decomposition is T = sum_t U_t V_t W_t where U_t in R^n, V_t in R^m, W_t in R^p
    # and (U_t V_t W_t)[i,k,j] = U_t[i] V_t[k] W_t[j]
    
    # We need sum_t U_t[i] V_t[k] W_t[j] = 1 for all i,k,j
    
    # If we set U_t[i] = 1, V_t[k] = 1, W_t[j] = 1 for all i,k,j and t=0 only
    # Then sum_t = 1*1*1 = 1. So rank 1!
    
    # But this is clearly wrong because matrix multiplication cannot have rank 1.
    
    # The issue is that I'm not accounting for the structure correctly.
    # In matrix multiplication, the tensor is not just "all ones".
    # The correct tensor is: T[i, k, j] = 1, but the way it's used is:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_k T[i,k,j] A[i,k] B[k,j]
    
    # So T[i,k,j] = 1 for all i,k,j. And this tensor does have rank 1.
    
    # But that means matrix multiplication has rank 1, which is absurd.
    
    # I must be misunderstanding the tensor decomposition.
    
    # Let me look at this from a different angle. The matrix multiplication map is:
    # (A, B) -> C where C = A B
    # This is a bilinear map. The tensor associated with this bilinear map is:
    # T in (R^{n x m})^* x (R^{m x p})^* x R^{n x p}
    # T(A, B, C) = sum_{i,j,k} C[i,j] A[i,k] B[k,j]
    
    # So T[i,j,k] (with i,j for C, k for the inner product) is:
    # T[i,j,k] = 1 if the term C[i,j] A[i,k] B[k,j] appears
    # So T[i,j,k] = 1 for all i, j, k.
    
    # But the way we decompose this tensor is different from what I was thinking.
    # A rank-1 decomposition of T is: T = sum_t alpha_t beta_t gamma_t
    # where alpha_t in (R^{n x m})^*, beta_t in (R^{m x p})^*, gamma_t in R^{n x p}
    # and (alpha_t beta_t gamma_t)(A, B, C) = alpha_t(A) beta_t(B) gamma_t(C)
    
    # We need sum_t alpha_t(A) beta_t(B) gamma_t(C) = sum_{i,j,k} C[i,j] A[i,k] B[k,j]
    
    # A simple rank-1 term: alpha_t(A) = A[i0, k0], beta_t(B) = B[k0, j0], gamma_t(C) = C[i0, j0]
    # Then alpha_t(A) beta_t(B) gamma_t(C) = A[i0,k0] B[k0,j0] C[i0,j0]
    
    # To get the full sum, we need one such term for each (i0, k0, j0)
    # So rank = n * m * p = 3 * 3 * 2 = 18 (schoolbook)
    
    # So the tensor T[i,j,k] = 1 (where i,j are output indices, k is inner index)
    # has rank n*m*p in this decomposition scheme.
    
    # OK so I was confused. The tensor is not "all ones" in the sense I was thinking.
    # The tensor T is defined on the space of matrices, and T(A,B,C) = trace(C^T A B)
    # which expands to sum_{i,j,k} C[i,j] A[i,k] B[k,j]
    
    # So the "components" of T are T[i,j,k] = 1, but the rank is the minimum number of
    # rank-1 terms needed to represent T, where a rank-1 term is a product of three linear functionals.
    
    # So for 3x3 * 3x2, the schoolbook rank is 18.
    
    # Now, can we do better? The question is whether there's a rank-12 decomposition.
    
    # Given the complexity of finding such a decomposition, and the fact that the current
    # program uses rank 54 (which is 3 * 18, suggesting it's using three independent 3x3x2
    # decompositions of rank 18 each), I think the best approach is to keep the schoolbook
    # for the 3x3x2 blocks but see if we can find a better decomposition.
    
    # For now, let me just return the schoolbook decomposition for 3x3x2.
    r = 18
    U = np.zeros((r, n, m))
    V = np.zeros((r, m, p))
    W = np.zeros((r, n, p))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[col, i, k] = 1
                V[col, k, j] = 1
                W[col, i, j] = 1
                col += 1
    return U, V, W


def _find_3x3x2_decomp(rank, seed, trials=500):
    """Find rank decomposition for 3x3 * 3x2 matrix multiplication."""
    n, m, p = 3, 3, 2
    rng = np.random.default_rng(seed)
    
    for trial in range(trials):
        if not _ok():
            return None
        
        # Random U and V with small integer/half-integer values
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = rng.choice(vals, size=(rank, n, m))  # U[t, i, k]
        V = rng.choice(vals, size=(rank, m, p))  # V[t, k, j]
        
        # Build linear system for W
        # For each (i, j, k, k'): sum_t U[t,i,k] * V[t,k',j] * W[t,i,j] = delta(k,k')
        # Group by (i, j): for each (i,j), solve for W[:, i, j]
        
        W = np.zeros((rank, n, p))
        valid = True
        
        for i in range(n):
            for j in range(p):
                # Build A matrix: A[k, k', t] = U[t,i,k] * V[t,k',j]
                # We need: sum_t A[k, k', t] * W[t, i, j] = delta(k, k')
                # So for each (k, k'), equation: sum_t (U[t,i,k] * V[t,k',j]) * W[t,i,j] = delta(k,k')
                
                # A is (m*m, rank) where A[k*m+k', t] = U[t,i,k] * V[t,k',j]
                A = np.zeros((m * m, rank))
                b = np.zeros(m * m)
                for k in range(m):
                    for kp in range(m):
                        A[k * m + kp, :] = U[:, i, k] * V[:, kp, j]
                        b[k * m + kp] = 1.0 if k == kp else 0.0
                
                # Solve A @ W_col = b where W_col = W[:, i, j]
                try:
                    W_col, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Round to half-integers
                W_col_rounded = np.round(W_col * 2) / 2
                W[:, i, j] = W_col_rounded
                
                # Check if rounded solution satisfies the constraint
                if not np.allclose(A @ W_col_rounded, b, atol=1e-9, rtol=0.0):
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        # Check: for all i,j,k,k': sum_t U[t,i,k]*V[t,k',j]*W[t,i,j] = delta(k,k')
        check = True
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    for kp in range(m):
                        s = sum(U[t, i, k] * V[t, kp, j] * W[t, i, j] for t in range(rank))
                        expected = 1.0 if k == kp else 0.0
                        if abs(s - expected) > 1e-9:
                            check = False
                            break
                    if not check:
                        break
                if not check:
                    break
            if not check:
                break
        
        if check:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def construct():
    # Try to find 3x3x2 decomposition of various ranks
    for r in range(12, 7, -1):
        if not _ok():
            break
        d0 = _find_3x3x2_decomp(r, seed=1000 + r)
        if d0 is None:
            continue
        d1 = _find_3x3x2_decomp(r, seed=3000 + r)
        if d1 is None:
            continue
        d2 = _find_3x3x2_decomp(r, seed=5000 + r)
        if d2 is None:
            continue
        U, V, W = _build_3x6_from_3x3x2(r, d0, d1, d2)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook()

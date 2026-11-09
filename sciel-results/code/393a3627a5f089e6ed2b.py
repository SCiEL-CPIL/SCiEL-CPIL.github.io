"""
Rank-minimal tensor decomposition for 3x3 by 3x6 matrix multiplication.

Strategy:
1. Start with the guaranteed schoolbook rank-54 decomposition.
2. Attempt a structured rank-reduction search:
   - Use the Kronecker product structure: 3x3x6 = 3x3x3 ⊗ 1x1x2 (conceptually).
   - Try to find a rank-r decomposition for 3x3x6 directly by solving the
     polynomial system via random initialization + Newton-style refinement.
   - Use the fact that the tensor has symmetries to reduce search space.
3. If no lower-rank solution is found within time budget, return schoolbook.

The key insight from the inspirations: use numpy for fast tensor contractions
and a verification loop. The current program's structure (hardcoded schoolbook)
is safe but suboptimal. We add a search layer on top.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition as (U, V, W) numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, rng, atol=1e-9):
    """Verify decomposition by testing on random matrices."""
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C_actual = np.zeros((n, p))
    for c in range(U.shape[1]):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # W[k*n+i] => wc reshaped as (p, n), transpose to (n, p)
        C_actual += (uc @ vc) * wc.T
    if not np.allclose(C_actual, C_expected, atol=atol):
        return False
    # Second test
    A2 = rng.randn(n, m)
    B2 = rng.randn(m, p)
    C2_expected = A2 @ B2
    C2_actual = np.zeros((n, p))
    for c in range(U.shape[1]):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        C2_actual += (uc @ vc) * wc.T
    if not np.allclose(C2_actual, C2_expected, atol=atol):
        return False
    return True


def _verify_exact(U, V, W, n, m, p, atol=1e-10):
    """Exact tensor entry verification: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > atol:
                    return False
    return True


def _try_rank_reduction(n, m, p, max_rank, deadline, seed=42):
    """
    Attempt to find a decomposition with rank < max_rank.
    Uses random initialization + iterative refinement (Gauss-Newton style).
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    best = None
    best_rank = max_rank

    # Try ranks from max_rank-1 down
    for target_rank in range(max_rank - 1, max(10, max_rank - 10), -1):
        if time.time() > deadline:
            break

        for attempt in range(30):
            if time.time() > deadline:
                break

            # Random initialization
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)

            # Normalize to avoid scale issues
            for c in range(target_rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                    U[:, c] /= scale
                    V[:, c] /= scale
                    W[:, c] /= scale

            # Iterative refinement: minimize ||T_decomp - T_true||_F
            # T_true[i,j,k] = 1 for all i,j,k
            # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            #
            # Use gradient descent on the residual.
            # Residual R[i,j,k] = T_decomp[i,j,k] - 1
            # Gradient w.r.t. U[i*m+j,c] = sum_{j',k'} V[j'*p+k',c] * W[k'*n+i,c] * R[i,j',k']
            # This is expensive. Instead, use a simpler approach:
            # Alternate minimization (like ALS for tensor decomposition).

            success = False
            for iteration in range(500):
                if time.time() > deadline:
                    break

                # Check current quality
                if iteration % 50 == 0:
                    if _verify(U, V, W, n, m, p, rng, atol=1e-7):
                        success = True
                        break

                # Alternate minimization:
                # Fix V, W, solve for U
                # Fix U, W, solve for V
                # Fix U, V, solve for W

                # For each column c, the contribution is outer(u_c, v_c, w_c)
                # We want sum_c outer(u_c, v_c, w_c) = ones(n, m, p)

                # Fix V and W, update U:
                # For each (i, j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is a linear system in U[i*m+j, :] for each (i, j, k) but coupled across k.
                # Simpler: use gradient step.

                # Compute residual
                R = np.zeros((n, m, p))
                for c in range(target_rank):
                    uc = U[:, c].reshape(n, m)
                    vc = V[:, c].reshape(m, p)
                    wc = W[:, c].reshape(p, n)
                    # R[i,j,k] += uc[i,j] * vc[j,k] * wc.T[i,k]
                    R += np.einsum('ij,jk,ik->ijk', uc, vc, wc.T)
                R -= 1.0

                # Gradient step with small learning rate
                lr = 0.01 / (1 + iteration * 0.01)
                for c in range(target_rank):
                    uc = U[:, c].reshape(n, m)
                    vc = V[:, c].reshape(m, p)
                    wc = W[:, c].reshape(p, n)

                    # dR/dU[i*m+j,c] = sum_{k} vc[j,k] * wc.T[i,k] * R[i,j,k]
                    grad_u = np.einsum('jk,ik,ijk->ij', vc, wc.T, R)
                    grad_v = np.einsum('ij,ik,ijk->jk', uc, wc.T, R)
                    grad_w = np.einsum('ij,jk,ijk->ik', uc, vc, R)

                    U[:, c] -= lr * grad_u.ravel()
                    V[:, c] -= lr * grad_v.ravel()
                    W[:, c] -= lr * grad_w.ravel()

                if success:
                    break

            if success:
                if _verify(U, V, W, n, m, p, rng, atol=1e-9):
                    best = (U, V, W)
                    best_rank = target_rank
                    break
                # Try with tighter verification
                elif _verify(U, V, W, n, m, p, rng, atol=1e-8):
                    best = (U, V, W)
                    best_rank = target_rank
                    break

        if best is not None and best_rank < max_rank - 1:
            break

    return best


def _try_kronecker_embedding(n, m, p, deadline, seed=123):
    """
    Try to exploit the structure: 3x3x6 can be seen as 3x3x3 with B having 6 columns.
    If we have a rank-r decomposition for 3x3x3, we get rank 2r for 3x3x6.
    But we can potentially do better by sharing factors.
    
    Try: assume the decomposition has the form where the p-dimension is split
    into two groups of 3, and the factors for the two groups are related.
    """
    rng = np.random.RandomState(seed)
    n, m, p = 3, 3, 6
    p_half = 3

    # Start from schoolbook rank 54 and try to reduce
    U, V, W = _build_schoolbook(n, m, p)
    rank = U.shape[1]

    # Try to find linear dependencies among the columns
    # The tensor T has rank at most 54. We want to find if there's a lower rank decomposition.
    # Use the method of "tensor rank via linear algebra":
    # Flatten the tensor and look for low-rank structure.

    # Alternative: use the fact that for 3x3x6, the multiplication tensor
    # T(i,j,k) = 1 for all i in [0,3), j in [0,3), k in [0,6)
    # This is just the all-ones tensor! Wait, no.
    # The multiplication tensor for C = A*B where A is 3x3, B is 3x6:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_{i,j,k} C[i,k] * (sum_j A[i,j]*B[j,k])
    # The tensor T has entries T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # So T is indeed the all-ones tensor of shape (n, m, p) = (3, 3, 6).
    #
    # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) if we allow
    # arbitrary rank-1 tensors? No, that's for the border rank or for specific structures.
    # Actually, the all-ones tensor 1_{n,m,p} has rank max(n,m,p) because:
    # 1_{n,m,p} = sum_{r=1}^{max(n,m,p)} u_r v_r w_r where we can set
    # u_r[i] = 1 if i < r else 0 (or similar), but this doesn't work directly.
    #
    # Actually, the all-ones tensor has rank max(n, m, p). Proof:
    # 1_{n,m,p}(i,j,k) = 1 = sum_{r=1}^{max(n,m,p)} (1/max) * 1 * 1 * 1? No.
    # 
    # Let me think again. The tensor T(i,j,k) = 1 for all i,j,k.
    # A rank-1 tensor is u(i) * v(j) * w(k).
    # We need sum_c u_c(i) * v_c(j) * w_c(k) = 1 for all i,j,k.
    # This means the matrix [sum_c u_c(i)*v_c(j)*w_c(k)] = 1 for all (i,j,k).
    # Fix k: sum_c u_c(i)*v_c(j)*w_c(k) = 1 for all i,j.
    # Let M_k[i,j] = sum_c u_c(i)*v_c(j)*w_c(k).
    # M_k = U * W_k * V^T where W_k is a diagonal matrix with entries w_c(k).
    # We need M_k = 1 (all-ones matrix) for all k.
    # So U * diag(w(0)) * V^T = 1, U * diag(w(1)) * V^T = 1, etc.
    # This means U * diag(w(k)) * V^T = 1 for all k.
    # So diag(w(k)) = U^{-1} * 1 * (V^T)^{-1} for all k, which means w(k) is constant.
    # So we need rank(U * V^T) >= rank(1) = 1, and we need U * c * V^T = 1 where c = w(k) (constant).
    # So U * V^T = (1/c) * 1.
    # The rank of U * V^T is at most min(rank(U), rank(V^T)) = min(rank(U), rank(V)).
    # We need U * V^T to be a scalar multiple of the all-ones matrix, which has rank 1.
    # So we need rank(U) >= 1 and rank(V) >= 1, and U * V^T proportional to 1.
    # This is possible with rank 1: U = [1,1,1]^T * a, V = [1,1,1]^T * b, then U*V^T = ab * 1.
    # So w(k) = 1/(ab) for all k.
    # Thus the all-ones tensor has rank 1?!
    #
    # Wait, that can't be right. Let me re-examine.
    # The multiplication tensor is NOT the all-ones tensor in the sense I was thinking.
    # 
    # For matrix multiplication C = A*B:
    # The bilinear form is sum_{i,k} C[i,k] * (sum_j A[i,j] * B[j,k])
    # = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k]
    # contributes to the bilinear form with coefficient 1 when paired with C[i,k]).
    #
    # But the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # This IS the all-ones tensor of shape (n*m, m*p, p*n) but with the specific index mapping.
    # Wait no, it's a tensor of shape (n, m, p) in the (i,j,k) indices.
    # T(i,j,k) = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # So yes, it IS the all-ones tensor. And I showed it has rank 1?
    # Let me re-check.
    #
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is NOT the same as the all-ones tensor in the naive (i,j,k) sense because
    # the indices of U, V, W are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    # The tensor is T(i,j,k) = 1, but the factorization couples the indices cyclically.
    #
    # So it's a tensor in the space (R^{n*m}) x (R^{m*p}) x (R^{p*n}) with the
    # specific index structure. This is the standard matrix multiplication tensor.
    #
    # For 3x3x3, the rank is known to be 23 (Lader).
    # For 3x3x6, the rank is at most 2*23 = 46 (by splitting B into two 3x3 blocks).
    # The lower bound is max(n*p, m*p, n*m) = max(18, 18, 9) = 18.
    #
    # So we're looking for a rank between 18 and 46.
    #
    # Given the time constraint, let's try a few structured approaches:
    # 1. Embed two rank-23 decompositions (gives rank 46).
    # 2. Try random search for lower rank.
    #
    # Since I don't have a verified rank-23 decomposition hardcoded, let me try
    # to find one via search, or use a known construction.

    # For now, return None and let the main function decide.
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin

    # Schoolbook fallback
    U, V, W = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Try to find a lower-rank decomposition
    try:
        result = _try_rank_reduction(n, m, p, best_rank, deadline, seed=42)
        if result is not None:
            U2, V2, W2 = result
            if U2.shape[1] < best_rank:
                # Verify with exact check
                if _verify_exact(U2, V2, W2, n, m, p, atol=1e-8):
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = U2.shape[1]
    except Exception:
        pass

    # If still at schoolbook rank, try another search strategy
    if best_rank == n * m * p and time.time() < deadline - 10:
        try:
            # Try Kronecker-based approach
            result = _try_kronecker_embedding(n, m, p, deadline, seed=123)
            if result is not None:
                U2, V2, W2 = result
                if U2.shape[1] < best_rank:
                    if _verify_exact(U2, V2, W2, n, m, p, atol=1e-8):
                        best_U, best_V, best_W = U2, V2, W2
                        best_rank = U2.shape[1]
        except Exception:
            pass

    # Final verification of the best solution
    rng_final = np.random.RandomState(999)
    if not _verify(best_U, best_V, best_W, n, m, p, rng_final, atol=1e-8):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

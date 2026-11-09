"""
Matrix multiplication 3,3,7 - rank-minimal tensor decomposition.
Uses structured algebraic constructions to find a low-rank decomposition.
"""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np
from itertools import product, combinations, combinations_with_replacement

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin

    def schoolbook():
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape to 3D arrays
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)
        return np.all(np.abs(T - 1.0) <= tol)

    # Strategy 1: Try to find a decomposition using the structure of small matrix multiplication.
    # For 3x3 x 3x3, the rank is known to be 23 (Landsberg). For 3x3 x 3x7, we can try
    # to extend the 3x3x3 decomposition or use other algebraic methods.

    # Strategy 2: Use a bilinear form approach. The matrix multiplication tensor can be seen
    # as a linear map. We look for a decomposition of the form:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}

    # Strategy 3: Use the fact that for fixed i, the multiplication by A[i,:] (row i) gives
    # a linear map from R^{m*p} to R^p. We can try to decompose these maps.

    # Let's try a systematic search with structured patterns.
    # Key insight: For each (i,k) pair in the output, we need sum_j A[i,j]*B[j,k] = C[i,k].
    # This is a sum over j=0..m-1. The tensor has 1's at all (i,j,k) positions.

    # Approach: Use a decomposition based on the Strassen algorithm extended to 3x3.
    # For 2x2, Strassen uses 7 multiplications. For 3x3, the best known is 23.
    # For 3x3 x 3x7, we have more columns in B, so we might be able to share more.

    # Let's try to construct a decomposition using the following idea:
    # Group the k indices and use shared linear combinations.

    # Method: Random search with structured initialization
    # Start from schoolbook and try to merge terms.

    def try_merge_schoolbook():
        """Try to merge schoolbook terms to reduce rank."""
        if time.time() > deadline - 10:
            return None

        # Start with schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

        # Try to find linear dependencies among the columns
        # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if their u, v, w vectors
        # are linearly dependent in a way that allows a rank-1 combination.
        # This is complex, so let's try a different approach.
        return None

    # Method: Use the "slice rank" or "border rank" concepts
    # For our purposes, let's try a direct algebraic construction.

    # Key idea: The matrix multiplication tensor for (n,m,p) has a natural structure.
    # For each i in [0,n), the slice T[i,:,:] is the tensor for multiplying a 1xm row
    # by an mxp matrix, which has rank m*p (schoolbook) or potentially less.

    # Actually, let's try a more practical approach: use the known rank for 3x3x3 (23)
    # and extend it to 3x3x7.

    # For 3x3x3, one known rank-23 decomposition uses the following structure:
    # It's based on the work of Landsberg. Let me try to implement a version.

    # Alternative: Use a computer algebra approach with integer/rational entries.
    # Let's try to find a decomposition with small integer entries.

    # Method: Use the "bilinear complexity" approach.
    # The tensor T can be viewed as a 3-tensor. We want to find the minimum R such that
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.

    # For small cases, we can use Gröbner bases or other algebraic methods, but those
    # are too slow. Instead, let's use a heuristic search.

    # Heuristic: Start with a random rank-R decomposition and try to optimize it.
    # The objective is to minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F.

    def random_search(rank, num_trials=50, seed=42):
        """Try random initializations and optimize."""
        if time.time() > deadline - 15:
            return None

        np.random.seed(seed)
        best_obj = np.inf
        best_result = None

        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break

            # Initialize randomly
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            # Normalize columns to prevent divergence
            for r in range(rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                    # Keep the product of norms = 1
                    scale = (nU * nV * nW) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale

            # Optimize using gradient descent
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 15:
                    break

                # Compute T_approx = sum_r u_r ⊗ v_r ⊗ w_r
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                T_approx = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)

                # Objective: ||T_approx - 1||_F^2
                obj = np.sum((T_approx - 1.0) ** 2)

                if obj < best_obj:
                    best_obj = obj
                    if obj < 1e-10:
                        # Check exactness
                        if verify_fast(U, V, W):
                            best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                            return best_result

                # Compute gradients
                # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r] (summed over k)
                # More precisely: d/dU[i*m+j,r] sum_{i',j',k'} (T_approx[i',j',k'] - 1)^2
                # = 2 * (T_approx[i,j,k'] - 1) * V[j,k',r] * W[k',i,r] summed over k'

                # Use einsum for gradients
                dU = np.einsum('ijk,jkr,kr->ijr', T_approx - 1.0, V3, W3) * 2
                dV = np.einsum('ijk,ijr,kr->jkr', T_approx - 1.0, U3, W3) * 2
                dW = np.einsum('ijk,ijr,jkr->kr', T_approx - 1.0, U3, V3) * 2

                # Gradient step
                U -= lr * dU.reshape(n*m, rank)
                V -= lr * dV.reshape(m*p, rank)
                W -= lr * dW.reshape(p*n, rank)

                # Renormalize
                for r in range(rank):
                    nU = np.linalg.norm(U[:, r])
                    nV = np.linalg.norm(V[:, r])
                    nW = np.linalg.norm(W[:, r])
                    if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                        scale = (nU * nV * nW) ** (1/3)
                        U[:, r] /= scale
                        V[:, r] /= scale
                        W[:, r] /= scale

        if best_result is not None:
            return best_result

        # Return the best we found even if not exact, for potential further optimization
        if best_obj < 1e-6:
            # Try to polish
            pass

        return None

    # Method: Use a structured decomposition based on the algebraic structure.
    # For 3x3 x 3x7, let's try to use the following idea:
    # The multiplication can be decomposed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each i, this is a linear combination of the rows of B.
    # We can precompute linear combinations of A's rows and B's columns.

    # Let's try a "block" approach: divide the 7 columns of B into blocks.
    # For example, 7 = 3 + 3 + 1.

    # Actually, let's try a more systematic approach using the known rank bounds.
    # The rank of the 3x3x3 matrix multiplication tensor is 23.
    # For 3x3x7, a trivial upper bound is 3 * 23 = 69 (worse than schoolbook 63).
    # But we can do better by sharing across the k dimension.

    # Key insight: For fixed i and j, the term A[i,j] * B[j,k] appears for all k.
    # So we can factor out A[i,j] and have a single "multiplication" by the scalar A[i,j].
    # This doesn't directly help with rank reduction.

    # Let's try a different approach: use the "tensor rank" of the 3x3x3 case
    # and extend it.

    # For now, let's try several random searches with different ranks and seeds.
    # The lower bound for the rank is max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
    # A better lower bound comes from the "slice rank" or other methods.

    # Let's try ranks from 21 to 62 (just below schoolbook).
    best_result = schoolbook()
    best_rank = n * m * p

    # Try a few promising ranks
    candidate_ranks = [23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55, 60, 61, 62]

    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            break

        # Try multiple seeds
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 20:
                break

            result = random_search(rank, num_trials=10, seed=seed)
            if result is not None:
                best_result = result
                best_rank = rank
                break

        if best_rank < n * m * p:
            break

    # If we found a better result, try to polish it
    if best_rank < n * m * p:
        # Try to verify and potentially improve
        U = np.array(best_result['U'], dtype=float)
        V = np.array(best_result['V'], dtype=float)
        W = np.array(best_result['W'], dtype=float)

        # Try a few more gradient steps with smaller learning rate
        lr = 0.001
        for it in range(100):
            if time.time() > deadline - 10:
                break

            U3 = U.reshape(n, m, best_rank)
            V3 = V.reshape(m, p, best_rank)
            W3 = W.reshape(p, n, best_rank)
            T_approx = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)
            obj = np.sum((T_approx - 1.0) ** 2)

            if obj < 1e-12:
                break

            dU = np.einsum('ijk,jkr,kr->ijr', T_approx - 1.0, V3, W3) * 2
            dV = np.einsum('ijk,ijr,kr->jkr', T_approx - 1.0, U3, W3) * 2
            dW = np.einsum('ijk,ijr,jkr->kr', T_approx - 1.0, U3, V3) * 2

            U -= lr * dU.reshape(n*m, best_rank)
            V -= lr * dV.reshape(m*p, best_rank)
            W -= lr * dW.reshape(p*n, best_rank)

            # Renormalize
            for r in range(best_rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                    scale = (nU * nV * nW) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale

        # Final check
        if verify_fast(U, V, W):
            best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return best_result
# EVOLVE-BLOCK-END

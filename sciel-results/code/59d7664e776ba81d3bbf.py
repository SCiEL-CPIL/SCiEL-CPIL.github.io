import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Build the matmul tensor T[i,j,k] = delta(i,j,k) meaning
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C.
    # The tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    # The matmul tensor M has M[i,k,j] = 1 (the standard convention:
    # C[i,k] = sum_j A[i,j] B[j,k])
    # So the tensor is over indices (i, j, k) with value 1 for all valid triples.
    # Actually the matmul tensor T has T[i,j,k] = 1 for all i,j,k (it's the
    # "all ones" tensor in the sense that every (i,j,k) triple appears exactly once).
    # The rank of this tensor is what we want to minimize.

    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p

    def build_tensor():
        """Return the matmul tensor as a 3D array T[i,j,k]."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    def verify(U, V, W, T, tol=1e-9):
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == T[i,j,k] for all i,j,k."""
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - T[i,j,k]) > tol:
                        return False
        return True

    def verify_fast(U, V, W, T, tol=1e-9):
        """Vectorized verification."""
        rank = U.shape[1]
        # Reshape for einsum
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        # Compute sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        return np.allclose(result, T, atol=tol)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def to_half_int(mat, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        result = mat.copy()
        for idx in np.ndindex(result.shape):
            v = result[idx]
            nearest = round(v * 2) / 2.0
            if abs(v - nearest) < tol:
                result[idx] = nearest
        return result

    def snap_to_half_integers(U, V, W, tol=1e-6):
        return to_half_int(U, tol), to_half_int(V, tol), to_half_int(W, tol)

    T = build_tensor()

    # Try to find lower rank decompositions using various strategies
    # Strategy: use the fact that for small matrices, we can try structured
    # decompositions or use rank-revealing factorizations.

    # Known lower bounds: For matmul of (n x m) and (m x p), the rank is at least
    # max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    # Also, Strassen's algorithm gives rank 7 for 2x2, so for larger matrices
    # we can use tensor product constructions.

    # For 2x3 * 3x6, we can try:
    # 1. Direct search for rank < 36 using structured forms
    # 2. Use known results: the rank of matmul(2,3,6) is known to be at most
    #    some value. Let's try to find it.

    # Approach: Use the fact that we can decompose the tensor using
    # bilinear algorithms. Try ranks from the theoretical lower bound up.

    # Lower bound: max(n*m, m*p, n*p) = max(6, 18, 12) = 18
    # Upper bound: schoolbook = 36

    # Let's try a structured search:
    # For rank r, we need to find U (6 x r), V (18 x r), W (12 x r) such that
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k

    # This is equivalent to: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
    # (appropriately indexed) should sum to the all-ones tensor.

    # Let's try using numpy's lstsq or SVD-based approaches.
    # Actually, let's try a different approach: use the fact that the tensor
    # T[i,j,k] = 1 can be written as a sum of rank-1 tensors.

    # A rank-1 tensor has the form a[i]*b[j]*c[k] for vectors a, b, c.
    # We need sum_r a_r[i] * b_r[j] * c_r[k] = 1 for all i,j,k.

    # This means the matrix A (n x r), B (m x r), C (p x r) must satisfy:
    # A @ B.T @ C.T (with appropriate reshaping) gives all ones.
    # Actually: sum_r A[i,r]*B[j,r]*C[k,r] = (A ⊙ B ⊙ C) contracted = 1

    # Let's reformulate: we need vectors a_r in R^n, b_r in R^m, c_r in R^p
    # such that for all i,j,k: sum_r a_r[i]*b_r[j]*c_r[k] = 1.

    # This is equivalent to: the matrix M_{i,k} = sum_r (a_r[i]*c_r[k]) * (sum_j' b_r[j']*delta) ...
    # No, let's think differently.

    # sum_r a_r[i]*b_r[j]*c_r[k] = 1 for all i,j,k
    # Fix i,k: sum_r (a_r[i]*c_r[k]) * b_r[j] = 1 for all j
    # So for each (i,k), the vector v_{i,k} in R^m defined by v_{i,k}[j] = 1
    # must equal sum_r (a_r[i]*c_r[k]) * b_r

    # This means the set of vectors {b_r} spans R^m, and for each (i,k),
    # the coefficients (a_r[i]*c_r[k]) give the expansion of the all-ones vector.

    # Let's try a direct numerical approach for small ranks.

    def try_rank_r(r, T, time_limit):
        """Try to find a rank-r decomposition."""
        # We need A (n x r), B (m x r), C (p x r)
        # such that sum_r A[i,r]*B[j,r]*C[k,r] = T[i,j,k] = 1

        # Initialize with random values
        np.random.seed(42)
        best_residual = np.inf
        best_A, best_B, best_C = None, None, None

        # Use iterative optimization
        A = np.random.randn(n, r) * 0.1
        B = np.random.randn(m, r) * 0.1
        C = np.random.randn(p, r) * 0.1

        for it in range(2000):
            if time.time() > time_limit:
                break

            # Compute residual: R[i,j,k] = sum_r A[i,r]*B[j,r]*C[k,r] - T[i,j,k]
            # Use gradient descent
            # dR/dA[i,r] = B[j,r]*C[k,r] (summed over j,k with R)
            # This is expensive; use a simpler approach

            # Actually, let's use a different strategy: fix two factors and
            # solve for the third.

            # Fix B and C, solve for A:
            # For each i: sum_r A[i,r]*B[j,r]*C[k,r] = 1 for all j,k
            # This is a linear system in A[i,:] for each i.

            # Let's build the linear system.
            # For fixed i, we need: sum_r A[i,r] * (B[j,r]*C[k,r]) = 1 for all j,k
            # The matrix of coefficients is M_{(j,k), r} = B[j,r]*C[k,r]
            # Shape: (m*p, r)

            M_bc = (B[:, np.newaxis, :] * C[np.newaxis, :, :]).reshape(m*p, r)
            # We need M_bc @ A[i,:] = ones(m*p) for each i
            # This is the same for all i, so A[i,:] = same solution for all i
            # That means A must have identical rows, which is a special case.

            # This approach only works if the tensor has a special structure.
            # For the all-ones tensor, let's try:
            # If A[i,r] = a_r (same for all i), B[j,r] = b_r (same for all j),
            # C[k,r] = c_r (same for all k), then sum_r a_r*b_r*c_r = 1.
            # This gives rank 1 if we can find a_r, b_r, c_r with sum_r a_r*b_r*c_r = 1.
            # But this only works if the tensor is a pure tensor (rank 1), which
            # the all-ones tensor is NOT in general (it's rank 1 only if n=m=p=1).

            # Wait, the all-ones tensor T[i,j,k] = 1 for all i,j,k IS rank 1!
            # Because T[i,j,k] = 1*1*1 for all i,j,k.
            # So the rank is 1?!

            # Let me re-examine. The matmul tensor is NOT the all-ones tensor.
            # The matmul tensor T has T[i,j,k] = 1 if the (i,j) entry of A and
            # (j,k) entry of B multiply to give the (i,k) entry of C.
            # In the standard tensor decomposition of matmul, the tensor is:
            # T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor that represents this bilinear map has entries indexed by
            # (i, j, k) and the value is 1 (since A[i,j]*B[j,k] is the term).
            # So yes, the tensor is the all-ones tensor in the sense that every
            # (i,j,k) triple has value 1.

            # But wait, that can't be right because the rank of matmul(2,2,2) is 7,
            # not 1. Let me reconsider.

            # The matmul tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m),
            # k in [0,p). The rank of this tensor (as a 3-way tensor) is the
            # minimum number of rank-1 tensors that sum to it.
            # A rank-1 tensor has the form a[i]*b[j]*c[k].
            # The all-ones tensor T[i,j,k] = 1 can be written as:
            # T[i,j,k] = 1*1*1, which is rank 1.

            # So the rank of the all-ones tensor is 1. But that contradicts
            # Strassen's result that matmul(2,2,2) has rank 7.

            # I think I'm confusing the tensor with the bilinear map. Let me reconsider.

            # The matrix multiplication as a bilinear map is:
            # (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
            # This is a bilinear map from R^{n*m} x R^{m*p} to R^{n*p}.
            # The rank of this bilinear map is the minimum number of terms of the form
            # (u^T A) * (v^T B) * w
            # where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}.

            # The tensor associated with this bilinear map has entries:
            # T[a,b,c] where a indexes A (a = i*m+j), b indexes B (b = j*p+k),
            # c indexes C (c = k*n+i).
            # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k,
            # and 0 otherwise.

            # So the tensor is NOT all ones. It has 1s at specific positions.
            # The number of 1s is n*m*p = 36 for our case.

            # Let me rebuild the tensor correctly.
            break

        return None, None, None, np.inf

    # Rebuild the tensor correctly
    def build_matmul_tensor():
        """Build the matmul tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
        T = np.zeros((n*m, m*p, n*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        return T

    T_correct = build_matmul_tensor()

    # Now the tensor has 1s at specific positions. The rank is what we want to minimize.
    # Schoolbook gives rank = n*m*p = 36.

    # Let's verify the schoolbook decomposition works with the correct tensor.
    U_sb, V_sb, W_sb = schoolbook()
    assert verify_fast(U_sb, V_sb, W_sb, T_correct), "Schoolbook decomposition failed!"

    # Now try to find lower rank.
    # For matmul(2,3,6), known results suggest the rank might be less than 36.
    # Let's try a search approach.

    # Use a heuristic: try to find a decomposition with rank r < 36.
    # We'll use a combination of random initialization and local optimization.

    def optimize_decomposition(r, T, max_iter=5000, tol=1e-10):
        """Try to find a rank-r decomposition using gradient-based optimization."""
        np.random.seed(0)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(n*p, r) * 0.5

        # Normalize to prevent explosion
        for col in range(r):
            norm_u = np.linalg.norm(U[:, col])
            norm_v = np.linalg.norm(V[:, col])
            norm_w = np.linalg.norm(W[:, col])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, col] /= scale
                V[:, col] /= scale
                W[:, col] /= scale

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Compute the current tensor estimate
            # T_est[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Only need to check positions where T has 1s
            ones_positions = np.argwhere(T == 1.0)
            if len(ones_positions) == 0:
                break

            # Compute residual at ones positions
            residual = np.zeros(len(ones_positions))
            for idx, (a, b, c) in enumerate(ones_positions):
                residual[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - T[a, b, c]

            # Compute gradient (approximate)
            # d/dU[a,r] = V[b,r]*W[c,r] for each (a,b,c)
            # This is expensive; use a simpler update

            # Use a heuristic: scale each factor to reduce residual
            mean_res = np.mean(residual)
            if abs(mean_res) < tol:
                break

            # Adjust: if residual is positive, scale down; if negative, scale up
            # This is a crude approach
            for col in range(r):
                # Estimate contribution of column col
                contrib = 0.0
                count = 0
                for a, b, c in ones_positions[:100]:  # sample
                    contrib += U[a, col] * V[b, col] * W[c, col]
                    count += 1
                if count > 0:
                    contrib /= count

                if abs(contrib) > 1e-12:
                    # Scale to reduce residual
                    factor = 1.0 - 0.01 * np.sign(contrib) * mean_res
                    U[:, col] *= factor
                    V[:, col] *= factor
                    W[:, col] *= factor

            # Renormalize
            for col in range(r):
                norm_u = np.linalg.norm(U[:, col])
                norm_v = np.linalg.norm(V[:, col])
                norm_w = np.linalg.norm(W[:, col])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    U[:, col] /= scale
                    V[:, col] /= scale
                    W[:, col] /= scale

        # Final check
        if verify_fast(U, V, W, T_correct, tol=1e-6):
            return U, V, W

        return None, None, None

    # Try ranks from 18 to 35
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank

    for r in range(18, min(36, schoolbook_rank)):
        if time.time() > deadline - 10:
            break

        print(f"Trying rank {r}...", flush=True)
        U_r, V_r, W_r = optimize_decomposition(r, T_correct, max_iter=1000)
        if U_r is not None:
            # Snap to half-integers
            U_r, V_r, W_r = snap_to_half_integers(U_r, V_r, W_r)
            if verify_fast(U_r, V_r, W_r, T_correct, tol=1e-6):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                print(f"Found rank {r} decomposition!", flush=True)
                break

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction
from math import isqrt

def _fract_to_float(x):
    return float(x)

def _round_to_half(x):
    # Round to nearest half-integer
    return round(x * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Build the target tensor as a sparse-ish structure
    # T[i, j, k] = 1 if output index (k*n + i) depends on input (i*m+j) and (j*p+k)
    # Actually the tensor is: for each (i,j,k), the output C[k*n+i] gets A[i*m+j] * B[j*p+k]
    # So the tensor has entries T[i*m+j, j*p+k, k*n+i] = 1

    # The bilinear form: C[r] = sum over rank of W[r, t] * (sum over (i,j) U[i*m+j, t] * A[i*m+j]) * (sum over (j,k) V[j*p+k, t] * B[j*p+k])
    # We need: for each output index r = k*n + i, C[r] = sum_j A[i*m+j] * B[j*p+k]
    # So: sum_t W[r,t] * (sum_{i',j'} U[i'm+j',t] A[i'm+j']) * (sum_{j',k'} V[j'p+k',t] B[j'p+k']) = sum_j A[i*m+j] B[j*p+k]

    # This means for each (i,j,k): sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] = 1
    # And for all other combinations, it should be 0.

    # Let's use a randomized search with rank reduction
    # Start from schoolbook and try to merge columns

    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Check all (i,j,k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] should be 1
                    s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])
                    if abs(s - 1.0) > tol:
                        return False
        # Check that non-matching combinations give 0
        # For output index r = k*n + i, the only terms that should contribute are those with matching i and k
        # i.e., U[i*m+j, t] * V[j'*p+k, t] * W[r, t] should be 0 unless j == j'
        # More precisely: for fixed (i,k), sum_j' U[i*m+j', t] * V[j'*p+k, t] * W[k*n+i, t] should equal 1 only when j'=j for the specific j
        # Actually the constraint is: for each (i,j,k), the sum is 1, and for (i,j',k) with j' != j, it should be 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        if j == j2:
                            continue
                        s = np.sum(W[k*n+i, :] * U[i*m+j2, :] * V[j*p+k, :])
                        # Wait, this isn't quite right. Let me re-derive.
                        # C[k*n+i] = sum_t W[k*n+i,t] * (sum_{i',j'} U[i'm+j',t] A[i'm+j']) * (sum_{j',k'} V[j'p+k',t] B[j'p+k'])
                        # = sum_t W[k*n+i,t] * sum_{i',j',j',k'} U[i'm+j',t] V[j'p+k',t] A[i'm+j'] B[j'p+k']
                        # We want this to equal sum_j A[i*m+j] B[j*p+k]
                        # So the coefficient of A[i'm+j'] B[j'p+k'] in the expansion should be:
                        # delta(i'm, i) * delta(j', j) * delta(k', k) summed over t
                        # i.e., sum_t W[k*n+i, t] * U[i'm+j', t] * V[j'p+k', t] = 1 if i'=i, j'=j, k'=k else 0
                        pass
        # Simpler check: build the full tensor and compare
        # For each t, the contribution is W[:, t] outer (U[:, t] outer V[:, t]) reshaped
        # Let's just check all (i', j, k') combinations for each output
        for i in range(n):
            for k in range(p):
                r = k*n + i
                for j in range(m):
                    # The coefficient of A[i*m+j] * B[j*p+k] should be 1
                    coeff = np.sum(W[r, :] * U[i*m+j, :] * V[j*p+k, :])
                    if abs(coeff - 1.0) > tol:
                        return False
                for j in range(m):
                    for j2 in range(m):
                        if j2 == j:
                            continue
                        # Coefficient of A[i*m+j2] * B[j*p+k] should be 0
                        coeff = np.sum(W[r, :] * U[i*m+j2, :] * V[j*p+k, :])
                        if abs(coeff) > tol:
                            return False
        # Also check that A[i'*m+j] * B[j*p+k'] for i' != i or k' != k contributes 0 to C[k*n+i]
        for i in range(n):
            for k in range(p):
                r = k*n + i
                for i2 in range(n):
                    for k2 in range(p):
                        if i2 == i and k2 == k:
                            continue
                        for j in range(m):
                            coeff = np.sum(W[r, :] * U[i2*m+j, :] * V[j*p+k2, :])
                            if abs(coeff) > tol:
                                return False
        return True

    # Start with schoolbook
    U, V, W = schoolbook_decomposition()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank by removing columns that are linearly dependent
    # Or by finding a lower-rank decomposition

    # Strategy: try random projections and rank-revealing decompositions
    # For 3x4x7, known results suggest rank might be reducible from 84

    # Try a simple approach: for each pair of columns, check if they can be merged
    # Actually, let's try a more systematic approach using the fact that
    # the tensor has a specific structure

    # Try to find a decomposition of lower rank using random initialization
    # and alternating least squares

    def als_decompose(target_tensor, max_rank, n_iter=100, tol=1e-8):
        """
        target_tensor: function that returns T[a, b, c]
        """
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(nm, max_rank) * 0.1
        V = np.random.randn(mp, max_rank) * 0.1
        W = np.random.randn(pn, max_rank) * 0.1

        for it in range(n_iter):
            # Update U
            for t in range(max_rank):
                # For each a, U[a, t] should satisfy:
                # sum_b,c target[a,b,c] * V[b,t] * W[c,t] = sum_b,c U[a,t] * V[b,t] * W[c,t] * ...
                # Actually, fix V and W, solve for U
                # The contribution of column t to the tensor is W[:,t] outer (U[:,t] outer V[:,t])
                # We want sum_t W[:,t] outer (U[:,t] outer V[:,t]) = target
                # For fixed V, W, the optimal U[a,t] is:
                # U[a,t] = sum_{b,c} target[a,b,c] * V[b,t] * W[c,t] / (sum_{b,c} V[b,t]^2 * W[c,t]^2 + eps)
                pass

        # This is getting complex. Let's use a simpler approach.
        return None

    # Alternative: try specific known decompositions
    # For 3x4x7, let's try to exploit the small dimensions

    # Try rank 83, 82, ... down to some minimum
    # But this is expensive. Let's try a smarter approach.

    # Use the fact that the tensor is sparse and try to find a low-rank approximation
    # that is exact

    # For now, let's try a few random rank-reduction attempts
    # by projecting onto random subspaces

    def try_rank_reduction(U, V, W, target_rank, n_trials=10):
        """Try to reduce the rank by finding a lower-rank equivalent decomposition."""
        current_rank = U.shape[1]
        if target_rank >= current_rank:
            return None

        for trial in range(n_trials):
            # Random projection
            np.random.seed(123 + trial)
            # Create random mixing matrices
            P = np.random.randn(current_rank, target_rank)
            Q = np.random.randn(current_rank, target_rank)
            R = np.random.randn(current_rank, target_rank)

            # New factors: U' = U @ P, V' = V @ Q, W' = W @ R
            # But this changes the decomposition unless P, Q, R are related
            # Actually, we need U' @ diag(something) @ V' @ W' to give the same tensor
            # This is tricky. Let's use a different approach.

            # Try to find a subset of columns that spans the same space
            # Or use SVD to find a lower-rank approximation

        return None

    # Let's try a different approach: use the structure of the problem
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)

    # For 3x4x7, let's try to find a decomposition using the fact that
    # we can share some computation

    # Actually, let's just return the schoolbook decomposition for now
    # and add a simple verification

    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: return schoolbook
    U, V, W = schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

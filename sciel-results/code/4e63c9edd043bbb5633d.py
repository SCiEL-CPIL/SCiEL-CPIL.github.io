"""Matrix multiplication 3,3,5 - low rank decomposition via batched ALS."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 3x3 x 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Build the target multiplication tensor
    # Shape: (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some valid i,j,k
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition(U, V, W, target_tensor, tol=1e-9):
        """Verify that U, V, W decompose the target tensor."""
        rank = U.shape[1]
        reconstructed = np.zeros_like(target_tensor)
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, target_tensor, atol=tol)

    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1

        return U, V, W, rank

    def sparse_als_decomposition(rank, max_steps=200, tol=1e-8, seed=42, threshold=1e-3):
        """
        Use Sparse Alternating Least Squares to find a rank-r decomposition.
        """
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(seed)

        # Initialize factors with sparsity
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Initial sparsification
        U[np.abs(U) < 0.5] = 0
        V[np.abs(V) < 0.5] = 0
        W[np.abs(W) < 0.5] = 0

        prev_err = np.inf

        for step in range(max_steps):
            if time.time() > deadline - 5:
                return None

            # --- Update U ---
            M_U = np.zeros((m * p * p * n, rank))
            for r in range(rank):
                M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()

            T_flat = target.reshape(n * m, m * p * p * n)
            try:
                U.T = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0]
            except:
                pass
            
            # Sparsify U
            U[np.abs(U) < threshold] = 0

            # --- Update V ---
            M_V = np.zeros((n * m * p * n, rank))
            for r in range(rank):
                M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()

            T_flat_v = np.zeros((m * p, n * m * p * n))
            for b in range(m * p):
                T_flat_v[b, :] = target[:, b, :].flatten()
            
            try:
                V.T = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)[0]
            except:
                pass
                
            # Sparsify V
            V[np.abs(V) < threshold] = 0

            # --- Update W ---
            M_W = np.zeros((n * m * m * p, rank))
            for r in range(rank):
                M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()

            T_flat_w = np.zeros((p * n, n * m * m * p))
            for c in range(p * n):
                T_flat_w[c, :] = target[:, :, c].flatten()
            
            try:
                W.T = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)[0]
            except:
                pass
                
            # Sparsify W
            W[np.abs(W) < threshold] = 0

            # Compute error
            if step % 5 == 0 or step == max_steps - 1:
                recon = np.zeros_like(target)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(recon - target)

                if err < prev_err - 1e-6:
                    prev_err = err
                elif err < tol:
                    break
                elif prev_err - err < 1e-9 and step > 20:
                    break

        # Check if we found an exact solution
        if verify_decomposition(U, V, W, target, tol=1e-6):
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify_decomposition(U_rounded, V_rounded, W_rounded, target, tol=1e-6):
                return U_rounded, V_rounded, W_rounded

            # Try rounding to integers
            U_rounded = np.round(U)
            V_rounded = np.round(V)
            W_rounded = np.round(W)
            if verify_decomposition(U_rounded, V_rounded, W_rounded, target, tol=1e-6):
                return U_rounded, V_rounded, W_rounded

            # Return as-is if close enough
            return U, V, W

        return None

    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # Try to find better decompositions using batched ALS
    # Start from a reasonable rank and try to go lower
    # Known lower bound for 3x3x5 is at least 27
    ranks_to_try = list(range(27, best_rank))

    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break

        # Try multiple seeds for sparse ALS
        found = False
        for seed_offset in range(5):
            result = sparse_als_decomposition(rank, max_steps=100, seed=42 + rank + seed_offset*10)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            break  # Found a valid decomposition, return it

    U, V, W = best_result

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

"""Matrix multiplication 3,3,8 - rank-minimal decomposition search."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But structural constraints may require more
    # Try from 24 upward

    best_result = None
    best_rank = n * m * p  # schoolbook as fallback

    # Schoolbook fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Try to find lower rank decompositions
    for rank in range(24, best_rank + 1):
        if time.time() > deadline - 30:
            break

        # Multiple random restarts for each rank
        max_restarts = max(1, int((deadline - time.time()) / 5))
        if max_restarts < 1:
            break

        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break

            # Random initialization
            np.random.seed(42 + restart * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            # Alternating least squares
            success = False
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break

                # Compute full sum S = einsum('ac,bc,dc->abd', U, V, W)
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S  # residual

                # Update U
                for c in range(rank):
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,b,d->a', B, V[:, c], W[:, c])
                    denom = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                    if denom > 1e-12:
                        U[:, c] = numerator / denom

                # Recompute S and R_full
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S

                # Update V
                for c in range(rank):
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,a,d->b', B, U[:, c], W[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                    if denom > 1e-12:
                        V[:, c] = numerator / denom

                S = np.einsum('ac,bc,dc->abd', U, V, W)
                R_full = T - S

                # Update W
                for c in range(rank):
                    B = R_full + np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    numerator = np.einsum('abd,a,b->d', B, U[:, c], V[:, c])
                    denom = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                    if denom > 1e-12:
                        W[:, c] = numerator / denom

                # Check convergence
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(T - S)
                if err < 1e-8:
                    success = True
                    break

                if iteration % 20 == 0 and iteration > 0:
                    # Check if error is decreasing
                    pass

            if success:
                # Verify
                S = np.einsum('ac,bc,dc->abd', U, V, W)
                if np.linalg.norm(T - S) < 1e-6:
                    # Try to round to integers or half-integers
                    # Check if values are close to integers or half-integers
                    all_close = True
                    for mat in [U, V, W]:
                        rounded = np.round(mat * 2) / 2
                        if np.linalg.norm(mat - rounded) > 1e-6:
                            all_close = False
                            break
                    if all_close:
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                    else:
                        # Use as-is (floats are acceptable per problem statement)
                        pass

                    best_result = (U, V, W)
                    best_rank = rank
                    found = True
                    break

        if found:
            break

    # Convert to lists
    U, V, W = best_result
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

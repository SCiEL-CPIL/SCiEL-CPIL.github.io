import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
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

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        # A is 2 rows, 4 cols. B is 4 rows, 7 cols. C is 2 rows, 7 cols.
        # Split A cols into pairs: (0,1), (2,3)
        # Split B rows into pairs: (0,1), (2,3)
        # Split B cols: (0,1), (2,3), (4,5), and col 6 separately
        # Split C cols: (0,1), (2,3), (4,5), and col 6 separately

        # Strassen blocks for 2x2 x 2x2:
        # j0 = 0,2 (A col pairs / B row pairs)
        # k0 = 0,2,4 (B col pairs / C col pairs)
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        # A[i,j] * B[j,6] for i=0,1; j=0,1,2,3
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def try_lower_rank_random():
        """Attempt to find a lower rank decomposition via random initialization."""
        if time.time() > deadline:
            return None
        
        best_rank = None
        best_decomp = None
        
        # Try ranks from 46 down to 42
        for target_rank in [46, 45, 44, 43, 42]:
            if time.time() > deadline:
                break
            
            # Try a few random initializations
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + target_rank * 100 + attempt)
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # Use gradient descent to minimize ||sum_r U_r V_r W_r - T||
                # This is a non-convex problem, but we can try a few iterations
                
                lr = 0.01
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Compute gradient
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = recon - T
                    
                    # Gradients
                    dU = np.einsum('abc,br,cr->ar', error, V, W)
                    dV = np.einsum('abc,ar,cr->br', error, U, W)
                    dW = np.einsum('abc,ar,br->cr', error, U, V)
                    
                    # Update
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    # Check convergence
                    if np.max(np.abs(error)) < 1e-6:
                        # Verify exactly
                        if verify_decomp(U, V, W):
                            if best_rank is None or target_rank < best_rank:
                                best_rank = target_rank
                                best_decomp = (U.copy(), V.copy(), W.copy())
                        break
                
                if best_rank is not None and target_rank == best_rank:
                    break
        
        return best_decomp

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank decomposition if we have time
    if time.time() < deadline - 30:  # Leave 30s safety margin
        lower_decomp = try_lower_rank_random()
        if lower_decomp is not None:
            U, V, W = lower_decomp
            if verify_decomp(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END

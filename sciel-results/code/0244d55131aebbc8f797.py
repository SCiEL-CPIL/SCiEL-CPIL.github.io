import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via a fixed-size Kronecker-of-Strassen construction,
    then a greedy column-pruning search to reduce rank while preserving exactness.
    """
    t0 = time.time()
    DEADLINE = 120.0  # wall-clock limit; keep ~30s safety margin
    SAFETY = 10.0

    n = m = p = 4

    # Strassen 2x2x2 factor matrices (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Kronecker products -> rank 49
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)

    # Build the matrix-multiplication tensor T of shape (4,4,4)
    # T[i, j, k] = 1 if the output C[i,k] gets contribution A[i,j]*B[j,k]
    # In the factorization convention: C = sum_r (A @ U[:,r]) * (V[:,r] @ B)
    # The tensor T has T[i,j,k] = 1 for each (i,j,k) triple.
    # Actually, the standard formulation:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # Factorization: C = sum_r A @ U_r * V_r @ B  (outer product of vectors)
    # More precisely with the convention W[k*n+i, r]:
    #   C[k*n+i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+k, r] * B[j,k])
    # Wait, let me re-read the convention.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # U shape: (n*m, rank) -> U[i*m+j, r]
    # V shape: (m*p, rank) -> V[j*p+k, r]
    # W shape: (p*n, rank) -> W[k*n+i, r]
    #
    # The decomposition says:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    #   sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] ... 
    # Actually the standard tensor decomposition:
    #   M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where M[i,j,k] = 1 (the multiplication tensor is all ones).
    #
    # Let's verify: the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k in [0..3].
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # Verify the Kronecker construction
    def verify(U, V, W):
        # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify initial construction
    assert verify(U4, V4, W4), "Initial Kronecker construction failed verification"

    # Greedy pruning: try to remove columns that are "redundant"
    # A column r is redundant if T - sum_{r' != r} (outer product) can be matched
    # by adjusting... actually for exact decomposition, we can only remove a column
    # if it's zero, or if we can re-express.
    #
    # Better approach: try removing one column at a time and check if the remaining
    # columns still span the tensor. Since we need exact equality (not just span),
    # we need: sum_{r in remaining} U[:,r]*V[:,r]*W[:,r] == T (the all-ones tensor).
    #
    # Actually, the simplest structural improvement: use the fact that the Kronecker
    # construction may have linear dependencies. Let's try to find a smaller subset.
    #
    # Alternative: use a completely different known construction.
    # For 4x4x4, the best known is rank 49 (Kronecker of Strassen).
    # There is a result that rank(M_4,4,4) = 49 is not proven; lower bound is 26.
    # But no known decomposition below 49 exists in the literature that's simple.
    #
    # Let's try a different approach: use the "border rank" or "Landsberg" type
    # constructions? No, those are for symbolic.
    #
    # Let's try a greedy algorithm that attempts to find a lower-rank decomposition
    # by starting from the Kronecker construction and trying to remove columns.

    def tensor_residual(U, V, W, excluded_cols):
        """Compute the residual tensor after excluding certain columns."""
        res = np.zeros((n, m, p), dtype=float)
        for r in range(U.shape[1]):
            if r in excluded_cols:
                continue
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # The contribution to T[i,j,k] is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return res

    def is_valid_decomp(U, V, W, cols_to_keep):
        """Check if the subset of columns still gives exact decomposition."""
        n_keep = len(cols_to_keep)
        U_sub = U[:, cols_to_keep]
        V_sub = V[:, cols_to_keep]
        W_sub = W[:, cols_to_keep]
        return verify(U_sub, V_sub, W_sub)

    # Try to greedily remove columns
    current_cols = list(range(U4.shape[1]))
    removed = []
    
    for col in range(U4.shape[1]):
        if time.time() > DEADLINE - SAFETY:
            break
        if col not in current_cols:
            continue
        trial = [c for c in current_cols if c != col]
        if is_valid_decomp(U4, V4, W4, trial):
            current_cols = trial
            removed.append(col)
    
    # Try a second pass
    for col in range(U4.shape[1]):
        if time.time() > DEADLINE - SAFETY:
            break
        if col not in current_cols:
            continue
        trial = [c for c in current_cols if c != col]
        if is_valid_decomp(U4, V4, W4, trial):
            current_cols = trial

    # Try pairs of columns to remove
    if time.time() < DEADLINE - SAFETY:
        n_cols = len(current_cols)
        for a in range(n_cols):
            for b in range(a+1, n_cols):
                if time.time() > DEADLINE - SAFETY:
                    break
                ca, cb = current_cols[a], current_cols[b]
                trial = [c for c in current_cols if c != ca and c != cb]
                if is_valid_decomp(U4, V4, W4, trial):
                    current_cols = trial
                    # Restart the pair search since columns changed
                    break
            if time.time() > DEADLINE - SAFETY:
                break

    U_final = U4[:, current_cols]
    V_final = V4[:, current_cols]
    W_final = W4[:, current_cols]

    rank = len(current_cols)
    print(f"Final rank: {rank}")

    # Fallback to schoolbook if something goes wrong or if rank didn't improve
    if rank >= 64 or not verify(U_final, V_final, W_final):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col_idx] = 1
                    V[j*p+k, col_idx] = 1
                    W[k*n+i, col_idx] = 1
                    col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

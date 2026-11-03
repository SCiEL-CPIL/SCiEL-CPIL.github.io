"""Matrix multiplication 3x4x8 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_88_decomposition():
    """Build the rank-88 decomposition using Strassen blocks."""
    n, m, p = 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    # 8 Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    # 8 schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j1 * p + k0, j0 * p + k1, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True


def _try_rank_reduction(U, V, W, n, m, p, max_time):
    """
    Try to reduce the rank of the decomposition by removing redundant columns.
    Uses a greedy approach: for each column, check if removing it still gives
    a valid decomposition. If so, remove it and continue.
    """
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    n, m, p = n, m, p
    
    U_m = U_arr.reshape(n * m, -1)
    V_m = V_arr.reshape(m * p, -1)
    W_m = W_arr.reshape(p * n, -1)
    
    current_rank = U_m.shape[1]
    
    # Precompute the tensor values
    # T[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    def compute_tensor(U_m, V_m, W_m):
        """Compute the full tensor from factor matrices."""
        n, m, p = n, m, p
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(
                        U_m[i * m + j] * V_m[j * p + k] * W_m[k * n + i]
                    )
        return T
    
    target_tensor = np.ones((n, m, p))
    current_tensor = compute_tensor(U_m, V_m, W_m)
    
    if not np.allclose(current_tensor, target_tensor, atol=1e-9):
        return U, V, W  # Can't start if already invalid
    
    removed_cols = []
    
    # Try removing each column one at a time
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        
        found_removal = False
        for c in range(current_rank):
            if time.time() - start_time > max_time:
                break
            
            # Try removing column c
            U_trial = np.delete(U_m, c, axis=1)
            V_trial = np.delete(V_m, c, axis=1)
            W_trial = np.delete(W_m, c, axis=1)
            
            trial_tensor = compute_tensor(U_trial, V_trial, W_trial)
            
            if np.allclose(trial_tensor, target_tensor, atol=1e-9):
                # Column c is redundant! Remove it.
                U_m = U_trial
                V_m = V_trial
                W_m = W_trial
                current_rank -= 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if current_rank == U_m.shape[1]:
        return U, V, W
    
    # Convert back to list format
    new_U = U_m.tolist()
    new_V = V_m.tolist()
    new_W = W_m.tolist()
    
    return new_U, new_V, new_W


def _construct_3x4x8():
    """Construct the decomposition."""
    n, m, p = DIMENSIONS
    
    # Return the known rank-88 solution
    return _build_88_decomposition()


def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    
    # Verify the decomposition to ensure correctness
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Try to reduce rank if we have time
    if time.time() < deadline - 30:
        time_remaining = deadline - time.time() - 10  # Leave 10s margin
        if time_remaining > 5:
            try:
                new_U, new_V, new_W = _try_rank_reduction(
                    result["U"], result["V"], result["W"], n, m, p, time_remaining
                )
                # Verify the reduced decomposition
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    result = {"U": new_U, "V": new_V, "W": new_W}
            except Exception:
                pass  # Keep original if reduction fails
    
    return result

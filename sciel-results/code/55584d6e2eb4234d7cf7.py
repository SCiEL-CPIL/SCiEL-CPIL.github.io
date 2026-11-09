"""Program for 3x5x5 matrix multiplication tensor decomposition.

Strategy: Use a greedy local search to find a low-rank decomposition with
integer entries. Start from the schoolbook rank (75) and search downwards.
If no lower rank is found, return the schoolbook decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _time_remaining():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _schoolbook():
    """Return the standard rank-75 decomposition."""
    n, m, p = N, M, P
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

def _verify(decomp, tolerance=1e-6):
    """Check if the decomposition exactly realizes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    n, m, p = N, M, P
    rank = U.shape[1]
    
    # Reshape to tensor dimensions
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    
    # Compute sum over rank dimension c: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    total = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    
    # The target tensor for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # is T[i,j,k] = 1 for all i,j,k.
    target = np.ones((n, m, p))
        
    return np.allclose(total, target, atol=tolerance)

def _local_search(rank, max_steps=2000, alphabet=(-1, 0, 1)):
    """
    Attempt to find a decomposition of the given rank using greedy local search.
    Returns (decomp, success)
    """
    n, m, p = N, M, P
    
    # Initialize random sparse matrices
    U = np.zeros((n, m, rank), dtype=np.int8)
    V = np.zeros((m, p, rank), dtype=np.int8)
    W = np.zeros((p, n, rank), dtype=np.int8)
    
    # Fill with random values from alphabet
    for c in range(rank):
        # Randomly select a few non-zero entries
        u_sparsity = random.randint(1, 3)
        v_sparsity = random.randint(1, 3)
        w_sparsity = random.randint(1, 3)
        
        u_idx = random.sample(range(n*m), u_sparsity)
        v_idx = random.sample(range(m*p), v_sparsity)
        w_idx = random.sample(range(p*n), w_sparsity)
        
        for idx in u_idx:
            i, j = divmod(idx, m)
            U[i, j, c] = random.choice(alphabet)
        for idx in v_idx:
            j, k = divmod(idx, p)
            V[j, k, c] = random.choice(alphabet)
        for idx in w_idx:
            k, i = divmod(idx, n)
            W[k, i, c] = random.choice(alphabet)
            
    target = np.ones((n, m, p), dtype=np.float64)
    
    # Compute initial S and Error
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    E = target - S
    current_err = np.sum(E**2)
    
    best_err = current_err
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Precompute indices for fast access
    u_indices = [(i, j, c) for i in range(n) for j in range(m) for c in range(rank)]
    v_indices = [(j, k, c) for j in range(m) for k in range(p) for c in range(rank)]
    w_indices = [(k, i, c) for k in range(p) for i in range(n) for c in range(rank)]
    
    for step in range(max_steps):
        if _time_remaining() < 1.0:
            break
            
        # Pick a random move
        mat = random.randint(0, 2)
        
        if mat == 0:
            i, j, c = random.choice(u_indices)
            old_val = U[i, j, c]
            new_val = random.choice(alphabet)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            # Delta S[i, j, k] = delta * V[j, k, c] * W[k, i, c]
            # V[j, :, c] shape (p,), W[:, i, c] shape (p,)
            delta_S = delta * V[j, :, c] * W[:, i, c]
            # Update Error: E' = E - delta_S
            # Err' = Err - 2 * sum(E[i,j,:] * delta_S) + sum(delta_S**2)
            e_slice = E[i, j, :]
            delta_err = -2 * np.dot(e_slice, delta_S) + np.dot(delta_S, delta_S)
            
            if delta_err < -1e-9:
                U[i, j, c] = new_val
                S[i, j, :] += delta_S
                E[i, j, :] -= delta_S
                current_err += delta_err
                if current_err < best_err:
                    best_err = current_err
                    if best_err < 1e-6:
                        break
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
        elif mat == 1:
            j, k, c = random.choice(v_indices)
            old_val = V[j, k, c]
            new_val = random.choice(alphabet)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            # Delta S[i, j, k] = U[i, j, c] * delta * W[k, i, c]
            # U[:, j, c] shape (n,), W[k, :, c] shape (n,)
            delta_S = U[:, j, c] * delta * W[k, :, c]
            # This delta_S is a vector over i.
            # We need to update S[i, j, k] for all i.
            # E[i, j, k] is affected.
            # delta_err = -2 * sum(E[:, j, k] * delta_S) + sum(delta_S**2)
            e_slice = E[:, j, k]
            delta_err = -2 * np.dot(e_slice, delta_S) + np.dot(delta_S, delta_S)
            
            if delta_err < -1e-9:
                V[j, k, c] = new_val
                S[:, j, k] += delta_S
                E[:, j, k] -= delta_S
                current_err += delta_err
                if current_err < best_err:
                    best_err = current_err
                    if best_err < 1e-6:
                        break
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
        else:
            k, i, c = random.choice(w_indices)
            old_val = W[k, i, c]
            new_val = random.choice(alphabet)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            # Delta S[i, j, k] = U[i, j, c] * V[j, k, c] * delta
            # U[i, :, c] shape (m,), V[:, k, c] shape (m,)
            delta_S = U[i, :, c] * V[:, k, c] * delta
            # delta_S is a vector over j.
            # Updates S[i, j, k] for all j.
            e_slice = E[i, :, k]
            delta_err = -2 * np.dot(e_slice, delta_S) + np.dot(delta_S, delta_S)
            
            if delta_err < -1e-9:
                W[k, i, c] = new_val
                S[i, :, k] += delta_S
                E[i, :, k] -= delta_S
                current_err += delta_err
                if current_err < best_err:
                    best_err = current_err
                    if best_err < 1e-6:
                        break
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

    if best_err < 1e-6:
        # Convert back to list format
        def to_list(mat_3d, rows, cols):
            # mat_3d is (dim1, dim2, rank)
            # We need (dim1*dim2, rank)
            flat = mat_3d.reshape(rows * cols, rank)
            return flat.tolist()
            
        decomp = {
            "U": to_list(best_U, n, m),
            "V": to_list(best_V, m, p),
            "W": to_list(best_W, p, n)
        }
        return decomp, True
    return None, False

def construct():
    # Start with the safe, known schoolbook decomposition
    base = _schoolbook()
    best_rank = N * M * P  # 75
    
    # Try to find a better decomposition
    # Search ranks from 74 down to 25
    for rank in range(best_rank - 1, 24, -1):
        if _time_remaining() < 5:
            break
        
        # Try a few restarts for this rank
        for restart in range(20):
            if _time_remaining() < 2:
                break
                
            candidate, success = _local_search(rank, max_steps=1000)
            if success:
                # Verify to be absolutely sure
                if _verify(candidate):
                    return candidate
                    
    return base

if __name__ == "__main__":
    result = construct()
    if _verify(result):
        print("Valid decomposition found with rank", len(result["U"][0]))
    else:
        print("ERROR: Invalid decomposition")

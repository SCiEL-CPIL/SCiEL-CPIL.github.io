"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T as a 3D array.
    T[i, j, k] = 1 if i < n, j < m, k < p.
    But we need to map to the specific factor shapes:
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    Tensor T[a, b, c] where a is index in U (0..nm-1), b in V (0..mp-1), c in W (0..pn-1).
    
    Index mapping:
    a = i * m + j
    b = j * p + k
    c = k * n + i
    
    T[a, b, c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n)
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _tensor_error(U, V, W, T, rank):
    """
    Compute the error of the decomposition U, V, W against tensor T.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    T: (nm, mp, pn)
    Error = sum_{a,b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
    """
    # Efficiently compute the reconstructed tensor using einsum or loops
    # R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # This is equivalent to:
    # Temp[a, b, r] = U[a, r] * V[b, r]
    # R[a, b, c] = sum_r Temp[a, b, r] * W[c, r]
    
    # Using einsum for speed
    # R = np.einsum('ar,br,cr->abc', U, V, W)
    # However, for small ranks and dimensions, loops might be competitive or we can use matrix multiplication tricks.
    # Let's try einsum first.
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a 3-way contraction.
    # We can do: M1[a,b,r] = U[a,r] V[b,r]
    # Then R[a,b,c] = sum_r M1[a,b,r] W[c,r]
    
    # M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]
    # R = np.einsum('abr,cr->abc', M1, W_np)
    
    # Let's just do it with loops if numpy isn't available or to be safe with small sizes, but numpy is available.
    # The sizes are small: U (6, r), V (12, r), W (8, r).
    # r <= 24.
    # Total ops ~ 6*12*8*24 = 13824 multiplications. Very fast.
    
    # Using numpy for vectorization
    M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]  # (nm, mp, rank)
    R = np.einsum('abr,cr->abc', M1, W_np)                # (nm, mp, pn)
    
    err = np.sum((R - T) ** 2)
    return err

def _try_polish(U, V, W, T, rank, deadline, initial_err):
    """
    Local search refinement to find an exact or near-exact solution.
    """
    best_err = initial_err
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Gradient descent or random perturbation?
    # Random perturbation with acceptance is simpler and robust for small problems.
    # We want integer/half-integer entries.
    # The problem asks for "integer/half-integer entries".
    # This suggests we should search in a discrete space or snap to the nearest half-integer.
    
    # Let's do a hybrid: continuous search, then snap.
    # Or discrete search directly?
    # Given the small size, we can try discrete search over {-1, -0.5, 0, 0.5, 1} or similar.
    # But the search space is huge: 5^(6*24) etc.
    
    # Better: Use the continuous optimization to get close, then snap to nearest half-integer and verify.
    
    # Let's do a simple random walk with step size decay.
    step_size = 0.5
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Pick a random matrix and element
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            mat_np = U_np
            rows = len(U_np)
        elif mat_choice == 1:
            mat_np = V_np
            rows = len(V_np)
        else:
            mat_np = W_np
            rows = len(W_np)
        
        r = random.randint(0, rows - 1)
        c = random.randint(0, rank - 1)
        
        old_val = mat_np[r, c]
        new_val = old_val + random.gauss(0, step_size)
        mat_np[r, c] = new_val
        
        # Update the corresponding list
        if mat_choice == 0:
            U[r][c] = new_val
            U_np[r, c] = new_val
        elif mat_choice == 1:
            V[r][c] = new_val
            V_np[r, c] = new_val
        else:
            W[r][c] = new_val
            W_np[r, c] = new_val
        
        new_err = _tensor_error(U, V, W, T, rank)
        
        if new_err < best_err:
            best_err = new_err
            if best_err < 1e-10:
                # Snap to nearest half-integer
                U = [[round(x * 2) / 2 for x in row] for row in U]
                V = [[round(x * 2) / 2 for x in row] for row in V]
                W = [[round(x * 2) / 2 for x in row] for row in W]
                # Verify exactness
                if _tensor_error(U, V, W, T, rank) < 1e-10:
                    return U, V, W, True
                # If snap failed, revert to continuous best? No, keep continuous best but mark as not exact.
                # Actually, if snap fails, the continuous solution might be exact but not half-integer?
                # The task requires integer/half-integer.
                # So we must return a valid half-integer solution.
                # If the continuous solution is close to a half-integer solution, snapping should work.
                # If not, we should probably continue searching or return best effort.
                # Let's just return the snapped version if it's close, otherwise continue.
                # But if we snapped and it's not exact, we lost the exact continuous solution.
                # Let's keep the continuous solution and only snap at the end.
                # For now, just accept the improvement.
        else:
            # Revert
            if mat_choice == 0:
                U[r][c] = old_val
                U_np[r, c] = old_val
            elif mat_choice == 1:
                V[r][c] = old_val
                V_np[r, c] = old_val
            else:
                W[r][c] = old_val
                W_np[r, c] = old_val
        
        # Decay step size
        step_size *= 0.999
        
    # Final snap
    U = [[round(x * 2) / 2 for x in row] for row in U]
    V = [[round(x * 2) / 2 for x in row] for row in V]
    W = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _tensor_error(U, V, W, T, rank) < 1e-10:
        return U, V, W, True
    
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145.0
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    T = _build_tensor(n, m, p)
    
    # Try ranks from 24 down to 20
    # 24 is schoolbook (exact)
    # 23, 22, 21, 20
    for target_rank in [24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook truncated to target_rank
        U, V, W = _schoolbook(n, m, p)
        U = [row[:target_rank] for row in U]
        V = [row[:target_rank] for row in V]
        W = [row[:target_rank] for row in W]
        
        initial_err = _tensor_error(U, V, W, T, target_rank)
        
        # If rank is 24, it's already exact
        if initial_err < 1e-10:
            return {"U": U, "V": V, "W": W}
        
        # Try to refine
        U, V, W, success = _try_polish(U, V, W, T, target_rank, deadline, initial_err)
        
        if success:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

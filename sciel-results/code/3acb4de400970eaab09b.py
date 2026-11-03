"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 36).
2. Use a structured "pebbling" / sharing approach: identify groups of schoolbook
   terms that share the same A-index or B-index and attempt to combine them using
   bilinear identities (e.g., (a+b)(c+d) = ac + ad + bc + bd) to reduce rank.
3. Additionally, attempt a randomized local-search / annealing pass that perturbs
   the factor matrices and accepts moves that reduce a smooth surrogate loss
   (sum of squared violations of the tensor identity) while keeping entries
   close to half-integers.
4. Return the best valid decomposition found before the deadline.

The tensor identity to satisfy is:
  For all i in [0,n), j in [0,m), j' in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j, j'}

This ensures that C[i,k] = sum_j A[i,j]*B[j,k] when computed via the
decomposition.
"""

import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]


def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """
    Verify the exact tensor condition:
      sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j, j'}
    for all i, j, j', k.
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[jp * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - expected) > tol:
                        return False
    return True


def _verify_decomposition(U, V, W, n, m, p, num_tests=3):
    """Verify by actually computing matrix products."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    rng = np.random.RandomState(12345)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = 0.0
                    for j in range(m):
                        u_val += U_arr[i * m + j, r] * A[i, j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V_arr[j * p + k, r] * B[j, k]
                    s += u_val * v_val * W_arr[k * n + i, r]
                C_computed[i, k] = s

        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

    return U, V, W


def _compute_loss(U_arr, V_arr, W_arr, n, m, p):
    """
    Compute a smooth surrogate loss: sum of squared violations of the tensor identity.
    For all i, j, j', k: violation = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r] - delta_{j,j'}
    """
    rank = U_arr.shape[1]
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[jp * p + k, r] * W_arr[k * n + i, r]
                    loss += (s - expected) ** 2
    return loss


def _try_paired_combination(U, V, W, n, m, p):
    """
    Try to reduce rank by 1 by finding two schoolbook terms that can be merged.
    Two terms (i,j,k) and (i',j',k') can potentially be merged if there exist
    scalars a, b such that the combined term satisfies the tensor identity.
    
    For a simple case: if two terms share the same i and k but different j,
    we can try to use the identity:
      A[i,j]*B[j,k] + A[i,j']*B[j',k] 
    This doesn't directly help. Instead, we look for terms where the product
    structure allows sharing.
    
    A more practical approach: try to find a linear dependency among the
    rank-1 tensors. We set up a system and try to find a non-trivial linear
    combination of rank terms that equals zero.
    
    For now, we try a simpler heuristic: for each pair of terms with the same
    (i,k) but different j, check if we can replace them with a single term
    using the identity (x+y)(z+w) = xz + xw + yz + yw, but this increases rank.
    
    Actually, the standard way to reduce rank is to find that the tensor lies
    in a lower-dimensional subspace of decomposable tensors. We use a randomized
    approach: perturb the factors and see if we can drive the loss to zero
    with fewer terms.
    """
    return None


def _try_local_search(n, m, p, target_rank, deadline, max_iters=5000):
    """
    Try to find a rank-`target_rank` decomposition using local search.
    Initialize randomly, then perform gradient-free perturbations that
    reduce the loss. Accept moves with some probability (simulated annealing).
    """
    rng = np.random.RandomState(42)
    
    U_arr = rng.uniform(-1, 1, (n * m, target_rank))
    V_arr = rng.uniform(-1, 1, (m * p, target_rank))
    W_arr = rng.uniform(-1, 1, (p * n, target_rank))
    
    current_loss = _compute_loss(U_arr, V_arr, W_arr, n, m, p)
    best_loss = current_loss
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    temperature = 1.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random factor and a random entry to perturb
        factor = rng.randint(0, 3)
        if factor == 0:
            idx = (rng.randint(0, n * m), rng.randint(0, target_rank))
            delta = rng.uniform(-0.5, 0.5)
            old_val = U_arr[idx]
            U_arr[idx] += delta
        elif factor == 1:
            idx = (rng.randint(0, m * p), rng.randint(0, target_rank))
            delta = rng.uniform(-0.5, 0.5)
            old_val = V_arr[idx]
            V_arr[idx] += delta
        elif factor == 2:
            idx = (rng.randint(0, p * n), rng.randint(0, target_rank))
            delta = rng.uniform(-0.5, 0.5)
            old_val = W_arr[idx]
            W_arr[idx] += delta
        else:
            # Perturb an entire row
            row = rng.randint(0, n * m)
            U_arr[row, :] += rng.uniform(-0.1, 0.1, target_rank)
            factor = 0
            idx = (row, None)
            delta = 0
            old_val = None
        
        new_loss = _compute_loss(U_arr, V_arr, W_arr, n, m, p)
        
        # Accept if better or with probability exp(-(new_loss - current_loss)/temperature)
        if new_loss < current_loss or rng.random() < np.exp(-(new_loss - current_loss) / temperature):
            current_loss = new_loss
            if new_loss < best_loss:
                best_loss = new_loss
                best_U = U_arr.copy()
                best_V = V_arr.copy()
                best_W = W_arr.copy()
                # Snap to nearest half-integer if close
                if best_loss < 1e-6:
                    # Try snapping
                    U_snap = np.round(best_U * 2) / 2
                    V_snap = np.round(best_V * 2) / 2
                    W_snap = np.round(best_W * 2) / 2
                    snap_loss = _compute_loss(U_snap, V_snap, W_snap, n, m, p)
                    if snap_loss < 1e-8:
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        else:
            # Revert
            if factor == 0 and idx[1] is not None:
                U_arr[idx] = old_val
            elif factor == 1 and idx[1] is not None:
                V_arr[idx] = old_val
            elif factor == 2 and idx[1] is not None:
                W_arr[idx] = old_val
            elif factor == 3:
                U_arr[row, :] -= rng.uniform(-0.1, 0.1, target_rank)  # approximate revert
        
        temperature *= cooling
    
    if best_loss < 1e-8:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        snap_loss = _compute_loss(U_snap, V_snap, W_snap, n, m, p)
        if snap_loss < 1e-8:
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None


def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Try to find better decompositions using local search
    # Lower bound is max(n*m, m*p, n*p) = max(9, 12, 12) = 12
    # Try ranks from 12 up to 35, but focus on promising ranges
    
    # Try a few key target ranks with local search
    target_ranks = [12, 13, 14, 15, 16, 18, 20, 22, 24, 27, 30, 32, 34, 35]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result = _try_local_search(n, m, p, target_rank, deadline, max_iters=3000)
        if result is not None:
            U, V, W = result
            if _verify_exact(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break  # Found a valid decomposition at this rank
    
    return best_result

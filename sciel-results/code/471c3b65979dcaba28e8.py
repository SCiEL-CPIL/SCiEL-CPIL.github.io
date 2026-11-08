import numpy as np
import time
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]


def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def _build_rank50_baseline():
    """Build the known rank-50 Strassen+schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    The multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    contributes to C[i,k]. Specifically, T[i,j,k] = 1 for all valid i,j,k (since
    C[i,k] = sum_j A[i,j]*B[j,k]).
    
    The decomposition gives: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Compute the tensor from decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Use einsum: 'ir,jr,kr->ijk'
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)

    # The true multiplication tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k], so each term A[i,j]*B[j,k] has coefficient 1.
    T_true = np.ones((n, m, p), dtype=float)

    diff = np.abs(T_decomp - T_true)
    return diff.max() < tol, diff.max()


def _project_to_half_integers(U, V, W, tol=1e-6):
    """Try to project the decomposition to half-integer values.
    
    For each column r, try to find a scaling that makes all entries half-integers.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    U_out = np.zeros_like(U_np)
    V_out = np.zeros_like(V_np)
    W_out = np.zeros_like(W_np)

    for r in range(rank):
        u = U_np[:, r]
        v = V_np[:, r]
        w = W_np[:, r]

        # Check if already half-integers
        all_vals = np.concatenate([u, v, w])
        # Multiply by 2 and check if close to integers
        doubled = all_vals * 2
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            U_out[:, r] = u
            V_out[:, r] = v
            W_out[:, r] = w
            continue

        # Try to find a common scale factor
        # The entries should be of the form a/2 for integer a
        # Try scaling by 1, 2, 0.5, etc.
        found = False
        for scale in [1.0, 2.0, 0.5, 4.0, 0.25]:
            u_s = u * scale
            v_s = v * scale
            w_s = w / scale  # Keep the product invariant: (u*scale)*(v*scale)*(w/scale^2) ... 
            # Actually, we need U*V*W to be invariant. If we scale U by s and V by t, 
            # we must scale W by 1/(s*t).
            # Let's try: U' = s*U, V' = t*V, W' = W/(s*t)
            # For half-integer: s*U, t*V, W/(s*t) should all be half-integers.
            pass

        # Simpler approach: just check if the current values work as-is
        # The baseline uses only integers, so they're already half-integers
        U_out[:, r] = u
        V_out[:, r] = v
        W_out[:, r] = w

    return U_out, V_out, W_out


def _hosvd_rank_reveal(U, V, W, n, m, p, max_rank_to_try=50):
    """Use HOSVD to find the effective rank of the tensor and try to reduce the decomposition.
    
    The key insight: if the tensor has a lower intrinsic rank, we can find a 
    lower-rank decomposition. HOSVD gives us the best rank-r approximation in 
    the Frobenius norm, but we need exact decomposition.
    
    Strategy: 
    1. Compute the tensor T from the current decomposition
    2. Try HOSVD with decreasing rank to see if we can exactly represent T
    3. If the residual is zero (within tolerance) at some rank r, we have a valid decomposition
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = len(U[0])

    # Compute the true tensor
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)

    # Try ranks from low to high
    for r in range(1, min(rank, max_rank_to_try) + 1):
        if _time_remaining() < 10:
            break

        try:
            # HOSVD: decompose T into rank-r components
            # Reshape T to (n, m*p) and (m, n*p) and (p, n*m)
            T_nm_p = T.reshape(n, m * p)
            T_m_np = T.reshape(m, n * p)
            T_p_nm = T.reshape(p, n * m)

            # SVD for each mode
            U_svd, s_U, Vt_U = np.linalg.svd(T_nm_p, full_matrices=False)
            U_svd, s_V, Vt_V = np.linalg.svd(T_m_np, full_matrices=False)
            U_svd_p, s_W, Vt_W = np.linalg.svd(T_p_nm, full_matrices=False)

            # Take top r components
            U_r = U_svd[:, :r]  # n x r
            V_r = U_svd[:, :r]  # m x r  (from mode-2 SVD)
            W_r = U_svd_p[:, :r]  # p x r  (from mode-3 SVD)

            # Scale to get the correct tensor
            # T ≈ sum_r U_r[:,r] * V_r[:,r] * W_r[:,r] * s_r
            # But the scaling is tricky. Let's use a different approach.
            
            # Actually, for exact decomposition, HOSVD won't directly help unless
            # the tensor is exactly rank-r. Let's check the residual.
            
            # Build the rank-r approximation
            T_approx = np.zeros((n, m, p), dtype=float)
            for rr in range(r):
                T_approx += np.outer(np.outer(U_r[:, rr], V_r[:, rr]), W_r[:, rr]) * (s_U[rr] * s_V[rr] * s_W[rr] / (s_U[0] * s_V[0] * s_W[0]))
            
            # This scaling is not quite right. Let's use a proper HOSVD implementation.
            # For now, let's just check if the tensor is exactly representable at lower rank
            # by checking the singular values.
            
            # Check if the (r+1)-th singular value is (approximately) zero
            if r < min(n, m * p) and r < len(s_U):
                if abs(s_U[r]) < 1e-10:
                    # The tensor might be exactly rank-r
                    # Try to extract the decomposition
                    pass
        except Exception:
            continue

    return None  # No reduction found


def _greedy_parafac_reduce(U, V, W, n, m, p, target_rank, max_iters=100):
    """Try to reduce the rank using a greedy approach.
    
    Start with the full decomposition and try to remove components that are
    linear combinations of others, or merge components.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    current_rank = len(U[0])

    if current_rank <= target_rank:
        return U, V, W

    # Try to find linear dependencies among the columns
    # The decomposition is: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # If we can find coefficients c_r such that sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0,
    # then we can eliminate one component.
    
    # For now, this is a placeholder for a more sophisticated algorithm.
    # The baseline rank-50 is likely already minimal or near-minimal.
    return U, V, W


def _try_block_rearrangement():
    """Try different block arrangements to see if we can achieve lower rank.
    
    The current arrangement uses:
    - 2 Strassen 2x2x2 blocks (7 each) for j=0,1 and j=2,3 with k=0,1,2,3,4,5
    - 2 schoolbook 2x2x1 blocks (4 each) for k=6
    
    Total: 2*7 + 2*4 = 22... wait, that's only 22, not 50.
    
    Let me recount: for each j_start in [0, 2]:
    - 3 Strassen blocks (k=0,2,4): 3*7 = 21
    - 1 schoolbook block (k=6): 4
    Total per j_start: 25
    Total: 2*25 = 50. Yes, that's correct.
    
    Alternative: try different splits of the 7 output columns.
    """
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = None, None, None
    best_rank = 50

    # Try different ways to split the 7 columns into groups
    # Current: [2, 2, 2, 1] -> 3 Strassen + 1 schoolbook per j-block
    # Try: [2, 2, 1, 1, 1] -> 2 Strassen + 3 schoolbook
    # Try: [1, 1, 1, 1, 1, 1, 1] -> 7 schoolbook
    # Try: [2, 1, 1, 1, 1, 1] -> 1 Strassen + 5 schoolbook
    
    splits = [
        [2, 2, 2, 1],  # current
        [2, 2, 1, 1, 1],
        [2, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [3, 2, 2],  # 3x2x2 would need different algorithm
    ]

    for split in splits:
        if _time_remaining() < 15:
            break

        rank = 0
        U = [[0] * 100 for _ in range(n * m)]  # Pre-allocate with extra space
        V = [[0] * 100 for _ in range(m * p)]
        W = [[0] * 100 for _ in range(p * n)]
        col = 0

        for j_start in [0, 2]:
            k_pos = 0
            for group_size in split:
                if group_size == 2:
                    # Strassen 2x2x2
                    k0 = k_pos
                    # ... (same as before)
                    a_idx = 0 * m + j_start
                    b_idx = 0 * m + (j_start + 1)
                    c_idx = 1 * m + j_start
                    d_idx = 1 * m + (j_start + 1)
                    e_idx = j_start * p + k0
                    f_idx = j_start * p + (k0 + 1)
                    g_idx = (j_start + 1) * p + k0
                    h_idx = (j_start + 1) * p + (k0 + 1)
                    c00_idx = k0 * n + 0
                    c01_idx = (k0 + 1) * n + 0
                    c10_idx = k0 * n + 1
                    c11_idx = (k0 + 1) * n + 1

                    # M1
                    U[a_idx][col] += 1; U[d_idx][col] += 1
                    V[e_idx][col] += 1; V[h_idx][col] += 1
                    W[c00_idx][col] += 1; W[c11_idx][col] += 1
                    col += 1
                    # M2
                    U[c_idx][col] += 1; U[d_idx][col] += 1
                    V[e_idx][col] += 1
                    W[c10_idx][col] += 1; W[c11_idx][col] -= 1
                    col += 1
                    # M3
                    U[a_idx][col] += 1
                    V[f_idx][col] += 1; V[h_idx][col] -= 1
                    W[c01_idx][col] += 1; W[c11_idx][col] += 1
                    col += 1
                    # M4
                    U[d_idx][col] += 1
                    V[g_idx][col] += 1; V[e_idx][col] -= 1
                    W[c00_idx][col] += 1; W[c10_idx][col] += 1
                    col += 1
                    # M5
                    U[a_idx][col] += 1; U[b_idx][col] += 1
                    V[h_idx][col] += 1
                    W[c00_idx][col] -= 1; W[c01_idx][col] += 1
                    col += 1
                    # M6
                    U[c_idx][col] += 1; U[a_idx][col] -= 1
                    V[e_idx][col] += 1; V[f_idx][col] += 1
                    W[c11_idx][col] += 1
                    col += 1
                    # M7
                    U[b_idx][col] += 1; U[d_idx][col] -= 1
                    V[g_idx][col] += 1; V[h_idx][col] += 1
                    W[c00_idx][col] += 1
                    col += 1
                    rank += 7
                elif group_size == 1:
                    # Schoolbook 2x2x1
                    k0 = k_pos
                    a_idx = 0 * m + j_start
                    b_idx = 0 * m + (j_start + 1)
                    c_idx = 1 * m + j_start
                    d_idx = 1 * m + (j_start + 1)
                    e_idx = j_start * p + k0
                    g_idx = (j_start + 1) * p + k0
                    c00_idx = k0 * n + 0
                    c10_idx = k0 * n + 1

                    U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
                    col += 1
                    U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
                    col += 1
                    U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
                    col += 1
                    U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
                    col += 1
                    rank += 4
                k_pos += group_size

        # Trim to actual rank
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]

        # Verify
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        if valid and rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W

    if best_U is not None:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS

    # Step 1: Build the baseline rank-50 decomposition
    U, V, W = _build_rank50_baseline()

    # Step 2: Try to find a lower-rank decomposition
    # First, try different block arrangements
    if _time_remaining() > 30:
        better = _try_block_rearrangement()
        if better is not None:
            U, V, W = better

    # Step 3: Verify the final decomposition
    valid, max_err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fall back to the baseline if something went wrong
        U, V, W = _build_rank50_baseline()

    rank = len(U[0])

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

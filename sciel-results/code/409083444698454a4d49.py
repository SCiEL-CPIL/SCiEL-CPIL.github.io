import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook rank (vectorized construction)
    rank_sb = n * m * p
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    col_indices = np.arange(rank_sb)
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat

    U_sb = np.zeros((nm, rank_sb), dtype=np.float64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.float64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.float64)
    U_sb[u_rows, col_indices] = 1
    V_sb[v_rows, col_indices] = 1
    W_sb[w_rows, col_indices] = 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap entries to nearest half-integer if close."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def snap_to_integer(arr, tol=1e-6):
        """Snap entries to nearest integer if close."""
        snapped = np.round(arr)
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_solve_for_W(U, V, rank):
        """Given U and V, try to solve for W such that the decomposition holds."""
        W = np.zeros((pn, rank))
        valid = True

        for i in range(n):
            for k in range(p):
                # Build A_ik: m x rank matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]

                # Solve A_ik @ w = ones(m)
                try:
                    w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                        valid = False
                        break
                    W[k * n + i, :] = w
                except:
                    valid = False
                    break

        if not valid:
            return None
        return W

    def random_search_for_rank(rank, num_trials, entry_set, seed=None):
        """Random search: generate random U, V from entry_set, solve for W, snap and verify."""
        if seed is not None:
            np.random.seed(seed)
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice(entry_set, size=(nm, rank))
            V = np.random.choice(entry_set, size=(mp, rank))

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            # Try snapping to half-integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def structured_search_for_rank(rank, num_trials):
        """Structured search using patterns that exploit the small n=2 structure."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            # Generate U with structure: rows for i=0 and i=1 might have relationships
            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))

            # Try to make some rows of U equal or negative of each other
            # Row i*m+j for i=0 and i=1
            if np.random.random() < 0.4:
                # Make U[0, :] = U[m, :] or U[0, :] = -U[m, :]
                if np.random.random() < 0.5:
                    U[m, :] = U[0, :].copy()
                else:
                    U[m, :] = -U[0, :].copy()
            
            # Also try making rows within same i block related
            if np.random.random() < 0.3:
                U[1, :] = U[0, :].copy()
            if np.random.random() < 0.3:
                U[2, :] = -U[1, :].copy()

            W = try_solve_for_W(U, V, rank)
            if W is None:
                continue

            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue

            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped

        return None, None, None

    def systematic_search(rank, entry_sets, max_trials_per_set=30):
        """Try multiple entry sets systematically."""
        for entry_set in entry_sets:
            if time.time() > deadline:
                return None, None, None
            
            U_found, V_found, W_found = random_search_for_rank(
                rank, max_trials_per_set, entry_set
            )
            if U_found is not None:
                return U_found, V_found, W_found
        
        return None, None, None

    # Search for lower ranks, starting from rank_sb-1 down
    # We'll try a range of ranks and different search strategies

    # Entry sets to try
    entry_sets = [
        [-1, 0, 1],
        [0, 1],
        [-1, 0, 1, 2],
        [-2, -1, 0, 1, 2]
    ]

    # First pass: try structured search for ranks close to schoolbook
    for rank in range(rank_sb - 1, max(30, rank_sb - 15), -1):
        if time.time() > deadline:
            break

        # Try structured search
        U_found, V_found, W_found = structured_search_for_rank(rank, 20)
        if U_found is not None:
            best_U, best_V, best_W = U_found, V_found, W_found
            best_rank = rank
            break

    # Second pass: try broader random search with multiple entry sets
    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(25, rank_sb - 25), -1):
            if time.time() > deadline:
                break

            # Try random search with different entry sets
            U_found, V_found, W_found = systematic_search(rank, entry_sets[:2], 15)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

    # Third pass: if still at schoolbook, try more aggressive search
    if best_rank == rank_sb:
        for rank in range(rank_sb - 1, max(20, rank_sb - 35), -1):
            if time.time() > deadline:
                break

            # Try all entry sets with more trials
            U_found, V_found, W_found = systematic_search(rank, entry_sets, 10)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                break

    # Final verification and return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {"U": U_ret, "V": V_ret, "W": W_ret}

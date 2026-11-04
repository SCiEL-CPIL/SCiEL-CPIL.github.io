"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS  # 4, 4, 5
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20


def _build_tensor(U, V, W):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]."""
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor[i, j, k] = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
    return tensor


def _is_valid(U, V, W, tol=1e-9):
    """Check if factors exactly decompose the all-ones tensor."""
    try:
        t = _build_tensor(U, V, W)
        if np.allclose(t, np.ones((n, m, p)), atol=tol):
            return True
    except Exception:
        pass
    return False


def _construct_strassen_rank72():
    """Deterministic Strassen-based decomposition with rank 72."""
    rank_total = 72
    U = np.zeros((NM, rank_total))
    V = np.zeros((MP, rank_total))
    W = np.zeros((PN, rank_total))
    state = [0]

    def add_strassen_2x2(ra, ca, rb, cb):
        col = state[0]
        # M1: (a+d)*(e+h)
        U[2*ra * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M2: (c+d)*e
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = -1.0
        state[0] += 1

        col = state[0]
        # M3: a*(f-h)
        U[2*ra * m + 2*ca, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M4: d*(g-e)
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[2*rb * p + 2*cb, col] = -1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M5: (a+b)*h
        U[2*ra * m + 2*ca, col] = 1.0
        U[2*ra * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        state[0] += 1

        col = state[0]
        # M6: (c-a)*(e+f)
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[2*ra * m + 2*ca, col] = -1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M7: (b-d)*(g+h)
        U[2*ra * m + (2*ca+1), col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = -1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        state[0] += 1

    def add_direct_2x2_2x1(ra, ca, rb, cb_col):
        for i in range(2):
            for j in range(2):
                col = state[0]
                U[(2*ra + i) * m + (2*ca + j), col] = 1.0
                V[(2*rb + j) * p + cb_col, col] = 1.0
                W[cb_col * n + (2*ra + i), col] = 1.0
                state[0] += 1

    # Block 0: B cols 0,1 (cb=0)
    add_strassen_2x2(0, 0, 0, 0)
    add_strassen_2x2(0, 1, 1, 0)
    add_strassen_2x2(1, 0, 0, 0)
    add_strassen_2x2(1, 1, 1, 0)

    # Block 1: B cols 2,3 (cb=1)
    add_strassen_2x2(0, 0, 0, 1)
    add_strassen_2x2(0, 1, 1, 1)
    add_strassen_2x2(1, 0, 0, 1)
    add_strassen_2x2(1, 1, 1, 1)

    # Block 2: B col 4 (cb_col=4)
    add_direct_2x2_2x1(0, 0, 0, 4)
    add_direct_2x2_2x1(0, 1, 1, 4)
    add_direct_2x2_2x1(1, 0, 0, 4)
    add_direct_2x2_2x1(1, 1, 1, 4)

    used_rank = state[0]
    U = U[:, :used_rank].tolist()
    V = V[:, :used_rank].tolist()
    W = W[:, :used_rank].tolist()
    return U, V, W


def _construct_schoolbook():
    """Schoolbook decomposition with rank 80."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _random_candidate(rank, entry_set, seed=None):
    """Generate a random candidate of given rank."""
    if seed is not None:
        random.seed(seed)
    U = [[random.choice(entry_set) for _ in range(rank)] for _ in range(NM)]
    V = [[random.choice(entry_set) for _ in range(rank)] for _ in range(MP)]
    W = [[random.choice(entry_set) for _ in range(rank)] for _ in range(PN)]
    return U, V, W


def _structured_low_rank_attempt(rank, deadline):
    """Try structured approaches for lower rank."""
    # Try various structured patterns
    entry_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
    ]
    
    for entry_set in entry_sets:
        if time.time() > deadline - 5:
            break
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            U, V, W = _random_candidate(rank, entry_set)
            if _is_valid(U, V, W):
                return U, V, W
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Start with deterministic Strassen rank-72 solution
    U_str, V_str, W_str = _construct_strassen_rank72()
    best_rank = 72
    best_solution = {"U": U_str, "V": V_str, "W": W_str}

    if not _is_valid(U_str, V_str, W_str):
        # Fallback to schoolbook if Strassen fails
        U_sb, V_sb, W_sb = _construct_schoolbook()
        best_rank = 80
        best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Search for lower rank solutions
    # Theoretical lower bound considerations:
    # For 4x4 * 4x5, known best constructions suggest rank ~ 30-50
    # Try ranks from 20 up to best_rank - 1
    
    min_rank_to_try = 20
    max_rank_to_try = best_rank - 1

    # Allocate search budget: more time for lower ranks
    time_per_rank = 8  # seconds per rank value
    entry_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
    ]

    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break

        remaining = deadline - time.time() - 10
        if remaining <= 0:
            break

        # Number of trials scales with remaining time
        num_trials = min(500, int(remaining * 50))
        num_trials = max(num_trials, 50)

        found = False
        for entry_set in entry_sets:
            if time.time() > deadline - 10:
                break
            for trial in range(num_trials // len(entry_sets)):
                if time.time() > deadline - 10:
                    break
                U, V, W = _random_candidate(r, entry_set)
                if _is_valid(U, V, W):
                    best_rank = r
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
            if found:
                break

        if found:
            # Found a solution at rank r, continue searching for lower
            continue

    # If we still have time, try more aggressive search at the best found rank
    if best_rank > 25 and time.time() < deadline - 15:
        for trial in range(200):
            if time.time() > deadline - 15:
                break
            for entry_set in entry_sets:
                U, V, W = _random_candidate(best_rank - 1, entry_set)
                if _is_valid(U, V, W):
                    best_rank -= 1
                    best_solution = {"U": U, "V": V, "W": W}
                    break

    return best_solution

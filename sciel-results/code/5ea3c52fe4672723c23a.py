"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _build_block_decomposition(n=3, m=3, p=6):
    """
    Build a decomposition by splitting k into two groups of 3.
    For each group, use a rank-r decomposition for 3x3x3 matrix mult.
    Share U factors across groups.
    """
    nm, mp, pn = n * m, m * p, p * n
    total_rank = n * m * p  # Start with schoolbook
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _compute_tensor_error(U, V, W, n=3, m=3, p=6, check_offdiag=True, sample_size=200):
    """Compute the total tensor reconstruction error."""
    nm, mp, pn = n * m, m * p, p * n
    err = 0.0
    
    # On-diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (val - 1.0) ** 2
    
    # Off-diagonal (sampled for speed)
    if check_offdiag:
        if sample_size is None or sample_size >= nm * mp * pn:
            # Check all
            for a in range(nm):
                i_a, j_a = a // m, a % m
                for b in range(mp):
                    j_b, k_b = b // p, b % p
                    for c in range(pn):
                        k_c, i_c = c // n, c % n
                        if not (j_a == j_b and k_b == k_c and i_c == i_a):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += val ** 2
        else:
            rng = np.random.RandomState(42)
            for _ in range(sample_size):
                a = rng.randint(0, nm)
                b = rng.randint(0, mp)
                c = rng.randint(0, pn)
                i_a, j_a = a // m, a % m
                j_b, k_b = b // p, b % p
                k_c, i_c = c // n, c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += val ** 2
    
    return err


def _als_step(U, V, W, n=3, m=3, p=6):
    """One ALS iteration: update U, V, W in sequence."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # Update U
    new_U = np.zeros_like(U)
    for a in range(nm):
        i_a, j_a = a // m, a % m
        rows = []
        for k in range(p):
            b_idx = j_a * p + k
            c_idx = k * n + i_a
            rows.append(V[b_idx, :] * W[c_idx, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_U[a, :] = U[a, :]
    U = new_U
    
    # Update V
    new_V = np.zeros_like(V)
    for b in range(mp):
        j_b, k_b = b // p, b % p
        rows = []
        for i in range(n):
            a_idx = i * m + j_b
            c_idx = k_b * n + i
            rows.append(U[a_idx, :] * W[c_idx, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_V[b, :] = V[b, :]
    V = new_V
    
    # Update W
    new_W = np.zeros_like(W)
    for c in range(pn):
        k_c, i_c = c // n, c % n
        rows = []
        for j in range(m):
            a_idx = i_c * m + j
            b_idx = j * p + k_c
            rows.append(U[a_idx, :] * V[b_idx, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_W[c, :] = W[c, :]
    W = new_W
    
    return U, V, W


def _try_rank(r, n=3, m=3, p=6, deadline=None, n_restarts=5, seed_base=0):
    """Try to find a rank-r decomposition using ALS with multiple restarts."""
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    best_err = float('inf')
    best_result = None
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(seed_base + restart * 1000)
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Sparse random with structure
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for col in range(r):
                # Each term should contribute to a specific (i,j,k) pattern
                # Try to make sparse
                n_u = rng.randint(1, min(4, nm))
                n_v = rng.randint(1, min(4, mp))
                n_w = rng.randint(1, min(4, pn))
                u_idx = rng.choice(nm, n_u, replace=False)
                v_idx = rng.choice(mp, n_v, replace=False)
                w_idx = rng.choice(pn, n_w, replace=False)
                U[u_idx, col] = rng.choice([-1, 1], n_u)
                V[v_idx, col] = rng.choice([-1, 1], n_v)
                W[w_idx, col] = rng.choice([-1, 1], n_w)
        elif restart % 4 == 1:
            # Dense random
            U = rng.randn(nm, r) * 0.3
            V = rng.randn(mp, r) * 0.3
            W = rng.randn(pn, r) * 0.3
        elif restart % 4 == 2:
            # Structured: U depends on i, V on j, W on k
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for a in range(nm):
                i_a = a // m
                U[a, :] = rng.randn(r) * (1.0 + 0.5 * i_a)
            for b in range(mp):
                j_b = b // p
                V[b, :] = rng.randn(r) * (1.0 + 0.5 * j_b)
            for c in range(pn):
                k_c = c // n
                W[c, :] = rng.randn(r) * (1.0 + 0.5 * k_c)
        else:
            # Near-schoolbook perturbation
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            # Assign terms to approximate schoolbook structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1.0 + rng.randn() * 0.1
                        V[b, col] = 1.0 + rng.randn() * 0.1
                        W[c, col] = 1.0 + rng.randn() * 0.1
                        col += 1
                    if col >= r:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        # ALS iterations
        max_iter = 100
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            U, V, W = _als_step(U, V, W, n, m, p)
            
            # Compute error (sampled for speed)
            err = _compute_tensor_error(U, V, W, n, m, p, check_offdiag=True, sample_size=100)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                # Verify exactly
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            if abs(prev_err - err) < 1e-14 and it > 5:
                break
            prev_err = err
    
    return best_result


def _try_polish(U, V, W, n=3, m=3, p=6, deadline=None, max_iter=200):
    """High-precision polish pass with tighter tolerance."""
    if deadline is None:
        deadline = time.time() + 50
    
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    prev_err = float('inf')
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        U, V, W = _als_step(U, V, W, n, m, p)
        
        err = _compute_tensor_error(U, V, W, n, m, p, check_offdiag=True, sample_size=None)
        
        if err < 1e-15:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        
        if abs(prev_err - err) < 1e-16 and it > 10:
            break
        prev_err = err
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    best_rank = n * m * p
    best_result = result
    
    # Try to find better decompositions at lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But due to structure, rank is likely higher.
    # Let's try from rank 20 upward, with more restarts for lower ranks
    
    # Allocate time budget
    time_per_rank = 8  # seconds per rank attempt
    
    for r in range(20, min(54, 45)):  # Try ranks 20 to 44
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # More restarts for lower ranks (harder to find)
        n_restarts = max(3, int(remaining / time_per_rank / 2))
        
        found = _try_rank(r, n, m, p, deadline=deadline, n_restarts=n_restarts, seed_base=r * 100)
        
        if found is not None:
            # Verify
            if _verify(found["U"], found["V"], found["W"], n, m, p):
                best_result = found
                best_rank = r
                # Try to polish
                U_arr = np.array(found["U"], dtype=float)
                V_arr = np.array(found["V"], dtype=float)
                W_arr = np.array(found["W"], dtype=float)
                polished = _try_polish(U_arr, V_arr, W_arr, n, m, p, deadline=deadline)
                if polished is not None:
                    best_result = polished
                break  # Found a valid decomposition at this rank, stop
    
    return best_result
# EVOLVE-BLOCK-END

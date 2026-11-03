"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _build_system_for_rank(n, m, p, R, U, V):
    """
    Build the linear system A @ W_flat = b for solving W.
    U: (n*m, R), V: (m*p, R)
    Returns (A, b, valid_indices) where A is (num_constraints, R*p*n), b is (num_constraints,)
    """
    nm, mp, pn = n*m, m*p, p*n
    total_constraints = nm * mp * pn
    A = np.zeros((total_constraints, R * pn))
    b = np.zeros(total_constraints)
    
    idx = 0
    for x in range(pn):
        i = x % n
        k = x // n
        for y in range(nm):
            j = y % m
            i2 = y // m
            for z in range(mp):
                j2 = z // p
                k2 = z % p
                # Row idx of A
                row = idx
                if i == i2 and j == j2 and k == k2:
                    b[row] = 1.0
                else:
                    b[row] = 0.0
                
                # For each rank component r, W[x, r] appears in the sum
                # sum_r U[y,r] * V[z,r] * W[x,r]
                # So A[row, r*p*n + x] = U[y,r] * V[z,r]
                for r in range(R):
                    A[row, r * pn + x] = U[y, r] * V[z, r]
                idx += 1
    
    return A, b

def _try_solve_for_W(n, m, p, R, U, V, tol=1e-9):
    """
    Try to solve for W given fixed U and V.
    Returns W if solution exists and is valid, else None.
    """
    nm, mp, pn = n*m, m*p, p*n
    A, b = _build_system_for_rank(n, m, p, R, U, V)
    
    # Solve the linear system
    try:
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return None
    
    # Check if solution is exact
    if len(residuals) > 0 and residuals[0] > tol:
        return None
    
    W = W_flat.reshape(R, pn).T  # (p*n, R)
    
    # Verify the solution
    if not _verify_decomposition(U, V, W, n, m, p, tol):
        return None
    
    return W

def _generate_sparse_UV(n, m, p, R, density=0.15, seed=None):
    """
    Generate random sparse U and V matrices with entries in {0, ±1}.
    """
    if seed is not None:
        random.seed(seed)
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    
    # Generate U
    for i in range(n*m):
        for r in range(R):
            if random.random() < density:
                U[i, r] = random.choice([-1, 1])
    
    # Generate V
    for i in range(m*p):
        for r in range(R):
            if random.random() < density:
                V[i, r] = random.choice([-1, 1])
    
    return U, V

def _snap_to_half_integer(x, tol=1e-6):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _snap_matrix(M, tol=1e-6):
    """Snap all entries of matrix to nearest half-integer if close."""
    return np.array([[_snap_to_half_integer(x) for x in row] for row in M])

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank 54 solution
    base_sol = _build_54_rank(n, m, p)
    base_U, base_V, base_W = base_sol["U"], base_sol["V"], base_sol["W"]
    
    # Verify base solution
    if not _verify_decomposition(base_U, base_V, base_W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = base_U.shape[1]
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()
    
    # Try structured random linear combination strategy
    # Try ranks from 53 down to 30
    for R in range(best_rank - 1, 29, -1):
        if _time_left() < 10:
            break
        
        # Try multiple random configurations
        max_attempts = 50 if R >= 45 else 100
        for attempt in range(max_attempts):
            if _time_left() < 5:
                break
            
            seed = attempt * 1000 + R
            U_rand, V_rand = _generate_sparse_UV(n, m, p, R, density=0.15, seed=seed)
            
            # Check if U and V have enough rank
            if np.linalg.matrix_rank(U_rand) < R:
                continue
            if np.linalg.matrix_rank(V_rand) < R:
                continue
            
            W_sol = _try_solve_for_W(n, m, p, R, U_rand, V_rand)
            if W_sol is not None:
                # Found a valid decomposition with rank R
                best_rank = R
                best_U, best_V, best_W = U_rand, V_rand, W_sol
                break  # Move to next (lower) rank
    
    # Snap to half-integers if possible
    U_snap = _snap_matrix(best_U)
    V_snap = _snap_matrix(best_V)
    W_snap = _snap_matrix(best_W)
    
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

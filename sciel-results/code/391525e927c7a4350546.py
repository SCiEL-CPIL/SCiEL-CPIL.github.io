"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
import scipy.linalg as la
from itertools import product

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _snap_to_integers(matrix):
    """Snap matrix entries to nearest integer."""
    return np.round(matrix)

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _als_step(U, V, W, n, m, p, fix='U'):
    """
    Perform one step of Alternating Least Squares.
    Fix two matrices, solve for the third to minimize residual.
    """
    r = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    if fix == 'U':
        for i in range(n):
            for j in range(m):
                A = np.zeros(r)
                for k in range(p):
                    A += V[j*p+k] * W[k*n+i]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    U_new[i*m+j] = A / norm_sq
                else:
                    U_new[i*m+j] = 0
                    
    elif fix == 'V':
        for j in range(m):
            for k in range(p):
                A = np.zeros(r)
                for i in range(n):
                    A += U[i*m+j] * W[k*n+i]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    V_new[j*p+k] = A / norm_sq
                else:
                    V_new[j*p+k] = 0

    elif fix == 'W':
        for k in range(p):
            for i in range(n):
                A = np.zeros(r)
                for j in range(m):
                    A += U[i*m+j] * V[j*p+k]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    W_new[k*n+i] = A / norm_sq
                else:
                    W_new[k*n+i] = 0

    if fix == 'U':
        return U_new, V, W
    elif fix == 'V':
        return U, V_new, W
    else:
        return U, V, W_new

def _als_optimize(n, m, p, r, U0, V0, W0, max_iters=200, tol=1e-12):
    """
    Run Alternating Least Squares to minimize the residual.
    """
    U, V, W = U0.copy(), V0.copy(), W0.copy()
    
    prev_err = float('inf')
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Cycle through fixing U, V, W
        U, V, W = _als_step(U, V, W, n, m, p, fix='U')
        U, V, W = _als_step(U, V, W, n, m, p, fix='V')
        U, V, W = _als_step(U, V, W, n, m, p, fix='W')
        
        # Calculate error
        err = _compute_residual(U, V, W, n, m, p)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _try_structured_initialization(n, m, p, r):
    """
    Generate structured initializations that respect the tensor structure.
    """
    initializations = []
    
    # Strategy 1: Random but with structure
    for _ in range(5):
        U0 = np.zeros((n*m, r))
        V0 = np.zeros((m*p, r))
        W0 = np.zeros((p*n, r))
        
        # Assign each column to a random "type" that determines its pattern
        for c in range(r):
            # Choose a random subset of i indices
            i_subset = np.random.choice(n, size=np.random.randint(1, n+1), replace=False)
            j_subset = np.random.choice(m, size=np.random.randint(1, m+1), replace=False)
            k_subset = np.random.choice(p, size=np.random.randint(1, p+1), replace=False)
            
            # Assign non-zero values to these indices
            for i in i_subset:
                for j in j_subset:
                    U0[i*m+j, c] = np.random.choice([-1, 1])
            for j in j_subset:
                for k in k_subset:
                    V0[j*p+k, c] = np.random.choice([-1, 1])
            for k in k_subset:
                for i in i_subset:
                    W0[k*n+i, c] = np.random.choice([-1, 1])
        
        initializations.append((U0, V0, W0))
    
    # Strategy 2: Pure random with small values
    for _ in range(5):
        U0 = np.random.choice([-1, 0, 1], size=(n*m, r))
        V0 = np.random.choice([-1, 0, 1], size=(m*p, r))
        W0 = np.random.choice([-1, 0, 1], size=(p*n, r))
        initializations.append((U0, V0, W0))
        
    # Strategy 3: All ones (for rank 1 test)
    if r == 1:
        U0 = np.ones((n*m, 1))
        V0 = np.ones((m*p, 1))
        W0 = np.ones((p*n, 1))
        initializations.append((U0, V0, W0))

    return initializations

def _try_rank_r(n, m, p, r, max_restarts=30):
    """
    Try to find a rank-r decomposition using structured and random restarts.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Generate structured initializations
    initializations = _try_structured_initialization(n, m, p, r)
    
    # Add more random initializations
    for _ in range(max_restarts - len(initializations)):
        U0 = np.random.choice([-1, 0, 1], size=(n*m, r))
        V0 = np.random.choice([-1, 0, 1], size=(m*p, r))
        W0 = np.random.choice([-1, 0, 1], size=(p*n, r))
        initializations.append((U0, V0, W0))
    
    for U0, V0, W0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        U_sol, V_sol, W_sol = _als_optimize(n, m, p, r, U0, V0, W0)
        
        # Calculate final error
        err = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            # Check for exact solution after snapping
            # Try half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
            if ok:
                return U_snap, V_snap, W_snap
                
            # Try integers
            U_snap2 = _snap_to_integers(U_sol)
            V_snap2 = _snap_to_integers(V_sol)
            W_snap2 = _snap_to_integers(W_sol)
            ok2, _ = _verify_decomposition(U_snap2, V_snap2, W_snap2, n, m, p)
            if ok2:
                return U_snap2, V_snap2, W_snap2
                
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try ranks from 1 up to 35
    for r in range(1, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        # Try structured search first
        result = _try_rank_r(n, m, p, r, max_restarts=30)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

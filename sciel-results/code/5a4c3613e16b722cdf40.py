"""Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _verify_exact(n, m, p, U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_float(n, m, p, U, V, W, atol=1e-9):
    """Quick float verification."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T_decomp[i*m+j, j*p+k, k*n+i] - 1.0) > atol:
                    return False
    return True

def _snap_to_half(arr):
    """Snap entries to nearest 0.5."""
    return np.round(np.array(arr) * 2) / 2

def _als_decompose(n, m, p, rank, U_init, V_init, W_init, max_iter=100, tol=1e-10):
    """Alternating Least Squares for tensor decomposition."""
    U = np.array(U_init, dtype=float).copy()
    V = np.array(V_init, dtype=float).copy()
    W = np.array(W_init, dtype=float).copy()
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    for iteration in range(max_iter):
        # Update U: for each column a, solve (V^T W^T) U[:,a] = target_a
        # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # For fixed a: sum_t U[a,t] (V[b,t] W[c,t]) = T[a,b,c]
        # Let M_b_c = V[b,:] * W[c,:] (outer product per t, summed)
        # Actually: for fixed a, we need sum_t U[a,t] * (V[:,t] * W[:,t])_flattened = T[a, :, :].flatten()
        
        # Build the linear system for U[:,a]
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # For fixed a: sum_t U[a,t] * (V[b,t] * W[c,t]) = T[a,b,c]
        # The "feature" for variable U[a,t] is V[b,t]*W[c,t]
        # So we have: sum_t U[a,t] * F_t[b,c] = T[a,b,c] where F_t[b,c] = V[b,t]*W[c,t]
        
        # This is a least squares problem: A @ U_a = b
        # where A[t, (b,c)] = V[b,t] * W[c,t]
        # and b[(b,c)] = T[a,b,c]
        
        # Precompute A
        # A has shape (rank, size_v * size_w)
        # A[t, b*size_w + c] = V[b,t] * W[c,t]
        
        A = np.zeros((rank, size_v * size_w))
        for t in range(rank):
            for b in range(size_v):
                for c in range(size_w):
                    A[t, b * size_w + c] = V[b, t] * W[c, t]
        
        # Target tensor
        T = np.zeros((size_u, size_v, size_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Solve for U
        U_new = np.zeros_like(U)
        for a in range(size_u):
            b_vec = T[a, :, :].flatten()
            # Solve A @ U_a = b_vec (least squares)
            try:
                U_a, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                U_new[a, :] = U_a
            except:
                U_new[a, :] = U[a, :]
        
        # Update V
        B_mat = np.zeros((rank, size_u * size_w))
        for t in range(rank):
            for a in range(size_u):
                for c in range(size_w):
                    B_mat[t, a * size_w + c] = U_new[a, t] * W[c, t]
        
        V_new = np.zeros_like(V)
        for b in range(size_v):
            b_vec = np.zeros(size_u * size_w)
            for a in range(size_u):
                for c in range(size_w):
                    b_vec[a * size_w + c] = T[a, b, c]
            try:
                V_a, _, _, _ = np.linalg.lstsq(B_mat, b_vec, rcond=None)
                V_new[b, :] = V_a
            except:
                V_new[b, :] = V[b, :]
        
        # Update W
        C_mat = np.zeros((rank, size_u * size_v))
        for t in range(rank):
            for a in range(size_u):
                for b in range(size_v):
                    C_mat[t, a * size_v + b] = U_new[a, t] * V_new[b, t]
        
        W_new = np.zeros_like(W)
        for c in range(size_w):
            b_vec = np.zeros(size_u * size_v)
            for a in range(size_u):
                for b in range(size_v):
                    b_vec[a * size_v + b] = T[a, b, c]
            try:
                W_a, _, _, _ = np.linalg.lstsq(C_mat, b_vec, rcond=None)
                W_new[c, :] = W_a
            except:
                W_new[c, :] = W[c, :]
        
        # Check convergence
        U_diff = np.max(np.abs(U_new - U))
        V_diff = np.max(np.abs(V_new - V))
        W_diff = np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        
        if max(U_diff, V_diff, W_diff) < tol:
            break
    
    return U, V, W

def _try_rank_als(n, m, p, rank, num_trials=50, seed=42):
    """Try to find a rank-r decomposition using ALS."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    random.seed(seed)
    np.random.seed(seed)
    
    best_result = None
    best_loss = float('inf')
    
    for trial in range(num_trials):
        # Initialize with random values
        U_init = np.random.randn(size_u, rank) * 0.5
        V_init = np.random.randn(size_v, rank) * 0.5
        W_init = np.random.randn(size_w, rank) * 0.5
        
        # Run ALS
        U, V, W = _als_decompose(n, m, p, rank, U_init, V_init, W_init, max_iter=50)
        
        # Check if it's exact
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Build target
        T = np.zeros((size_u, size_v, size_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        loss = np.sum((T_decomp - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_result = (U.copy(), V.copy(), W.copy())
        
        # If loss is very small, try to snap to half-integers
        if loss < 1e-6:
            U_s = _snap_to_half(U)
            V_s = _snap_to_half(V)
            W_s = _snap_to_half(W)
            
            T_decomp_s = np.einsum('at,bt,ct->abc', U_s, V_s, W_s)
            loss_s = np.sum((T_decomp_s - T) ** 2)
            
            if loss_s < 1e-9:
                # Convert to lists with half-integer values
                U_list = U_s.tolist()
                V_list = V_s.tolist()
                W_list = W_s.tolist()
                
                # Verify with exact arithmetic
                if _verify_exact(n, m, p, U_list, V_list, W_list):
                    return U_list, V_list, W_list
    
    return None

def _structured_init(n, m, p, rank):
    """Create structured initializations based on the problem structure."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    inits = []
    
    # Initialization 1: Based on schoolbook with merging
    # Group some (i,j,k) triples
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        
        # Assign random signs to a subset of entries
        for c in range(rank):
            # Random support
            for i in range(n):
                for j in range(m):
                    if random.random() < 0.8:
                        U[i*m+j, c] = random.choice([1, -1])
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.8:
                        V[j*p+k, c] = random.choice([1, -1])
            for k in range(p):
                for i in range(n):
                    if random.random() < 0.8:
                        W[k*n+i, c] = random.choice([1, -1])
        
        inits.append((U, V, W))
    
    # Initialization 2: Try to use the algebraic structure
    # For each j, the computation involves A[:,j] and B[j,:]
    # We can try to factor through shared intermediates
    for seed in range(5):
        random.seed(seed + 100)
        np.random.seed(seed + 100)
        
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        
        # Assign structured patterns
        for c in range(rank):
            # Pattern based on c
            pattern = c % 3
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = 1 if (i + j) % 3 == pattern else 0
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = 1 if (j + k) % 3 == pattern else 0
            for k in range(p):
                for i in range(n):
                    W[k*n+i, c] = 1 if (k + i) % 3 == pattern else 0
        
        inits.append((U, V, W))
    
    return inits

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook fallback
    U_fb = [[0]*schoolbook_rank for _ in range(n*m)]
    V_fb = [[0]*schoolbook_rank for _ in range(m*p)]
    W_fb = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    # Try ranks from 24 to 35
    for rank in range(24, 36):
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        # Try ALS with multiple initializations
        result = _try_rank_als(n, m, p, rank, num_trials=30, seed=rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            if rank <= 25:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

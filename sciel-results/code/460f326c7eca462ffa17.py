import numpy as np
from scipy.optimize import minimize
import time
from itertools import product
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W):
    """Verify exact decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
    n, m, p = DIMENSIONS
    if len(U) == 0:
        return False
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    try:
        U_f = [[Fraction(str(x)) for x in row] for row in U]
        V_f = [[Fraction(str(x)) for x in row] for row in V]
        W_f = [[Fraction(str(x)) for x in row] for row in W]
    except:
        # Try as floats
        U_f = U
        V_f = V
        W_f = W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0) if isinstance(U_f[0][0], Fraction) else 0.0
                for c in range(rank):
                    if isinstance(U_f[0][0], Fraction):
                        s += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                    else:
                        s += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1.0) > 1e-10:
                        return False
    return True

def _is_half_integer(val, tol=1e-9):
    """Check if value is a half-integer (multiple of 0.5)."""
    scaled = val * 2
    rounded = round(scaled)
    return abs(scaled - rounded) < tol

def _snap_to_half_int(val):
    """Snap value to nearest half-integer."""
    return round(val * 2) / 2.0

def _compute_tensor_error(U, V, W, n=2, m=3, p=4):
    """Compute S[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] and return error norm."""
    rank = len(U[0])
    S = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    return np.sum((S - 1.0)**2)

def _local_search_discrete(U, V, W, n=2, m=3, p=4, max_iter=2000, tol=1e-10):
    """Local search over discrete half-integer values to minimize error."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0])
    
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    # Snap initial values
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    
    best_error = _compute_tensor_error(U, V, W, n, m, p)
    
    # Random coordinate perturbation
    np.random.seed(42)
    for it in range(max_iter):
        # Pick a random matrix and entry
        mat_choice = np.random.choice(3)  # 0=U, 1=V, 2=W
        if mat_choice == 0:
            mat = U
            rows, cols = nm, rank
        elif mat_choice == 1:
            mat = V
            rows, cols = mp, rank
        else:
            mat = W
            rows, cols = pn, rank
        
        r_idx = np.random.randint(rows)
        c_idx = np.random.randint(cols)
        
        old_val = mat[r_idx, c_idx]
        # Try neighboring half-integers
        neighbors = [old_val + d for d in [-1.0, -0.5, 0.5, 1.0] if abs(old_val + d) <= 5.0]
        
        best_neighbor = None
        best_neighbor_error = best_error
        
        for nv in neighbors:
            nv = round(nv * 2) / 2.0
            mat[r_idx, c_idx] = nv
            err = _compute_tensor_error(U, V, W, n, m, p)
            if err < best_neighbor_error:
                best_neighbor_error = err
                best_neighbor = nv
            mat[r_idx, c_idx] = old_val
        
        if best_neighbor is not None and best_neighbor_error < best_error - 1e-12:
            mat[r_idx, c_idx] = best_neighbor
            best_error = best_neighbor_error
            
            if best_error < tol:
                if mat_choice == 0:
                    U = mat
                elif mat_choice == 1:
                    V = mat
                else:
                    W = mat
                return U, V, W, True
        elif it > 100 and np.random.random() < 0.05:
            # Occasional larger perturbation to escape local minima
            mat[r_idx, c_idx] = np.random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            mat[r_idx, c_idx] = round(mat[r_idx, c_idx] * 2) / 2.0
    
    return U, V, W, False

def _structured_init(n, m, p, rank):
    """Create structured initialization based on block decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Try to exploit structure: for each (i,j) pair in A, we need contributions
    # to all k in C[i,k]. Similarly for each (j,k) in B.
    
    # Initialize with small random values
    np.random.seed(123)
    U[:] = np.random.uniform(-1, 1, (nm, rank))
    V[:] = np.random.uniform(-1, 1, (mp, rank))
    W[:] = np.random.uniform(-1, 1, (pn, rank))
    
    # Snap to half-integers for discrete search
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    
    return U, V, W

def _optimize_continuous(U, V, W, n=2, m=3, p=4, max_iter=1000):
    """Use scipy to minimize the continuous objective."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0])
    
    def objective_and_grad(x):
        U_c = x[:nm*rank].reshape(nm, rank)
        V_c = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W_c = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        S = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_c[:, c].reshape(n, m)
            v_c = V_c[:, c].reshape(m, p)
            w_c = W_c[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        error = S - 1.0
        loss = np.sum(error**2)
        
        # Gradient
        grad_U = np.zeros_like(U_c)
        grad_V = np.zeros_like(V_c)
        grad_W = np.zeros_like(W_c)
        
        for c in range(rank):
            u_c = U_c[:, c].reshape(n, m)
            v_c = V_c[:, c].reshape(m, p)
            w_c = W_c[:, c].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    s_val = 0.0
                    for k in range(p):
                        s_val += error[i,j,k] * v_c[j,k] * w_c[k,i]
                    grad_U[i*m+j, c] = 2 * s_val
                    
            for j in range(m):
                for k in range(p):
                    s_val = 0.0
                    for i in range(n):
                        s_val += error[i,j,k] * u_c[i,j] * w_c[k,i]
                    grad_V[j*p+k, c] = 2 * s_val
                    
            for k in range(p):
                for i in range(n):
                    s_val = 0.0
                    for j in range(m):
                        s_val += error[i,j,k] * u_c[i,j] * v_c[j,k]
                    grad_W[k*n+i, c] = 2 * s_val
                    
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        
        x_opt = res.x
        U_opt = x_opt[:nm*rank].reshape(nm, rank)
        V_opt = x_opt[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W_opt = x_opt[nm*rank+mp*rank:].reshape(pn, rank)
        
        return U_opt, V_opt, W_opt, res.fun
    except Exception as e:
        return U, V, W, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Schoolbook solution (rank 24)
    U_school = np.zeros((nm, n*m*p))
    V_school = np.zeros((mp, n*m*p))
    W_school = np.zeros((pn, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n*m*p
    
    # Try ranks from low to high
    for rank in range(19, 24):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(rank * 1000)
        
        found = False
        max_attempts = 8 if rank <= 20 else 4
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization
            U_init, V_init, W_init = _structured_init(n, m, p, rank)
            
            # Continuous optimization first
            U_cont, V_cont, W_cont, loss = _optimize_continuous(U_init, V_init, W_init, n, m, p, max_iter=300)
            
            if loss < 1e-6:
                # Snap to half-integers and verify
                U_snap = np.round(U_cont * 2) / 2.0
                V_snap = np.round(V_cont * 2) / 2.0
                W_snap = np.round(W_cont * 2) / 2.0
                
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    best_rank = rank
                    found = True
                    break
                else:
                    # Try local search from snapped values
                    U_ls, V_ls, W_ls, success = _local_search_discrete(U_snap, V_snap, W_snap, n, m, p, max_iter=3000)
                    if success:
                        if _verify(U_ls.tolist(), V_ls.tolist(), W_ls.tolist()):
                            best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                            best_rank = rank
                            found = True
                            break
            
            # Also try direct discrete search from random init
            U_rand = np.random.uniform(-2, 2, (nm, rank))
            V_rand = np.random.uniform(-2, 2, (mp, rank))
            W_rand = np.random.uniform(-2, 2, (pn, rank))
            U_rand = np.round(U_rand * 2) / 2.0
            V_rand = np.round(V_rand * 2) / 2.0
            W_rand = np.round(W_rand * 2) / 2.0
            
            U_ls, V_ls, W_ls, success = _local_search_discrete(U_rand, V_rand, W_rand, n, m, p, max_iter=2000)
            if success:
                if _verify(U_ls.tolist(), V_ls.tolist(), W_ls.tolist()):
                    best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result

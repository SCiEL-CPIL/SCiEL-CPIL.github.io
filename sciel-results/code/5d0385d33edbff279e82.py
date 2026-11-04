import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build target tensor for verification
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Part 2: 4x4x3 for k=4,5,6 using schoolbook (rank 48)
    U_3, V_3, W_3 = _schoolbook_4x4x3(n, m, p, k_start=4)
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Combine
    U_all = np.hstack([U_4, U_3])
    V_all = np.hstack([V_4, V_3])
    W_all = np.hstack([W_4, W_3])
    
    rank = U_all.shape[1]
    
    # Verify
    T_decomp = np.einsum('ar,br,cr->abc', U_all, V_all, W_all)
    
    if not np.allclose(T_decomp, T_target, atol=1e-10):
        return _schoolbook(n, m, p)
    
    # Try to reduce rank via Gaussian elimination
    if time.time() < deadline - 20:
        U_red, V_red, W_red = _reduce_rank_gauss(U_all, V_all, W_all, T_target, deadline)
        if U_red is not None:
            T_red = np.einsum('ar,br,cr->abc', U_red, V_red, W_red)
            if np.allclose(T_red, T_target, atol=1e-10):
                U_all, V_all, W_all = U_red, V_red, W_red
                rank = U_all.shape[1]
    
    return {
        "U": U_all.tolist(),
        "V": V_all.tolist(),
        "W": W_all.tolist()
    }


def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(7):
        for s in range(7):
            u = np.kron(U2[:, r], U2[:, s])
            
            v_kron = np.kron(V2[:, r], V2[:, s])
            v_full = np.zeros(28)
            for j in range(4):
                for k in range(4):
                    v_full[j * 7 + k] = v_kron[j * 4 + k]
            
            w_kron = np.kron(W2[:, r], W2[:, s])
            w_full = np.zeros(28)
            w_full[:16] = w_kron
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W


def _schoolbook_4x4x3(n, m, p, k_start=4):
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(k_start, p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _reduce_rank_gauss(U, V, W, T_target, deadline):
    """Reduce rank via Gaussian elimination on the decomposition."""
    rank = U.shape[1]
    dim_u, dim_v, dim_w = U.shape[0], V.shape[0], W.shape[0]
    
    # Convert to Fraction for exact arithmetic
    U_frac = np.array([[Fraction(U[i, r]) for r in range(rank)] for i in range(dim_u)])
    V_frac = np.array([[Fraction(V[i, r]) for r in range(rank)] for i in range(dim_v)])
    W_frac = np.array([[Fraction(W[i, r]) for r in range(rank)] for i in range(dim_w)])
    
    keep = list(range(rank))
    
    # Try to remove each column by expressing it as a linear combination of others
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if r not in keep:
            continue
        
        # Try to express column r as a linear combination of other kept columns
        # We need to find alpha_s such that:
        # U[:, r] = sum_{s in keep, s != r} alpha_s * U[:, s]
        # V[:, r] = sum_{s in keep, s != r} alpha_s * V[:, s]  
        # W[:, r] = sum_{s in keep, s != r} alpha_s * W[:, s]
        
        # This is a system of dim_u + dim_v + dim_w equations with (len(keep)-1) unknowns
        # But we need the same alpha_s for all three, so it's overdetermined
        
        # Build the system: for each row i in U, j in V, k in W
        # The constraint is: U[i,r] - sum_s alpha_s * U[i,s] = 0
        #                   V[j,r] - sum_s alpha_s * V[j,s] = 0
        #                   W[k,r] - sum_s alpha_s * W[k,s] = 0
        
        other_cols = [s for s in keep if s != r]
        num_unknowns = len(other_cols)
        
        if num_unknowns == 0:
            continue
        
        # Build coefficient matrix and RHS
        # Each row corresponds to one equation
        # For U: dim_u equations
        # For V: dim_v equations  
        # For W: dim_w equations
        
        num_eqs = dim_u + dim_v + dim_w
        A_mat = np.zeros((num_eqs, num_unknowns), dtype=object)
        rhs = np.zeros(num_eqs, dtype=object)
        
        for idx, s in enumerate(other_cols):
            # U part
            for i in range(dim_u):
                A_mat[i, idx] = -U_frac[i, s]
            rhs[i] = U_frac[i, r]
            
            # V part
            for j in range(dim_v):
                A_mat[dim_u + j, idx] = -V_frac[j, s]
            rhs[dim_u + j] = V_frac[j, r]
            
            # W part
            for k in range(dim_w):
                A_mat[dim_u + dim_v + k, idx] = -W_frac[k, s]
            rhs[dim_u + dim_v + k] = W_frac[k, r]
        
        # Solve the overdetermined system using least squares over rationals
        # This is complex, so let's use a simpler approach:
        # Try to find a subset of equations that gives a consistent solution
        
        # For now, use a heuristic: try to solve using a subset of equations
        # and verify the solution works for all equations
        
        # Use Gaussian elimination on a subset
        # Pick the first min(num_unknowns, num_eqs) equations
        subset_size = min(num_unknowns, num_eqs)
        
        # Build the system for the subset
        A_sub = A_mat[:subset_size, :]
        b_sub = rhs[:subset_size]
        
        # Solve using Gaussian elimination
        solution = _gauss_solve(A_sub, b_sub)
        
        if solution is not None:
            # Verify the solution works for all equations
            valid = True
            for eq_idx in range(num_eqs):
                lhs = Fraction(0)
                for idx in range(num_unknowns):
                    lhs += A_mat[eq_idx, idx] * solution[idx]
                if lhs != rhs[eq_idx]:
                    valid = False
                    break
            
            if valid:
                keep.remove(r)
    
    if len(keep) >= rank:
        return None, None, None
    
    # Build reduced matrices
    U_red = U_frac[:, keep]
    V_red = V_frac[:, keep]
    W_red = W_frac[:, keep]
    
    U_out = np.array([[float(U_red[i, j]) for j in range(len(keep))] for i in range(dim_u)])
    V_out = np.array([[float(V_red[i, j]) for j in range(len(keep))] for i in range(dim_v)])
    W_out = np.array([[float(W_red[i, j]) for j in range(len(keep))] for i in range(dim_w)])
    
    return U_out, V_out, W_out


def _gauss_solve(A, b):
    """Solve Ax = b using Gaussian elimination over rationals.
    
    Returns solution vector or None if no unique solution.
    """
    n = len(b)
    m = A.shape[1]
    
    if n < m:
        return None
    
    # Augmented matrix
    M = np.zeros((n, m + 1), dtype=object)
    for i in range(n):
        for j in range(m):
            M[i, j] = A[i, j]
        M[i, m] = b[i]
    
    # Forward elimination
    rank = 0
    for col in range(m):
        # Find pivot
        pivot = -1
        for row in range(rank, n):
            if M[row, col] != 0:
                pivot = row
                break
        
        if pivot == -1:
            continue
        
        # Swap rows
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
        
        # Normalize pivot row
        pivot_val = M[rank, col]
        for j in range(col, m + 1):
            M[rank, j] = M[rank, j] / pivot_val
        
        # Eliminate below
        for row in range(rank + 1, n):
            if M[row, col] != 0:
                factor = M[row, col]
                for j in range(col, m + 1):
                    M[row, j] = M[row, j] - factor * M[rank, j]
        
        rank += 1
    
    if rank < m:
        return None
    
    # Back substitution
    x = np.zeros(m, dtype=object)
    for i in range(m - 1, -1, -1):
        x[i] = M[i, m]
        for j in range(i + 1, m):
            x[i] -= M[i, j] * x[j]
    
    return x

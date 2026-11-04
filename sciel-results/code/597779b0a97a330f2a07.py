# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    # Double centering: B = -1/2 J M J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    # Eigendecomposition (B symmetric)
    w, V = np.linalg.eigh(B)
    # Take largest k eigenvalues
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    # X = V * sqrt(w)
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian-like greedy.
    Actually solve the assignment: minimize sum_{i} ||X[i] - Y[p[i]]||^2.
    This is a linear assignment problem on the cost matrix C[i,j] = ||X[i]-Y[j]]||^2.
    Use scipy's linear_sum_assignment."""
    n = X.shape[0]
    # C[i,j] = ||X[i] - Y[j]]||^2 = ||X[i]]||^2 - 2 X[i].Y[j] + ||Y[j]]||^2
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    # linear_sum_assignment minimizes sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _two_opt_search(a, b, p, time_limit, max_iter=500000):
    """Best Improvement 2-opt local search. p is numpy int64 array (facility i -> location p[i])."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # Precompute RC and CC for fast delta computation
    RC = b[:, p].astype(np.float64)  # shape (n, n)
    CC = b[p, :].astype(np.float64)  # shape (n, n)
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Find best improvement swap
        # We can vectorize the delta calculation for a fixed i against all j > i
        for i in range(n):
            li = p[i]
            a_i = a_float[i, :]
            
            # drow for all j: RC[:, p[j]] - RC[:, li] is not right.
            # drow_ij = RC[p[j], :] - RC[li, :]
            # dcol_ij = CC[:, p[j]] - CC[:, li]
            
            # Let's iterate j to keep memory low, but check time
            for j in range(i + 1, n):
                lj = p[j]
                a_j = a_float[j, :]
                
                # drow = RC[lj, :] - RC[li, :]
                # dcol = CC[:, lj] - CC[:, li]
                # delta = (a_i - a_j) @ drow + (a_float[:, i] - a_float[:, j]) @ dcol
                
                # Optimization: Precompute a_diff_row and a_diff_col? 
                # They depend on i, j.
                
                drow = RC[lj, :] - RC[li, :]
                dcol = CC[:, lj] - CC[:, li]
                
                a_diff_row = a_i - a_j
                a_diff_col = a_float[:, i] - a_float[:, j]
                
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
        
        # Apply best swap
        i, j = best_i, best_j
        li, lj = p[i], p[j]
        p[i], p[j] = lj, li
        cost += best_delta
        
        # Update RC and CC
        RC[:, i] = b_float[:, lj]
        RC[:, j] = b_float[:, li]
        CC[i, :] = b_float[lj, :]
        CC[j, :] = b_float[li, :]
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p, best_cost


def _simulated_annealing(a, b, p, time_limit):
    """Simulated Annealing with 2-opt moves."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # Precompute RC and CC
    RC = b[:, p].astype(np.float64)
    CC = b[p, :].astype(np.float64)
    
    start = time.time()
    
    # Temperature parameters
    # Estimate typical delta magnitude
    # Just use a heuristic start temperature
    T = 1000.0 
    T_min = 1.0
    cooling = 0.99
    
    while T > T_min and (time.time() - start) < time_limit:
        # Try a random swap
        i, j = np.random.choice(n, 2, replace=False)
        if i == j:
            continue
            
        li, lj = p[i], p[j]
        
        # Calculate delta
        drow = RC[lj, :] - RC[li, :]
        dcol = CC[:, lj] - CC[:, li]
        a_diff_row = a_float[i, :] - a_float[j, :]
        a_diff_col = a_float[:, i] - a_float[:, j]
        delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
        
        # Acceptance
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept move
            p[i], p[j] = lj, li
            cost += delta
            
            RC[:, i] = b_float[:, lj]
            RC[:, j] = b_float[:, li]
            CC[i, :] = b_float[lj, :]
            CC[j, :] = b_float[li, :]
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 115  # safety margin before 120s

    # Step 1: MDS embedding + Procrustes alignment
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    # Increased dimension to capture more structure
    k = min(30, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: 2-opt Local Search
    time_left = deadline - time.time()
    if time_left > 10:
        p, cost = _two_opt_search(a, b, p, time_left - 5)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    # Step 3: Simulated Annealing if time remains
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _simulated_annealing(a, b, best_p, time_left - 1)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END

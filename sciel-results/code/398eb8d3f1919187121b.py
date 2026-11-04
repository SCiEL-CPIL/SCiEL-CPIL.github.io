import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140

def _get_strassen_2x2x2():
    """
    Solve for 7 rank-1 tensors that decompose the 2x2x2 all-ones tensor.
    Returns a list of 7 tuples (u, v, w) where u,v,w are numpy arrays of shape (2,).
    """
    n_terms = 7
    # Unknowns: 
    # u_t[1] for t in 0..6 (7 vars, u_t[0] fixed to 1)
    # v_t[0], v_t[1] for t in 0..6 (14 vars)
    # w_t[0], w_t[1] for t in 0..6 (14 vars)
    # Total 35 vars.
    
    def func(x):
        # Residuals for the 8 equations
        res = np.zeros(8)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0.0
                    for t in range(n_terms):
                        u_i = 1.0 if i == 0 else x[t]
                        v_j = x[7 + 2*t + j]
                        w_k = x[21 + 2*t + k]
                        total += u_i * v_j * w_k
                    res[i*4 + j*2 + k] = total - 1.0
        return res

    # Initial guess
    x0 = np.zeros(35)
    for t in range(n_terms):
        x0[t] = 0.0 # u_t[1]
        x0[7 + 2*t] = 1.0/n_terms # v_t[0]
        x0[7 + 2*t + 1] = 1.0/n_terms # v_t[1]
        x0[21 + 2*t] = 1.0 # w_t[0]
        x0[21 + 2*t + 1] = 1.0 # w_t[1]
        
    # This guess gives sum = 7 * (1/7) * 1 * 1 = 1. So it's a valid starting point (schoolbook-like spread).
    
    sol = least_squares(func, x0, method='lm') # Levenberg-Marquardt
    
    if sol.success:
        tensors = []
        for t in range(n_terms):
            u = np.array([1.0, sol.x[t]])
            v = np.array([sol.x[7 + 2*t], sol.x[7 + 2*t + 1]])
            w = np.array([sol.x[21 + 2*t], sol.x[21 + 2*t + 1]])
            tensors.append((u, v, w))
        return tensors
    else:
        # Fallback to schoolbook (rank 8) if solver fails
        # This will increase rank but ensure validity
        print("Strassen solver failed, falling back to schoolbook")
        return None

def construct():
    n, m, p = DIMENSIONS
    
    i_groups = [(0, 1), (2, 3)]
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4,)]
    
    strassen_tensors = _get_strassen_2x2x2()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i_group in i_groups:
        for j_group in j_groups:
            for k_group in k_groups:
                i_size = len(i_group)
                j_size = len(j_group)
                k_size = len(k_group)
                
                if i_size == 2 and j_size == 2 and k_size == 2 and strassen_tensors is not None:
                    i0, i1 = i_group
                    j0, j1 = j_group
                    k0, k1 = k_group
                    
                    for (u, v, w) in strassen_tensors:
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        
                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(2):
                                    i_idx = i_group[ii]
                                    j_idx = j_group[jj]
                                    k_idx = k_group[kk]
                                    
                                    U_dict[(i_idx, j_idx)] = u[ii]
                                    V_dict[(j_idx, k_idx)] = v[jj]
                                    W_dict[(k_idx, i_idx)] = w[kk]
                        
                        U_cols.append(U_dict)
                        V_cols.append(V_dict)
                        W_cols.append(W_dict)
                else:
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                U_dict = {(i, j): 1.0}
                                V_dict = {(j, k): 1.0}
                                W_dict = {(k, i): 1.0}
                                U_cols.append(U_dict)
                                V_cols.append(V_dict)
                                W_cols.append(W_dict)
    
    rank = len(U_cols)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col in range(rank):
        for (i, j), val in U_cols[col].items():
            U[i * m + j, col] = val
        for (j, k), val in V_cols[col].items():
            V[j * p + k, col] = val
        for (k, i), val in W_cols[col].items():
            W[k * n + i, col] = val
    
    return {"U": U, "V": V, "W": W}

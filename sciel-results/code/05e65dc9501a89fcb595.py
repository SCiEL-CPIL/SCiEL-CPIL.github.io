<<<<<<< SEARCH
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s
    
    # Precompute support grid and state mapping
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}
    
    # Precompute value vectors
    v1_vec = np.array([s[0] for s in support_grid], dtype=float)
    v2_vec = np.array([s[1] for s in support_grid], dtype=float)
    
    # Precompute IC constraint matrix (same for all distributions)
    # IC: p_i - p_j <= v_i·x_i - v_i·x_j
    # Rearranged: p_i - p_j - v1_i*x1_i - v2_i*x2_i + v1_i*x1_j + v2_i*x2_j <= 0
    n_vars = 3 * n_states
    ic_rows = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_vec[i]
            row[n_states + i] = -v2_vec[i]
            row[j] = v1_vec[i]
            row[n_states + j] = v2_vec[i]
            ic_rows.append(row)
    A_ic = np.array(ic_rows) if ic_rows else np.empty((0, n_vars))
    b_ic = np.zeros(len(ic_rows))
    
    # Precompute IR constraint matrix
    # IR: p_i - v1_i*x1_i - v2_i*x2_i <= 0
    ir_rows = []
    for i in range(n_states):
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_vec[i]
        row[n_states + i] = -v2_vec[i]
        ir_rows.append(row)
    A_ir = np.array(ir_rows)
    b_ir = np.zeros(n_states)
    
    # Precompute bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states
=======
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s
    
    # Precompute support grid and state mapping
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}
    
    # Precompute value vectors
    v1_vec = np.array([s[0] for s in support_grid], dtype=float)
    v2_vec = np.array([s[1] for s in support_grid], dtype=float)
    v_full = np.column_stack((v1_vec, v2_vec))  # shape (16, 2)
>>>>>>> REPLACE

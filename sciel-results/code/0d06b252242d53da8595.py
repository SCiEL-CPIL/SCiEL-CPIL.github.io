import numpy as np
import time
import random
import itertools

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks exploiting the shared A matrix structure.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: output columns k in {0,1}, rank indices 0..6
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[2][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = -1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[2][6] = -1
    
    # Block 2: output columns k in {2,3}, rank indices 7..13
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[6][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[4][9] = 1; W[5][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[6][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[5][11] = 1; W[7][11] = -1
    
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[6][12] = 1; W[4][12] = -1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[5][13] = 1; W[6][13] = -1
    
    return {"U": U, "V": V, "W": W}

def _peel_rank_search(max_rank=13, max_iters=5000, deadline=None):
    """
    Attempt to find a rank-R decomposition using a peeling-like heuristic.
    We model the tensor T[i,j,k] = 1 (for 2x2x4) as a sum of R rank-1 terms.
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    Strategy:
    1. Start with a random sparse guess for U, V, W.
    2. Iteratively fix two matrices (e.g., U, V) and solve for the third (W) 
       using least squares or direct linear algebra on the flattened tensor.
       However, since T is fixed to 1, we are solving for U,V,W such that the 
       reconstruction matches 1.
       
    A more effective peeling approach for small tensors:
    - Assume a specific sparsity pattern for the rank-1 terms.
    - For rank 13, we might have 13 terms. 
    - We can use a local search on the coefficients, but restricted to 
      half-integers or integers to ensure exactness.
      
    Given the difficulty of general peeling, we will try a structured 
    "merging" of two Strassen terms.
    In the rank-14 solution, we have two independent blocks.
    Can we merge one term from Block 1 and one from Block 2?
    Term r1: (u1, v1, w1)
    Term r2: (u2, v2, w2)
    If we can find u3, v3, w3 such that 
    u1 v1 w1 + u2 v2 w2 = u3 v3 w3 + u4 v4 w4 + ... (fewer terms)
    This is generally hard.
    
    Instead, we perform a randomized search over a restricted space of 
    "almost-Strassen" decompositions where we allow small perturbations 
    in the coefficients (half-integers) to see if rank 13 is possible.
    
    We will use a simulated annealing approach on the error of the 
    decomposition, minimizing sum (T_recon - 1)^2.
    """
    n, m, p = 2, 2, 4
    target_rank = max_rank
    
    # Pre-compute indices for T[i,j,k]
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..7)
    # W index: k*n + i (0..7)
    
    # We represent the state as a list of tuples (u, v, w) where 
    # u is len 4, v is len 8, w is len 8.
    # Values are restricted to {-1, -0.5, 0, 0.5, 1, 1.5, 2} to allow 
    # for potential half-integer solutions.
    
    def random_state():
        state = []
        for _ in range(target_rank):
            u = [random.choice([-1, 0, 1]) for _ in range(4)]
            v = [random.choice([-1, 0, 1]) for _ in range(8)]
            w = [random.choice([-1, 0, 1]) for _ in range(8)]
            state.append((u, v, w))
        return state

    def evaluate(state):
        # Calculate error: sum (T_recon[i,j,k] - 1)^2
        error = 0.0
        U = np.zeros((4, target_rank))
        V = np.zeros((8, target_rank))
        W = np.zeros((8, target_rank))
        
        for r in range(target_rank):
            u, v, w = state[r]
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    u_idx = 2*i + j
                    v_idx = 2*j + k
                    w_idx = 2*k + i
                    s = 0.0
                    for r in range(target_rank):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    error += (s - 1.0)**2
        return error

    def mutate(state):
        new_state = [list(s) for s in state]
        r = random.randint(0, target_rank-1)
        vec_idx = random.randint(0, 2)
        if vec_idx == 0:
            dim = 4
        elif vec_idx == 1:
            dim = 8
        else:
            dim = 8
            
        pos = random.randint(0, dim-1)
        
        old_val = new_state[r][vec_idx][pos]
        # Allow half-integers
        choices = [x for x in [-1, -0.5, 0, 0.5, 1, 1.5, 2] if x != old_val]
        new_val = random.choice(choices)
        new_state[r][vec_idx][pos] = new_val
        
        return [tuple(s) for s in new_state]

    best_state = None
    best_err = float('inf')
    
    # Start with a modified Strassen solution (remove one term, adjust others?)
    # Actually, random restarts are more robust for this small space if time-boxed.
    
    for restart in range(10):
        if deadline and time.time() > deadline:
            break
        state = random_state()
        err = evaluate(state)
        if err < best_err:
            best_err = err
            best_state = state
            if err == 0:
                break
        
        # Local search
        for step in range(200):
            if deadline and time.time() > deadline:
                break
            new_state = mutate(state)
            new_err = evaluate(new_state)
            if new_err < err:
                state = new_state
                err = new_err
                if err < best_err:
                    best_err = err
                    best_state = state
                    if err == 0:
                        break
            if err == 0:
                break
        if best_err == 0:
            break

    if best_err < 1e-9:
        U = [[0]*target_rank for _ in range(4)]
        V = [[0]*target_rank for _ in range(8)]
        W = [[0]*target_rank for _ in range(8)]
        for r in range(target_rank):
            u, v, w = best_state[r]
            for idx in range(4):
                U[idx][r] = u[idx]
            for idx in range(8):
                V[idx][r] = v[idx]
            for idx in range(8):
                W[idx][r] = w[idx]
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    
    # 1. Try to find a rank-13 solution via peeling-like search
    deadline_13 = start_time + 30.0
    sol_13 = _peel_rank_search(max_rank=13, deadline=deadline_13)
    if sol_13 is not None:
        if _verify_decomposition(sol_13['U'], sol_13['V'], sol_13['W'], n, m, p, tol=1e-9):
            return sol_13

    # 2. Try rank-12 (less likely, but check if we have time)
    if time.time() < start_time + 60.0:
        deadline_12 = start_time + 90.0
        sol_12 = _peel_rank_search(max_rank=12, deadline=deadline_12)
        if sol_12 is not None:
            if _verify_decomposition(sol_12['U'], sol_12['V'], sol_12['W'], n, m, p, tol=1e-9):
                return sol_12

    # 3. Fallback to rank-14 Strassen
    sol = _strassen_2x2x4()
    if _verify_decomposition(sol['U'], sol['V'], sol['W'], n, m, p, tol=1e-9):
        return sol

    # 4. Fallback to trivial rank-16
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

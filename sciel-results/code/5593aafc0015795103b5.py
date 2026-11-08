"""Search-based approach to find rank-minimal tensor decomposition for 3x3 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 3]

def build_target_tensor():
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                terms.append((u_idx, v_idx, w_idx))
    return terms

TARGET_TERMS = build_target_tensor()
# Pre-compute arrays for vectorized error computation
TARGET_U = np.array([t[0] for t in TARGET_TERMS], dtype=np.int32)
TARGET_V = np.array([t[1] for t in TARGET_TERMS], dtype=np.int32)
TARGET_W = np.array([t[2] for t in TARGET_TERMS], dtype=np.int32)

def compute_error_vec(U, V, W, rank):
    """Vectorized error computation. U,V,W are numpy arrays of shape (9, rank)."""
    # For each target term, compute sum_r U[u,r]*V[v,r]*W[w,r]
    # U[TARGET_U, :] has shape (27, rank), same for V, W
    u_vals = U[TARGET_U, :]  # (27, rank)
    v_vals = V[TARGET_V, :]  # (27, rank)
    w_vals = W[TARGET_W, :]  # (27, rank)
    # Sum over rank axis
    product = u_vals * v_vals * w_vals  # (27, rank)
    sums = product.sum(axis=1)  # (27,)
    err = np.sum((sums - 1.0) ** 2)
    return err

def validate_decomposition(U, V, W, rank):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    u_vals = U[TARGET_U, :]
    v_vals = V[TARGET_V, :]
    w_vals = W[TARGET_W, :]
    sums = (u_vals * v_vals * w_vals).sum(axis=1)
    return np.all(np.abs(sums - 1.0) < 1e-9)

def make_candidate(rank, U, V, W):
    return {'U': [list(row) for row in U], 'V': [list(row) for row in V], 'W': [list(row) for row in W]}

def random_candidate(rank):
    U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    return U, V, W

def mutate(U, V, W, rank, num_mutations=1):
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    for _ in range(num_mutations):
        matrix = random.choice([U, V, W])
        row = random.randint(0, 8)
        col = random.randint(0, rank - 1)
        if random.random() < 0.15:
            new_val = 0
        else:
            new_val = random.randint(-4, 4)
            if new_val == 0:
                new_val = random.choice([-1, 1])
        matrix[row][col] = new_val
    return U, V, W

def hill_climb(rank, U, V, W, max_steps=3000, deadline=None):
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_error = compute_error_vec(U_arr, V_arr, W_arr, rank)
    no_improve = 0
    max_no_improve = 80
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=1)
        U_new_arr = np.array(U_new, dtype=float)
        V_new_arr = np.array(V_new, dtype=float)
        W_new_arr = np.array(W_new, dtype=float)
        err_new = compute_error_vec(U_new_arr, V_new_arr, W_new_arr, rank)
        
        if err_new < best_error:
            U, V, W = U_new, V_new, W_new
            U_arr, V_arr, W_arr = U_new_arr, V_new_arr, W_new_arr
            best_error = err_new
            no_improve = 0
            if best_error < 1e-10:
                break
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Larger jump
                U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=random.randint(3, 6))
                U_new_arr = np.array(U_new, dtype=float)
                V_new_arr = np.array(V_new, dtype=float)
                W_new_arr = np.array(W_new, dtype=float)
                err_new = compute_error_vec(U_new_arr, V_new_arr, W_new_arr, rank)
                if err_new < best_error:
                    U, V, W = U_new, V_new, W_new
                    U_arr, V_arr, W_arr = U_new_arr, V_new_arr, W_new_arr
                    best_error = err_new
                    no_improve = 0
                else:
                    if random.random() < 0.05:
                        U, V, W = random_candidate(rank)
                        U_arr = np.array(U, dtype=float)
                        V_arr = np.array(V, dtype=float)
                        W_arr = np.array(W, dtype=float)
                        best_error = compute_error_vec(U_arr, V_arr, W_arr, rank)
                        no_improve = 0
    
    return U, V, W, best_error

def _known_rank26():
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # Strassen 2x2 block (top-left)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[4][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[4][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[3][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[4][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[3][col]=1; V[4][col]=1; W[0][col]=1; col+=1
    
    # Cross terms
    add_term(2,6,0,1); add_term(2,7,3,1); add_term(5,6,1,1); add_term(5,7,4,1)
    add_term(0,2,6,1); add_term(1,5,6,1); add_term(3,2,7,1); add_term(4,5,7,1)
    add_term(2,8,6,1); add_term(5,8,7,1)
    add_term(6,0,2,1); add_term(7,3,2,1); add_term(6,1,5,1); add_term(7,4,5,1)
    add_term(8,6,2,1); add_term(8,7,5,1)
    add_term(6,2,8,1); add_term(7,5,8,1)
    add_term(8,8,8,1)
    
    return {'U': U, 'V': V, 'W': W}

def _try_merge_from_rank26(target_rank, deadline, num_trials=200):
    """Try to merge columns of the rank-26 solution to reduce rank."""
    base = _known_rank26()
    U0 = [list(row) for row in base['U']]
    V0 = [list(row) for row in base['V']]
    W0 = [list(row) for row in base['W']]
    base_rank = 26
    merges_needed = base_rank - target_rank
    
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        
        U_try = [list(row) for row in U0]
        V_try = [list(row) for row in V0]
        W_try = [list(row) for row in W0]
        
        success = True
        for _ in range(merges_needed):
            cols = list(range(len(U_try[0])))
            if len(cols) < 2:
                success = False
                break
            c1, c2 = random.sample(cols, 2)
            for i in range(9):
                U_try[i][c1] += U_try[i][c2]
                V_try[i][c1] += V_try[i][c2]
                W_try[i][c1] += W_try[i][c2]
            for i in range(9):
                del U_try[i][c2]
                del V_try[i][c2]
                del W_try[i][c2]
        
        if not success or len(U_try[0]) != target_rank:
            continue
        
        U_try, V_try, W_try, err = hill_climb(target_rank, U_try, V_try, W_try, max_steps=400, deadline=deadline)
        if err < 1e-10:
            U_arr = np.array(U_try, dtype=float)
            V_arr = np.array(V_try, dtype=float)
            W_arr = np.array(W_try, dtype=float)
            if validate_decomposition(U_arr, V_arr, W_arr, target_rank):
                return make_candidate(target_rank, U_try, V_try, W_try)
    
    return None

def _try_direct_search(target_rank, deadline, num_trials=100):
    """Direct random search + hill climbing."""
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        
        # Structured sparse init
        U = [[0]*target_rank for _ in range(9)]
        V = [[0]*target_rank for _ in range(9)]
        W = [[0]*target_rank for _ in range(9)]
        
        for r in range(target_rank):
            u_nz = random.sample(range(9), k=random.randint(1, 4))
            v_nz = random.sample(range(9), k=random.randint(1, 4))
            w_nz = random.sample(range(9), k=random.randint(1, 4))
            for idx in u_nz:
                U[idx][r] = random.choice([-3,-2,-1,1,2,3])
            for idx in v_nz:
                V[idx][r] = random.choice([-3,-2,-1,1,2,3])
            for idx in w_nz:
                W[idx][r] = random.choice([-3,-2,-1,1,2,3])
        
        U, V, W, err = hill_climb(target_rank, U, V, W, max_steps=300, deadline=deadline)
        if err < 1e-10:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            if validate_decomposition(U_arr, V_arr, W_arr, target_rank):
                return make_candidate(target_rank, U, V, W)
    
    return None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    fallback = _known_rank26()
    best = None
    best_rank = 26
    
    # Try ranks from 25 downward
    for rank in [25, 24, 23]:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time: merge strategy gets 60%, direct search gets 40%
        merge_deadline = time.time() + remaining * 0.6
        result = _try_merge_from_rank26(rank, merge_deadline, num_trials=300)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _try_direct_search(rank, deadline, num_trials=80)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    if best is not None:
        return best
    return fallback
# EVOLVE-BLOCK-END

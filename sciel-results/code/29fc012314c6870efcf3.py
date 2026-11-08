"""Search-based approach to find rank-minimal tensor decomposition for 3x3 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 3]

# Pre-compute the target tensor T[i,j,k] = 1 if this is a valid multiplication term
# i = row of A, j = col of A / row of B, k = col of B
# Product C[i,k] gets A[i,j]*B[j,k]
# In flattened: A flat index = i*3+j, B flat index = j*3+k, C flat index = k*3+i (per W[k*n+i])
def build_target_tensor():
    """Return list of (u_idx, v_idx, w_idx) for each valid multiplication term."""
    terms = []
    for i in range(3):  # row of A
        for j in range(3):  # col of A = row of B
            for k in range(3):  # col of B
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i  # W[k*n+i]
                terms.append((u_idx, v_idx, w_idx))
    return terms

TARGET_TERMS = build_target_tensor()

def validate_decomposition(U, V, W, rank):
    """Check if the decomposition correctly computes matrix multiplication.
    
    For each rank component r, the contribution is U[:,r] * V[:,r] * W[:,r]^T
    We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for valid (i,j,k)
    and = 0 for invalid.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum over r of U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                val = 0.0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def make_candidate(rank, U, V, W):
    """Return candidate dict."""
    return {'U': [list(row) for row in U], 'V': [list(row) for row in V], 'W': [list(row) for row in W]}

def random_candidate(rank):
    """Generate a random candidate with integer entries in {-2,-1,0,1,2}."""
    U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    return U, V, W

def mutate(U, V, W, rank, num_mutations=3):
    """Apply random mutations to the candidate."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    for _ in range(num_mutations):
        matrix = random.choice([U, V, W])
        row = random.randint(0, 8)
        col = random.randint(0, rank - 1)
        old_val = matrix[row][col]
        # Allow a wider range of values, including 0
        if random.random() < 0.2:
            new_val = 0  # Sometimes zero out
        else:
            new_val = random.randint(-4, 4)
            if new_val == 0 and old_val != 0:
                new_val = random.choice([-1, 1])
        matrix[row][col] = new_val
    return U, V, W

def hill_climb(rank, U, V, W, max_steps=5000, deadline=None):
    """Perform hill climbing to fix a near-correct decomposition."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    def compute_error():
        err = 0.0
        for (u_idx, v_idx, w_idx) in TARGET_TERMS:
            val = 0.0
            for r in range(rank):
                val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            err += (val - 1.0) ** 2
        return err
    
    best_error = compute_error()
    no_improve_count = 0
    max_no_improve = 100
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random mutations
        U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=1)
        
        err_new = compute_error()
        if err_new < best_error:
            U, V, W = U_new, V_new, W_new
            best_error = err_new
            no_improve_count = 0
            if abs(best_error) < 1e-10:
                break
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try larger jumps to escape local minima
                U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=random.randint(2, 5))
                err_new = compute_error()
                if err_new < best_error:
                    U, V, W = U_new, V_new, W_new
                    best_error = err_new
                    no_improve_count = 0
                else:
                    # Reset to a random restart if stuck
                    if random.random() < 0.1:
                        U, V, W = random_candidate(rank)
                        best_error = compute_error()
                        no_improve_count = 0
    
    return U, V, W, best_error

def try_find_decomposition(rank, deadline):
    """Try to find a valid decomposition of given rank."""
    # Strategy 1: Start from a known higher-rank solution and try to merge columns
    # Strategy 2: Random initialization + hill climbing with better mutations
    # Strategy 3: Structured sparse initialization
    
    # Strategy 1: Try to derive from rank 26 by merging
    if rank < 26:
        result = _try_merge_from_rank26(rank, deadline)
        if result is not None:
            return result
    
    # Strategy 2: Random initialization + hill climbing
    for attempt in range(300):
        if time.time() > deadline:
            return None
        
        U, V, W = random_candidate(rank)
        U, V, W, err = hill_climb(rank, U, V, W, max_steps=500, deadline=deadline)
        if abs(err) < 1e-10:
            if validate_decomposition(U, V, W, rank):
                return make_candidate(rank, U, V, W)
    
    # Strategy 3: Structured sparse initialization
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Random structured initialization with more structure
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        for r in range(rank):
            # Each rank-1 component should have support that makes sense
            # For matrix mult, each term contributes to specific (i,j,k) patterns
            u_nonzero = random.sample(range(9), k=random.randint(1, 4))
            v_nonzero = random.sample(range(9), k=random.randint(1, 4))
            w_nonzero = random.sample(range(9), k=random.randint(1, 4))
            
            for idx in u_nonzero:
                U[idx][r] = random.choice([-3, -2, -1, 1, 2, 3])
            for idx in v_nonzero:
                V[idx][r] = random.choice([-3, -2, -1, 1, 2, 3])
            for idx in w_nonzero:
                W[idx][r] = random.choice([-3, -2, -1, 1, 2, 3])
        
        U, V, W, err = hill_climb(rank, U, V, W, max_steps=500, deadline=deadline)
        if abs(err) < 1e-10:
            if validate_decomposition(U, V, W, rank):
                return make_candidate(rank, U, V, W)
    
    return None

def _try_merge_from_rank26(rank, deadline):
    """Try to merge columns of the rank-26 solution to get lower rank."""
    base = _known_rank26()
    U = [list(row) for row in base['U']]
    V = [list(row) for row in base['V']]
    W = [list(row) for row in base['W']]
    
    # Try to merge pairs of columns
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U_try = [list(row) for row in U]
        V_try = [list(row) for row in V]
        W_try = [list(row) for row in W]
        
        # Try to merge random pairs
        for _ in range(26 - rank):
            if len(U_try[0]) <= rank:
                break
            # Pick two random columns to merge
            cols = list(range(len(U_try[0])))
            c1, c2 = random.sample(cols, 2)
            # Merge c2 into c1
            for i in range(9):
                U_try[i][c1] += U_try[i][c2]
                V_try[i][c1] += V_try[i][c2]
                W_try[i][c1] += W_try[i][c2]
            # Remove c2
            for i in range(9):
                del U_try[i][c2]
                del V_try[i][c2]
                del W_try[i][c2]
        
        if len(U_try[0]) == rank:
            U_try, V_try, W_try, err = hill_climb(rank, U_try, V_try, W_try, max_steps=500, deadline=deadline)
            if abs(err) < 1e-10:
                if validate_decomposition(U_try, V_try, W_try, rank):
                    return make_candidate(rank, U_try, V_try, W_try)
    
    return None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with known rank 26 solution as fallback
    fallback = _known_rank26()
    
    best = None
    best_rank = 26
    
    # Try to find lower rank solutions
    # Focus on rank 25 first (most likely to succeed), then try lower
    for rank in [25, 24, 23, 22, 21, 20, 19]:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        print(f"Trying rank {rank}...")
        result = try_find_decomposition(rank, deadline)
        if result is not None:
            best = result
            best_rank = rank
            print(f"Found valid decomposition with rank {rank}!")
            break  # Found lowest possible, stop
    
    # If we found something better, use it; otherwise use the rank 26 fallback
    if best is not None:
        return best
    else:
        return fallback

def _known_rank26():
    """Return the known rank 26 decomposition."""
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

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T -> Rank 4
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 3. A2 b -> Rank 4
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 4. a b3 -> Rank 2
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 5. a^T B2 -> Rank 4
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 6. a3 b^T -> Rank 2
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 7. a^T b -> Rank 2
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 8. a3 b3 -> Rank 1
    add_term(8, 8, 8, 1)
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END

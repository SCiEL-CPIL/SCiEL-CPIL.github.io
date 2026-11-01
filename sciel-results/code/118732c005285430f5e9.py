import time
import random

DIMENSIONS = [2, 3, 3]

def _deadline_guard(seconds=140.0):
    return time.time() + seconds

def _verify(U, V, W, n, m, p, rank):
    """Exact integer verification of the tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1 for all i,j,k."""
    U2 = [[2 * U[i][c] for c in range(rank)] for i in range(n * m)]
    V2 = [[2 * V[j][c] for c in range(rank)] for j in range(m * p)]
    W2 = [[2 * W[k][c] for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    deadline = _deadline_guard(140.0)
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find a lower rank solution using local search
    # We try ranks 17, 16, 15
    for target_rank in [17, 16, 15]:
        if time.time() > deadline:
            break
        if not _local_search(n, m, p, target_rank, deadline):
            continue
        # If _local_search finds a solution, it updates best and best_rank
        if best_rank <= target_rank:
            break

    return best

def _local_search(n, m, p, rank, deadline):
    """
    Attempts to find a rank 'rank' decomposition using stochastic local search.
    Returns True if a solution is found and stored in 'best' (nonlocal).
    """
    nonlocal best, best_rank
    rng = random.Random(int(time.time() * 1000) % 10000) # Pseudo-random seed for variety
    
    # Initialize with random sparse matrices
    # Entries are half-integers. We work with doubled integers to keep precision.
    # U2, V2, W2 are the doubled matrices.
    # We want sum(U2*V2*W2) == 8.
    
    def init_random():
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        # Fill with random small values
        vals = [0, 1, -1, 2, -2] # Doubled values: 0, 0.5, -0.5, 1, -1
        for i in range(n*m):
            for c in range(rank):
                if rng.random() < 0.3: # Sparsity
                    U[i][c] = rng.choice(vals)
        for j in range(m*p):
            for c in range(rank):
                if rng.random() < 0.3:
                    V[j][c] = rng.choice(vals)
        for k in range(p*n):
            for c in range(rank):
                if rng.random() < 0.3:
                    W[k][c] = rng.choice(vals)
        return U, V, W

    U, V, W = init_random()
    
    # Precompute indices for fast lookup
    # U_idx[i,j] = i*m+j
    # V_idx[j,k] = j*p+k
    # W_idx[k,i] = k*n+i
    
    def compute_residuals(U, V, W):
        # Returns list of residuals for all (i,j,k)
        # res[i,j,k] = 8 - sum(U2*V2*W2)
        res = [[[0]*p for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i*m + j
                    vj = j*p + k
                    wk = k*n + i
                    for c in range(rank):
                        s += U[ui][c] * V[vj][c] * W[wk][c]
                    res[i][j][k] = 8 - s
        return res

    def error_metric(res):
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = res[i][j][k]
                    err += r*r
        return err

    current_res = compute_residuals(U, V, W)
    current_err = error_metric(current_res)
    
    # Local search loop
    steps = 0
    max_steps = 50000
    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Pick a random variable to change
        # We pick a random equation (i,j,k) that is not satisfied
        unsat = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if current_res[i][j][k] != 0:
                        unsat.append((i,j,k))
        
        if not unsat:
            # Solution found!
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            return True
            
        i, j, k = rng.choice(unsat)
        
        # Pick a random column c
        c = rng.randint(0, rank-1)
        
        # Pick which matrix to modify: U, V, or W
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        mat_idx = rng.randint(0, 2)
        
        if mat_idx == 0:
            idx = i*m + j
            # Changing U[idx][c] affects all (i, j, k') for all k'
            # Specifically, it affects res[i][j][k'] for all k'
            old_val = U[idx][c]
            # Try a new value
            new_val = old_val + rng.choice([-2, -1, 1, 2]) # Step size 0.5, 1.0 in original scale
            
            # Calculate change in error
            delta_err = 0
            for kp in range(p):
                # res[i][j][kp] = 8 - sum(U[idx][c']*V[j*p+kp][c']*W[kp*n+i][c'])
                # Change in sum = (new_val - old_val) * V[j*p+kp][c] * W[kp*n+i][c]
                delta_sum = (new_val - old_val) * V[j*p+kp][c] * W[kp*n+i][c]
                old_r = current_res[i][j][kp]
                new_r = old_r - delta_sum # res = 8 - sum, so if sum increases, res decreases
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                U[idx][c] = new_val
                # Update residuals
                for kp in range(p):
                    delta_sum = (new_val - old_val) * V[j*p+kp][c] * W[kp*n+i][c]
                    current_res[i][j][kp] -= delta_sum
                current_err += delta_err
                continue # Accept and continue
        
        elif mat_idx == 1:
            idx = j*p + k
            # Changing V[idx][c] affects all (i', j, k) for all i'
            old_val = V[idx][c]
            new_val = old_val + rng.choice([-2, -1, 1, 2])
            
            delta_err = 0
            for ip in range(n):
                # res[ip][j][k]
                delta_sum = (new_val - old_val) * U[ip*m+j][c] * W[k*n+ip][c]
                old_r = current_res[ip][j][k]
                new_r = old_r - delta_sum
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                V[idx][c] = new_val
                for ip in range(n):
                    delta_sum = (new_val - old_val) * U[ip*m+j][c] * W[k*n+ip][c]
                    current_res[ip][j][k] -= delta_sum
                current_err += delta_err
                continue

        else:
            idx = k*n + i
            # Changing W[idx][c] affects all (i, j', k) for all j'
            old_val = W[idx][c]
            new_val = old_val + rng.choice([-2, -1, 1, 2])
            
            delta_err = 0
            for jp in range(m):
                # res[i][jp][k]
                delta_sum = (new_val - old_val) * U[i*m+jp][c] * V[jp*p+k][c]
                old_r = current_res[i][jp][k]
                new_r = old_r - delta_sum
                delta_err += new_r*new_r - old_r*old_r
                
            if delta_err < 0:
                W[idx][c] = new_val
                for jp in range(m):
                    delta_sum = (new_val - old_val) * U[i*m+jp][c] * V[jp*p+k][c]
                    current_res[i][jp][k] -= delta_sum
                current_err += delta_err
                continue

    return False
# EVOLVE-BLOCK-END

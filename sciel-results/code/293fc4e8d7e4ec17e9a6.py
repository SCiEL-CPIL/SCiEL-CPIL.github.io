"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Build target tensor for verification
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        result = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        return np.allclose(result, target, atol=tol)
    
    # 2. Structured search for rank reduction
    # Try to find rank base_rank-1 by checking if any single schoolbook term
    # can be expressed as a combination of the others (linear dependency check
    # on the trilinear form is complex, so we use a heuristic).
    
    # Heuristic: For each term to remove, try to see if setting its coefficient
    # to 0 and adjusting nearby terms works. This is a local search.
    
    # More practical: try small structured perturbations
    # For rank r, we need r columns. Start from schoolbook and try to merge.
    
    # Approach: Try rank base_rank - 1 with a specific structure.
    # If we remove term (i0,j0,k0), we need the remaining 59 terms to cover
    # the (i0,j0,k0) contribution. This requires at least one other term to
    # have non-zero entries at (i0,j0), (j0,k0), and (k0,i0) simultaneously,
    # which no schoolbook term does (each term has unique (i,j,k) support).
    
    # So a simple merge won't work. We need a more global restructuring.
    
    # Given the difficulty, let's do a bounded random search with verification.
    # Use numpy for speed.
    
    np.random.seed(42)  # Reproducibility
    
    # Try a few random sparse patterns for rank base_rank - 1
    r_target = base_rank - 1
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        # Generate random sparse decomposition
        U_rand = np.zeros((n * m, r_target), dtype=np.float64)
        V_rand = np.zeros((m * p, r_target), dtype=np.float64)
        W_rand = np.zeros((p * n, r_target), dtype=np.float64)
        
        # Each column: pick random indices and set to 1
        for c in range(r_target):
            ui = np.random.randint(0, n * m)
            vj = np.random.randint(0, m * p)
            wk = np.random.randint(0, p * n)
            U_rand[ui, c] = 1.0
            V_rand[vj, c] = 1.0
            W_rand[wk, c] = 1.0
        
        if verify_decomp(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
            best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
            best_rank = r_target
            break
    
    if time.time() > deadline - 5.0:
        return best
    
    return best
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Strategy: rank-28 decomposition via sparse structured search
    # 20 schoolbook terms for (i,j,k) with k in {0,1,2,3}
    # 8 free terms for k=4 (the 5th column), with ±1/0 entries
    
    result = _search_rank28(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback: rank-29 Strassen + schoolbook
    result = _rank29_fallback(n, m, p)
    if result is not None:
        return result
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _rank29_fallback(n, m, p):
    """Rank-29 via Strassen on 2x2x2 block + schoolbook for rest."""
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen on 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[1][2] = 1
    # M4 = A11(B01-B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (A00+A10)B11
    U[0][4] = 1; U[3][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (A10-A00)(B00+B10)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[3][5] = 1
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    # Schoolbook for remaining: (i,j,k) with j in {0,1}, k in {2,3,4} -> 12 terms
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Schoolbook for j=2: (i,2,k) for i in {0,1}, k in {0,1,2,3,4} -> 10 terms
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _search_rank28(n, m, p, deadline):
    """
    Search for a rank-28 decomposition.
    
    Structure:
    - 20 fixed schoolbook terms for (i,j,k) with k in {0,1,2,3}
    - 8 free terms for the k=4 column
    
    For the 8 free terms (c = 20..27):
    - U[i*m+j, c] ∈ {0, 1, -1} for all (i,j)
    - V[j*p+4, c] ∈ {0, 1, -1} for all j (only k=4 entry matters for k=4 output)
    - W[4*n+i, c] ∈ {0, 1, -1} for all i (only k=4 output)
    
    Actually, the free terms can have entries in any position, but to keep it manageable,
    we restrict: for the 8 free terms, only U entries (all 6 positions), V entries 
    (only the k=4 row, 3 positions), and W entries (only the k=4 row, 2 positions) are free.
    
    Wait, that's not quite right. The free terms contribute to ALL output entries, not just k=4.
    We need to ensure they don't disturb the k<4 outputs.
    
    Better approach: make the 8 free terms have ZERO entries for k<4 in V and W,
    so they only affect the k=4 output. Then:
    - V[j*p+k, c] = 0 for k < 4, V[j*p+4, c] ∈ {0,1,-1}
    - W[k*n+i, c] = 0 for k < 4, W[4*n+i, c] ∈ {0,1,-1}
    - U[i*m+j, c] ∈ {0,1,-1}
    
    Then the k=4 output entry T[i,j,4] = sum_{c=20}^{27} U[i*m+j,c] * V[j*p+4,c] * W[4*n+i,c]
    must equal 1 for all (i,j).
    
    This is a system of 6 equations (one per (i,j)) with 8*6 + 8*3 + 8*2 = 48+24+16 = 88 binary variables.
    We search for a solution.
    """
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 20 schoolbook terms for k in {0,1,2,3}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):  # k = 0,1,2,3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 20
    
    # 8 free terms for k=4
    # For each free term c in 20..27:
    #   U[i*m+j, c] ∈ {0,1,-1} for i in {0,1}, j in {0,1,2}
    #   V[j*p+4, c] ∈ {0,1,-1} for j in {0,1,2}
    #   W[4*n+i, c] ∈ {0,1,-1} for i in {0,1}
    #   All other V and W entries are 0
    
    # We need: for all (i,j): sum_{c=20}^{27} U[i*m+j,c] * V[j*p+4,c] * W[4*n+i,c] = 1
    
    # This is a 6-equation system. Let's search.
    
    # Variable encoding:
    # For each c in 0..7 (free term index):
    #   u[c][i*m+j] for i in {0,1}, j in {0,1,2} -> 6 values
    #   v[c][j] for j in {0,1,2} -> 3 values
    #   w[c][i] for i in {0,1} -> 2 values
    # Total: 8 * (6+3+2) = 88 variables, each in {-1, 0, 1}
    
    # The constraint for (i,j): sum_{c=0}^{7} u[c][i*m+j] * v[c][j] * w[c][i] = 1
    
    # Let's use simulated annealing / random search.
    
    best_solution = None
    best_score = float('inf')
    
    # Initialize randomly
    u = [[0]*6 for _ in range(8)]  # u[c][i*m+j]
    v = [[0]*3 for _ in range(8)]  # v[c][j]
    w = [[0]*2 for _ in range(8)]  # w[c][i]
    
    def compute_outputs(u, v, w):
        """Compute the 6 output values T[i,j,4] for i in {0,1}, j in {0,1,2}."""
        outputs = [[0]*3 for _ in range(2)]
        for i in range(2):
            for j in range(3):
                val = 0
                for c in range(8):
                    val += u[c][i*3+j] * v[c][j] * w[c][i]
                outputs[i][j] = val
        return outputs
    
    def score(u, v, w):
        """Score = sum of squared errors from 1."""
        outputs = compute_outputs(u, v, w)
        err = 0
        for i in range(2):
            for j in range(3):
                err += (outputs[i][j] - 1) ** 2
        return err
    
    # Try to find exact solution via random restarts + local search
    rng = random.Random(42)
    
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        # Random initialization
        for c in range(8):
            for idx in range(6):
                u[c][idx] = rng.choice([-1, 0, 1])
            for j in range(3):
                v[c][j] = rng.choice([-1, 0, 1])
            for i in range(2):
                w[c][i] = rng.choice([-1, 0, 1])
        
        cur_score = score(u, v, w)
        
        if cur_score == 0:
            best_solution = (u[:], v[:], w[:])
            break
        
        # Local search: flip one variable at a time
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Pick a random variable to flip
            c = rng.randint(0, 7)
            var_type = rng.choice(['u', 'v', 'w'])
            
            if var_type == 'u':
                idx = rng.randint(0, 5)
                old_val = u[c][idx]
                # Try all 3 values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    u[c][idx] = new_val
                    new_score = score(u, v, w)
                    if new_score < cur_score or (new_score == cur_score and rng.random() < 0.3):
                        cur_score = new_score
                    else:
                        u[c][idx] = old_val
                        break
            elif var_type == 'v':
                j = rng.randint(0, 2)
                old_val = v[c][j]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    v[c][j] = new_val
                    new_score = score(u, v, w)
                    if new_score < cur_score or (new_score == cur_score and rng.random() < 0.3):
                        cur_score = new_score
                    else:
                        v[c][j] = old_val
                        break
            else:  # w
                i = rng.randint(0, 1)
                old_val = w[c][i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    w[c][i] = new_val
                    new_score = score(u, v, w)
                    if new_score < cur_score or (new_score == cur_score and rng.random() < 0.3):
                        cur_score = new_score
                    else:
                        w[c][i] = old_val
                        break
        
        if cur_score == 0:
            best_solution = (u[:], v[:], w[:])
            break
        
        if cur_score < best_score:
            best_score = cur_score
            best_solution = (u[:], v[:], w[:])
    
    if best_solution is None:
        return None
    
    u_sol, v_sol, w_sol = best_solution
    
    # Build the full U, V, W matrices
    for c in range(8):
        col_idx = 20 + c
        for i in range(2):
            for j in range(3):
                U[i * m + j][col_idx] = u_sol[c][i * 3 + j]
        for j in range(3):
            V[j * p + 4][col_idx] = v_sol[c][j]
        for i in range(2):
            W[4 * n + i][col_idx] = w_sol[c][i]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    
    T = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        u_c = np.array([U[i * m + j][c] for i in range(n) for j in range(m)]).reshape(n, m)
        v_c = np.array([V[j * p + k][c] for j in range(m) for k in range(p)]).reshape(m, p)
        w_c = np.array([W[k * n + i][c] for k in range(p) for i in range(n)]).reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    return np.all(np.abs(T - 1) < 1e-9)

# EVOLVE-BLOCK-END

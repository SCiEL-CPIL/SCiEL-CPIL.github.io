import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Precompute valid indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
    
    # Precompute invalid indices mask for fast verification
    # We will verify by checking all 2500 entries, which is fast enough.

    def verify_exact(U, V, W):
        """
        Verify if U, V, W exactly decompose the tensor.
        U, V, W are lists of lists of Fractions or floats.
        We convert to Fraction for exact arithmetic.
        """
        r = len(U[0])
        
        # Convert to Fraction for exactness
        U_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in U]
        V_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in V]
        W_frac = [[Fraction(x).limit_denominator(10000) for x in row] for row in W]
        
        # Check valid indices
        for u, v, w in valid_indices:
            s = Fraction(0)
            for t in range(r):
                s += U_frac[u][t] * V_frac[v][t] * W_frac[w][t]
            if s != 1:
                return False
        
        # Check invalid indices
        # We need to check all (u, v, w) that are NOT in valid_indices
        # There are 2500 total. 50 are valid. 2450 are invalid.
        # This loop is 2450 * r operations. If r=50, that's 122,500 multiplications.
        # This is fast in Python if we optimize the inner loop.
        
        # Optimization: Precompute U_frac and V_frac products?
        # Let's just iterate.
        
        valid_set = set(valid_indices)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) in valid_set:
                        continue
                    s = Fraction(0)
                    for t in range(r):
                        s += U_frac[u][t] * V_frac[v][t] * W_frac[w][t]
                    if s != 0:
                        return False
        return True

    def solve_for_W(U, V):
        """
        Solve for W given U, V using least squares.
        Returns W as a list of lists of floats.
        """
        r = len(U[0])
        num_vars = pn * r
        num_eqs = nm * mp * pn
        
        # We group by w as in the original program, which is efficient.
        W_sol = [[0.0] * r for _ in range(pn)]
        
        for w in range(pn):
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    # Determine T[u,v,w]
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    b_w[idx] = target
                    for t in range(r):
                        A_w[idx, t] = U[u][t] * V[v][t]
                    idx += 1
            
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w] = sol.tolist()
        
        return W_sol

    def round_to_half_integer(W):
        """
        Round each entry to the nearest half-integer.
        """
        W_r = [[0.0] * len(row) for row in W]
        for i in range(len(W)):
            for j in range(len(W[i])):
                val = W[i][j]
                W_r[i][j] = round(val * 2) / 2.0
        return W_r

    def try_rank(r, num_trials=100):
        """
        Try to find a decomposition of rank r.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize U, V with small integers
            # Use different distributions to explore the space
            if trial % 3 == 0:
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            elif trial % 3 == 1:
                U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(nm)]
                V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(mp)]
            else:
                # Sparse initialization
                U = [[0.0] * r for _ in range(nm)]
                V = [[0.0] * r for _ in range(mp)]
                for u in range(nm):
                    num_nonzero = random.randint(1, min(3, r))
                    indices = random.sample(range(r), num_nonzero)
                    for idx in indices:
                        U[u][idx] = random.choice([-1, 1])
                for v in range(mp):
                    num_nonzero = random.randint(1, min(3, r))
                    indices = random.sample(range(r), num_nonzero)
                    for idx in indices:
                        V[v][idx] = random.choice([-1, 1])
            
            # Solve for W
            W_float = solve_for_W(U, V)
            
            # Round W
            W_r = round_to_half_integer(W_float)
            
            # Verify exactly
            if verify_exact(U, V, W_r):
                return {
                    "U": U,
                    "V": V,
                    "W": W_r
                }
                
            # If close, try local polish? (Skipped for now to save time)
            
        return None

    # Search for lower rank
    # Start from 49 down to 40
    for r in range(49, 39, -1):
        if time.time() > deadline:
            break
            
        # Increase trials for lower ranks? Or keep constant?
        # Let's do 50 trials per rank
        result = try_rank(r, num_trials=50)
        if result:
            return result

    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-END

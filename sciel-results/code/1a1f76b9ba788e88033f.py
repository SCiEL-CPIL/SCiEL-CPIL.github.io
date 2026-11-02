import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 140  # seconds, with safety margin
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def to_half_integers(matrix, tol=1e-6):
        rounded = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) > tol:
                    return None
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            rounded.append(res_row)
        return rounded

    # Strategy 1: Structured construction using block decomposition
    # For 3x4x5, we can exploit structure to get better than 60
    # Try a construction based on splitting the middle dimension
    
    def try_structured(R):
        """Try to build a decomposition of rank R using structured patterns."""
        # Initialize with zeros
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        
        # Use a pattern: for each "block" of the computation, assign terms
        # This is a heuristic structured approach
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if idx >= R:
                        return None
                    # Assign this (i,j,k) term to component idx
                    # But we want to share components where possible
                    # For now, just check if we can fit all terms
                    idx += 1
        
        if idx > R:
            return None
        
        # This simple approach won't work; need actual sharing
        return None
    
    # Strategy 2: Integer linear search over small coefficient sets
    # Search for U, V with entries in {-1, 0, 1} and W with entries in {-1, 0, 1}
    # such that the tensor equation holds
    
    def search_integer_decomposition(R, max_iters=2000):
        """Search for integer decomposition of rank R."""
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Random sparse initialization
            U = [[0] * R for _ in range(nm)]
            V = [[0] * R for _ in range(mp)]
            W = [[0] * R for _ in range(pn)]
            
            density = 0.3 + 0.2 * random.random()
            for i in range(nm):
                for c in range(R):
                    if random.random() < density:
                        U[i][c] = random.choice([-1, 1])
            for j in range(mp):
                for c in range(R):
                    if random.random() < density:
                        V[j][c] = random.choice([-1, 1])
            for k in range(pn):
                for c in range(R):
                    if random.random() < density:
                        W[k][c] = random.choice([-1, 1])
            
            if verify_decomposition(U, V, W, R):
                return (U, V, W)
        
        return None
    
    # Strategy 3: Use numpy to solve for W given random U,V, then round
    def search_with_w_solve(R, max_restarts=500):
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                return None
            
            density = 0.4 + 0.2 * random.random()
            U = [[0] * R for _ in range(nm)]
            V = [[0] * R for _ in range(mp)]
            
            for i in range(nm):
                for c in range(R):
                    if random.random() < density:
                        U[i][c] = random.choice([-1, 1])
            for j in range(mp):
                for c in range(R):
                    if random.random() < density:
                        V[j][c] = random.choice([-1, 1])
            
            # Build linear system for W
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        for c in range(R):
                            coeff = u_row[c] * v_row[c]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = W_flat.reshape((p * n, R))
            W_nice = to_half_integers(W_sol)
            if W_nice is not None:
                if verify_decomposition(U, V, W_nice, R):
                    return (U, V, W_nice)
        
        return None
    
    # Strategy 4: Greedy construction - build decomposition term by term
    def greedy_construct(R):
        """Greedy: pick U,V columns to maximize progress on remaining equations."""
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        
        # Track which equations are satisfied
        equations = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    equations[(i, j, k)] = 0
        
        for c in range(R):
            if time.time() - start_time > time_limit:
                return None
            
            # Find equations with largest deficit
            deficits = []
            for eq, val in equations.items():
                deficit = 1 - val
                if abs(deficit) > 1e-9:
                    deficits.append((abs(deficit), eq))
            
            if not deficits:
                break
            
            # Pick the equation with largest deficit
            deficits.sort(reverse=True)
            _, (i, j, k) = deficits[0]
            
            # Set U[i*m+j][c], V[j*p+k][c], W[k*n+i][c]
            # We want U*V*W = deficit for this equation
            # Simple choice: U=1, V=1, W=deficit (if deficit is half-integer)
            deficit_val = 1 - equations[(i, j, k)]
            
            # Try to assign integer/half-integer values
            if abs(deficit_val - round(deficit_val)) < 1e-9:
                w_val = round(deficit_val)
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = w_val
            elif abs(deficit_val * 2 - round(deficit_val * 2)) < 1e-9:
                w_val = round(deficit_val * 2) / 2.0
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = w_val
            else:
                # Can't satisfy this equation with half-integers
                return None
            
            # Update all equations
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        u_val = U[i2 * m + j2][c]
                        v_val = V[j2 * p + k2][c]
                        w_val = W[k2 * n + i2][c]
                        equations[(i2, j2, k2)] += u_val * v_val * w_val
        
        # Check if all equations are satisfied
        for eq, val in equations.items():
            if abs(val - 1) > 1e-9:
                return None
        
        return (U, V, W)
    
    # Try various ranks in increasing order
    # Lower bound for 3x4x5 is at least 30 (trivial bound is 15, but known better)
    # Try ranks from 30 to 59
    
    ranks_to_try = list(range(30, 60))
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
        
        if time.time() - start_time > time_limit:
            break
        
        # Try greedy construction first (fast, deterministic)
        result = greedy_construct(R)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            return {"U": best_U, "V": best_V, "W": best_W}
        
        if time.time() - start_time > time_limit:
            break
        
        # Try W-solve approach
        result = search_with_w_solve(R, max_restarts=100)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            return {"U": best_U, "V": best_V, "W": best_W}
        
        if time.time() - start_time > time_limit:
            break
        
        # Try pure integer search (slower)
        result = search_integer_decomposition(R, max_iters=500)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

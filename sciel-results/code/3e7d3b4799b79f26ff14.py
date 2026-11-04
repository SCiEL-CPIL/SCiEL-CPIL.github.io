"""
Matrix multiplication 3x5 * 5x5: rank-minimal exact tensor decomposition.
Strategy: Systematic Algebraic Search with Backtracking.
Instead of random guessing which fails due to sparsity, we enumerate
sparse support structures (skeletons) and solve for the coefficients.
"""
import time
import itertools
import random
from fractions import Fraction
import sys

DIMENSIONS = [3, 5, 5]
DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

def _validate_fast(U, V, W, n, m, p):
    """
    Validate if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    if not U:
        return False
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    uv = U[i*m+j][r]
                    if uv == 0:
                        continue
                    vw = V[j*p+k][r]
                    if vw == 0:
                        continue
                    s += uv * vw * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = Fraction(1)
                V[j*p+k][col] = Fraction(1)
                W[k*n+i][col] = Fraction(1)
                col += 1
    return U, V, W, rank

def _solve_linear_system_ints(A, b):
    """
    Solve A x = b over integers/half-integers.
    A is a list of lists (equations), b is a list of Fractions (RHS).
    Returns a list of Fractions x or None if no integer/half-integer solution exists.
    This is a simplified solver for small systems.
    """
    n_vars = len(A[0]) if A else 0
    n_eqs = len(A)
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian elimination
    rank = 0
    for col in range(n_vars):
        # Find pivot
        pivot = -1
        for row in range(rank, n_eqs):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
            
        # Swap
        M[rank], M[pivot] = M[pivot], M[rank]
        
        # Eliminate
        for row in range(n_eqs):
            if row == rank:
                continue
            factor = M[row][col] / M[rank][col]
            if factor != 0:
                for c in range(col, n_vars + 1):
                    M[row][c] -= factor * M[rank][c]
        
        rank += 1
        if rank == n_vars:
            break
            
    # Check for inconsistencies
    for row in range(rank, n_eqs):
        if any(M[row][c] != 0 for c in range(n_vars)):
            return None
            
    # Extract solution
    # We need to handle free variables. For small systems, we assume unique solution or
    # that free variables can be set to 0 to find one solution.
    x = [Fraction(0)] * n_vars
    for r in range(rank):
        # Find the pivot column for this row
        pivot_col = -1
        for c in range(n_vars):
            if M[r][c] != 0:
                pivot_col = c
                break
        if pivot_col == -1:
            continue
        # Sum known terms
        val = M[r][n_vars]
        for c in range(n_vars):
            if c != pivot_col and x[c] != 0:
                val -= M[r][c] * x[c]
        x[pivot_col] = val / M[r][pivot_col]
        
    # Verify solution
    for i in range(n_eqs):
        s = Fraction(0)
        for j in range(n_vars):
            s += A[i][j] * x[j]
        if s != b[i]:
            return None
            
    # Check if entries are integers or half-integers
    for val in x:
        # val is a Fraction. Check if 2*val is an integer.
        if (val * 2).denominator != 1:
            return None
            
    return x

def _try_rank_skeleton(n, m, p, rank, skeleton_seed=0):
    """
    Try to find a decomposition of given rank by enumerating sparse skeletons.
    Each rank term r is assigned a subset of non-zero entries in U, V, W.
    We force the structure such that each term contributes to a specific set of (i,j,k).
    """
    if _time_left() < 2:
        return None
        
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Strategy:
    # We assign each of the 'rank' terms a "center" output index (i,j,k).
    # The term r will have non-zeros in U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # It might also have non-zeros in other positions to allow for cancellation or addition.
    # For simplicity, start with the "monomial" skeleton where each term is a single
    # product of basis vectors.
    # If that fails (which it will for rank < n*m*p), we need to allow superposition.
    
    # A more powerful skeleton:
    # Let's try to construct the identity matrix C = A B.
    # We can try to decompose the tensor T = sum_r u_r v_r w_r.
    
    # Heuristic Skeleton Generation:
    # We pick 'rank' random subsets of indices for U, V, W.
    # To make it solvable, we ensure that for each output (i,j,k), there is at least one rank term
    # that has non-zeros in U[i,j], V[j,k], W[k,i].
    
    # Let's use a deterministic approach for small ranks first.
    # If rank is close to n*m*p, schoolbook works.
    # We are looking for rank < 75.
    
    # Let's try a specific structured decomposition:
    # For 3x5x5, we can try to block the 5x5 part.
    # But let's stick to the algebraic solver on random sparse patterns.
    
    rng = random.Random(skeleton_seed)
    
    # Generate a random sparse pattern
    # We want the pattern to be "full rank" in the sense that it covers all outputs.
    
    U_support = [set() for _ in range(n_m)]
    V_support = [set() for _ in range(m_p)]
    W_support = [set() for _ in range(p_n)]
    
    # Assign each rank term a primary responsibility
    outputs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    # Shuffle outputs
    rng.shuffle(outputs)
    
    # Assign first 'rank' outputs to distinct terms
    for r in range(rank):
        i, j, k = outputs[r]
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_support[u_idx].add(r)
        V_support[v_idx].add(r)
        W_support[w_idx].add(r)
        
        # Add some random extra connections to allow flexibility
        if rng.random() < 0.2:
            # Add a random extra U entry
            extra_u = rng.randint(0, n_m - 1)
            U_support[extra_u].add(r)
        if rng.random() < 0.2:
            extra_v = rng.randint(0, m_p - 1)
            V_support[extra_v].add(r)
        if rng.random() < 0.2:
            extra_w = rng.randint(0, p_n - 1)
            W_support[extra_w].add(r)
            
    # Now, set up the linear system.
    # We have variables: U[u,r], V[v,r], W[w,r] for all u,v,w in support.
    # Equations: For each (i,j,k), sum_{r in common support} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    # This is a system of NON-LINEAR equations (products of variables).
    # The linear solver above only works for linear systems.
    # We need a different approach for non-linear.
    # However, if we FIX the values of U and V to be +/-1, then the system becomes linear in W.
    # Let's try that:
    # 1. Randomly assign U[u,r] and V[v,r] to +/-1 for all non-zero entries.
    # 2. For each output (i,j,k), the equation becomes:
    #    sum_{r} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    #    Let coeff_r = U[i,j,r]*V[j,k,r].
    #    Then sum_{r} coeff_r * W[k,i,r] = 1.
    #    This is a linear equation in W[k,i,r].
    # 3. We can solve for W entries independently for each (k,i) pair?
    #    No, W[k,i,r] appears in equations for all j.
    #    For a fixed (k,i), we have m equations (one for each j) involving the same W[k,i,r] variables.
    #    So we can set up a linear system for each (k,i) pair.
    
    # Let's implement this "Fix U,V and solve for W" strategy.
    
    # Assign random +/-1 to U and V supports
    U_vals = {}
    V_vals = {}
    for u in range(n_m):
        for r in U_support[u]:
            U_vals[(u, r)] = Fraction(1) if rng.random() < 0.5 else Fraction(-1)
    for v in range(m_p):
        for r in V_support[v]:
            V_vals[(v, r)] = Fraction(1) if rng.random() < 0.5 else Fraction(-1)
            
    # Solve for W
    # For each (k, i), we have variables W[k*n+i, r] for r in W_support[k*n+i].
    # Equations: For each j, sum_{r in W_support[k*n+i]} (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    # Note: U[i*m+j, r] is 0 if r not in U_support[i*m+j].
    #        V[j*p+k, r] is 0 if r not in V_support[j*p+k].
    
    W_vals = {}
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            if not W_support[w_idx]:
                # No variables to solve for, but we need sum to be 1. Impossible if no terms.
                # Unless we add a dummy term? No, rank is fixed.
                # If support is empty, we can't satisfy the equation unless RHS is 0 (it's 1).
                # So this skeleton is invalid.
                return None
                
            # Build linear system for this (k,i)
            # Variables: W[w_idx, r] for r in W_support[w_idx]
            var_indices = sorted(list(W_support[w_idx]))
            var_map = {r: idx for idx, r in enumerate(var_indices)}
            num_vars = len(var_indices)
            
            A = []
            b = []
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                row = [Fraction(0)] * num_vars
                for r in var_indices:
                    u_val = U_vals.get((u_idx, r), Fraction(0))
                    v_val = V_vals.get((v_idx, r), Fraction(0))
                    if u_val != 0 and v_val != 0:
                        row[var_map[r]] = u_val * v_val
                A.append(row)
                b.append(Fraction(1))
                
            # Solve A x = b
            x = _solve_linear_system_ints(A, b)
            if x is None:
                return None
                
            for idx, r in enumerate(var_indices):
                W_vals[(w_idx, r)] = x[idx]
                
    # Construct final matrices
    U_mat = [[Fraction(0)] * rank for _ in range(n_m)]
    V_mat = [[Fraction(0)] * rank for _ in range(m_p)]
    W_mat = [[Fraction(0)] * rank for _ in range(p_n)]
    
    for (u, r), val in U_vals.items():
        U_mat[u][r] = val
    for (v, r), val in V_vals.items():
        V_mat[v][r] = val
    for (w, r), val in W_vals.items():
        W_mat[w][r] = val
        
    # Validate
    if _validate_fast(U_mat, V_mat, W_mat, n, m, p):
        return U_mat, V_mat, W_mat, rank
        
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best_rank = rank
    
    # Try to find better (lower rank) decompositions using algebraic search
    # We search from a lower bound upwards.
    # Lower bound for 3x5x5 is likely around 38-40 based on Strassen-like bounds.
    # Let's try a few ranks.
    
    # To save time, we only try a limited number of random skeletons per rank.
    # If we find a solution, we return it.
    
    # We will try ranks from 40 to 74.
    # But we must be careful with time.
    
    # Let's prioritize finding ANY improvement over 75.
    # We'll try a few specific ranks.
    
    # If time allows, try lower ranks.
    # For now, let's try a small set of ranks near the expected optimum.
    
    # Actually, let's just try a few random skeletons for rank = 60, 50, 40.
    # If we find one, we update best_rank.
    
    target_ranks = [60, 55, 50, 45, 40]
    
    for target_rank in target_ranks:
        if _time_left() < 5:
            break
            
        # Try a few random skeletons
        for attempt in range(20):
            if _time_left() < 2:
                break
                
            res = _try_rank_skeleton(n, m, p, target_rank, skeleton_seed=attempt * 1000 + target_rank)
            if res is not None:
                U, V, W, rank = res
                best_rank = rank
                break # Found one for this rank, try lower rank
                
    # Verify the best found is valid
    if not _validate_fast(U, V, W, n, m, p):
        U, V, W, rank = _schoolbook(n, m, p)
        
    # Convert to lists of ints (or half-ints represented as floats? No, task says integer/half-integer entries)
    # The output interface expects lists. Fractions like 1/2 should be 0.5.
    # "integer/half-integer entries" implies we can output 0.5.
    # Let's convert Fractions to floats if they are .5 or integers.
    
    def _frac_to_out(f):
        if f.denominator == 1:
            return int(f.numerator)
        elif f.denominator == 2:
            return f.numerator / 2.0
        else:
            # Should not happen if solver worked
            return float(f)
            
    U_out = [[_frac_to_out(x) for x in row] for row in U]
    V_out = [[_frac_to_out(x) for x in row] for row in V]
    W_out = [[_frac_to_out(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

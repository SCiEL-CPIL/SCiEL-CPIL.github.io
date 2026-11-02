"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, n, m, p, tol=1e-9):
        """Fast verification using NumPy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val - 1) > tol:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > tol:
                        return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    def try_rational_approx(U_float, V_float, W_float, n, m, p, max_denom=8):
        """Try to find rational approximations with small denominators."""
        def approx_rational(x, max_d):
            if abs(x) < 1e-10:
                return Fraction(0)
            # Try denominators 1 to max_d
            for d in range(1, max_d + 1):
                num = round(x * d)
                if abs(x - num / d) < 1e-9:
                    return Fraction(num, d)
            # Fall back to float as Fraction
            return Fraction(x).limit_denominator(max_d * max_d)
        
        U_r = [[approx_rational(float(x), max_denom) for x in row] for row in U_float]
        V_r = [[approx_rational(float(x), max_denom) for x in row] for row in V_float]
        W_r = [[approx_rational(float(x), max_denom) for x in row] for row in W_float]
        if verify_exact(U_r, V_r, W_r, n, m, p):
            return {"U": [[float(f) for f in row] for row in U_r],
                    "V": [[float(f) for f in row] for row in V_r],
                    "W": [[float(f) for f in row] for row in W_r]}
        return None

    # --- Structured parametric search exploiting block structure ---
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j=j', k=k', i=i'
    # 
    # Key observation: We can try decompositions where U, V, W have
    # Kronecker-product structure. For 2x3 x 3x7:
    # 
    # Try: rank = n*p + m*p + n*m - something, or use known lower bounds.
    # Known: rank(2x3 x 3x7) >= max(2*3, 3*7, 2*7) = max(6,21,14) = 21
    # Also rank >= ceil(n*m*p / max(n,m,p)) = ceil(42/7) = 6 (weak)
    # 
    # Strategy: Use a "block Kronecker" ansatz.
    # Let U = A ⊗ B type structure, but more generally, parameterize
    # with shared parameters across blocks.
    
    def structured_search(rank, seed=42):
        """
        Try to find a rank decomposition using structured parameterization.
        We exploit the fact that the multiplication tensor has symmetries
        related to the index ranges.
        """
        if time.time() > deadline - 5.0:
            return None
        
        np.random.seed(seed)
        
        # Parameterize U, V, W with fewer free parameters using block structure.
        # For 2x3 x 3x7, let's try a "semi-Kronecker" approach:
        # U[i*m+j, r] depends on i, j, r in a structured way.
        
        # Ansatz 1: U[i*m+j, r] = sum over sub-indices
        # We'll use random structured initialization and ALS.
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with structured random values
        for i in range(n):
            for j in range(m):
                U[i*m+j] = np.random.randn(rank) * 0.5
        for j in range(m):
            for k in range(p):
                V[j*p+k] = np.random.randn(rank) * 0.5
        for k in range(p):
            for i in range(n):
                W[k*n+i] = np.random.randn(rank) * 0.5
        
        valid_abc = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
        valid_set = set(valid_abc)
        all_abc = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
        invalid_abc = [t for t in all_abc if t not in valid_set]
        
        # Alternating least squares with structured updates
        for it in range(200):
            if time.time() > deadline - 5.0:
                return None
            
            # Update U: for each a, solve linear system
            # For fixed V, W: U[a] should satisfy:
            #   sum_c (V[b] * W[c])_r * U[a,r] = 1 for valid (a,b,c)
            #   sum_c (V[b] * W[c])_r * U[a,r] = 0 for invalid (a,b,c)
            # This is: M_a @ U[a] = y_a where M_a[b,c,:] = V[b]*W[c]
            
            for a in range(n * m):
                i_a = a // m
                j_a = a % m
                # Collect constraints for this a
                A_mat = []
                y_vec = []
                for b in range(m * p):
                    for c in range(p * n):
                        if (a, b, c) in valid_set:
                            A_mat.append(V[b] * W[c])
                            y_vec.append(1.0)
                        else:
                            A_mat.append(V[b] * W[c])
                            y_vec.append(0.0)
                if A_mat:
                    A_arr = np.array(A_mat)
                    y_arr = np.array(y_vec)
                    # Least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_arr, y_arr, rcond=None)
                        U[a] = sol
                    except:
                        pass
            
            # Update V similarly
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                A_mat = []
                y_vec = []
                for a in range(n * m):
                    for c in range(p * n):
                        if (a, b, c) in valid_set:
                            A_mat.append(U[a] * W[c])
                            y_vec.append(1.0)
                        else:
                            A_mat.append(U[a] * W[c])
                            y_vec.append(0.0)
                if A_mat:
                    A_arr = np.array(A_mat)
                    y_arr = np.array(y_vec)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_arr, y_arr, rcond=None)
                        V[b] = sol
                    except:
                        pass
            
            # Update W similarly
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                A_mat = []
                y_vec = []
                for a in range(n * m):
                    for b in range(m * p):
                        if (a, b, c) in valid_set:
                            A_mat.append(U[a] * V[b])
                            y_vec.append(1.0)
                        else:
                            A_mat.append(U[a] * V[b])
                            y_vec.append(0.0)
                if A_mat:
                    A_arr = np.array(A_mat)
                    y_arr = np.array(y_vec)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_arr, y_arr, rcond=None)
                        W[c] = sol
                    except:
                        pass
            
            # Check loss
            loss = 0.0
            for (a, b, c) in valid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss += (val - 1)**2
            for (a, b, c) in invalid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss += val**2
            
            if loss < 1e-12:
                result = try_rational_approx(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_denom=16)
                if result:
                    return result
        
        return None

    # Try various ranks from low to high
    best = schoolbook()
    best_rank = n * m * p
    
    # Lower bound is at least max(n*m, m*p, p*n) = 21
    # Try from 21 upward
    for r in range(21, best_rank):
        if time.time() > deadline - 20.0:
            break
        found = False
        for seed in [42, 123, 777, 2024, 999]:
            if time.time() > deadline - 20.0:
                break
            result = structured_search(r, seed=seed + r * 31)
            if result is not None:
                best = result
                best_rank = r
                found = True
                break
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END

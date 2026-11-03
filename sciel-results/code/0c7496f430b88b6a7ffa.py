import time
import numpy as np
from math import gcd
from functools import reduce
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def get_schoolbook():
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

    def validate(U, V, W):
        rank = len(U[0])
        # Use Fraction for exactness
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        # Check all tensor entries
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                lhs = Fraction(0)
                                for r in range(rank):
                                    lhs += U_frac[i*m+j][r] * V_frac[j_prime*p+k_prime][r] * W_frac[k_prime*n+i_prime][r]
                                rhs = 1 if (i == i_prime and j == j_prime and k == k_prime) else 0
                                if lhs != rhs:
                                    return False
        return True

    def solve_diophantine(coeffs, target=1):
        """
        Find integer x such that dot(coeffs, x) = target.
        Returns None if gcd(coeffs) does not divide target.
        """
        g = reduce(gcd, [abs(c) for c in coeffs if c != 0], 0)
        if g == 0:
            return None
        if target % g != 0:
            return None
        
        # Find a solution using extended Euclidean algorithm on pairs
        # This is a simplified version that finds one solution
        # If gcd is 1 or 2, and coeffs are small, this should work
        
        # We need x such that sum(c_i x_i) = target
        # We can set most x_i to 0 and solve for 2 variables
        # c_a x_a + c_b x_b = target
        # Find a, b such that gcd(c_a, c_b) divides target
        
        n_coeffs = len(coeffs)
        for a in range(n_coeffs):
            if coeffs[a] == 0: continue
            for b in range(a+1, n_coeffs):
                if coeffs[b] == 0: continue
                g_ab = gcd(abs(coeffs[a]), abs(coeffs[b]))
                if target % g_ab == 0:
                    # Solve c_a x_a + c_b x_b = target
                    # Use extended gcd
                    def egcd(a, b):
                        if b == 0:
                            return (a, 1, 0)
                        else:
                            g, x, y = egcd(b, a % b)
                            return (g, y, x - (a // b) * y)
                    
                    g, x, y = egcd(abs(coeffs[a]), abs(coeffs[b]))
                    scale = target // g
                    x_a = x * scale
                    x_b = y * scale
                    if coeffs[a] < 0: x_a = -x_a
                    if coeffs[b] < 0: x_b = -x_b
                    
                    x_vec = [0] * n_coeffs
                    x_vec[a] = x_a
                    x_vec[b] = x_b
                    
                    # Verify
                    s = sum(coeffs[i] * x_vec[i] for i in range(n_coeffs))
                    if s == target:
                        return x_vec
        return None

    def try_rank_search(target_rank, max_attempts=200000):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U and V with small integers
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # Compute M[i,k,r] = sum_j U[i,j,r] V[j,k,r]
            M = np.zeros((n, p, target_rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[i, k, r] = s
            
            valid = True
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    coeffs = M[i, k, :]
                    non_zero = [abs(c) for c in coeffs if c != 0]
                    if not non_zero:
                        valid = False
                        break
                    g = reduce(gcd, non_zero)
                    if g not in (1, 2):
                        valid = False
                        break
                    
                    target = 1
                    if g == 2:
                        target = 2
                        scale_factor = 2.0
                    else:
                        scale_factor = 1.0
                        
                    w_vec = solve_diophantine(coeffs, target)
                    if w_vec is None:
                        valid = False
                        break
                    
                    for r in range(target_rank):
                        W[k*n+i, r] = w_vec[r] / scale_factor
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Full validation
                if validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 27 downwards
    # Rank 27 is the known minimal rank for 2x3x5
    for rank in [27, 26, 25]:
        if time.time() > deadline - 10:
            break
        result = try_rank_search(rank, max_attempts=50000)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()

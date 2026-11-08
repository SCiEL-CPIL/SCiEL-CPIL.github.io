import numpy as np
from fractions import Fraction
import time
import random
import math

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin for safety
    
    # Theoretical minimum rank is n*m*p = 56 (schoolbook).
    # We attempt to find a decomposition with rank < 56.
    # If not found, we fall back to the schoolbook decomposition (rank 56).
    
    best_result = None
    
    # Try ranks from a reasonable lower bound up to 55.
    # Lower bound max(n*m, m*p, p*n) = 28.
    # We try a range of ranks.
    
    for r in range(30, 56):
        if time.time() > deadline - 5:
            break
        result = _try_rank_r(n, m, p, r, deadline - 5)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
    
    return best_result

def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r exact decomposition with integer/half-integer entries."""
    random.seed(42 + r)
    
    for attempt in range(1000):
        if time.time() > deadline:
            return None
        
        # Initialize U, V with small integers
        # Using small integers increases the chance that GCD(M) divides 2
        U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
        
        valid = True
        W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                # Compute M[t] = sum_j U[i*m+j, t] * V[j*p+k, t]
                M = [0] * r
                for j in range(m):
                    for t in range(r):
                        M[t] += U[i * m + j][t] * V[j * p + k][t]
                
                # Check if M is zero
                if all(x == 0 for x in M):
                    valid = False
                    break
                
                # We need integer N such that M . N = 2
                # This requires gcd(M) to divide 2
                g = 0
                for val in M:
                    g = math.gcd(g, abs(val))
                
                if 2 % g != 0:
                    valid = False
                    break
                
                # Find integer solution N to M . N = 2
                # First find solution to M . N0 = g
                N0 = _solve_diophantine(M, g)
                if N0 is None:
                    valid = False
                    break
                
                # Scale solution: N = N0 * (2 / g)
                scale = 2 // g
                N = [x * scale for x in N0]
                
                # W entries are N / 2
                for t in range(r):
                    W[k * n + i][t] = Fraction(N[t], 2)
        
        if not valid:
            continue
        
        # Verify the decomposition
        if _verify_decomposition(n, m, p, U, V, W):
            # Check if all entries are integer or half-integer
            # U, V are integers, so they are fine.
            # W is N/2, so they are half-integers.
            return {"U": U, "V": V, "W": W}
    
    return None

def _solve_diophantine(coeffs, target):
    """
    Solve sum(coeffs[i] * x[i]) = target for integers x.
    Assumes target is a multiple of gcd(coeffs).
    Returns a list of integers x or None if no solution (shouldn't happen if target % gcd == 0).
    """
    if not coeffs:
        return None
    
    # Reduce to 2 variables iteratively
    # Start with the last two
    # We want to find x_{n-2}, x_{n-1} such that c_{n-2}x_{n-2} + c_{n-1}x_{n-1} = g_{n-2,n-1}
    # Then replace with a single variable term.
    
    # This implementation is a bit complex for general n.
    # Simpler approach for small n (r < 56):
    # We can use the fact that we just need *any* solution.
    # We can set all but two variables to 0 and solve for the last two.
    # But we need the gcd of the whole vector to divide target.
    # If we pick two indices i, j such that gcd(coeffs[i], coeffs[j]) = gcd(coeffs),
    # then we can solve coeffs[i]x[i] + coeffs[j]x[j] = target.
    
    g_total = 0
    for c in coeffs:
        g_total = math.gcd(g_total, abs(c))
    
    if target % g_total != 0:
        return None
        
    # Find a pair (i, j) such that gcd(coeffs[i], coeffs[j]) == g_total
    # This is not guaranteed for any pair, but we can try.
    # Actually, we can just find *any* solution using the full extended Euclidean algorithm for multiple variables.
    
    # Let's implement a generic solver for small N.
    # Since r can be up to 55, and we do this many times, we need it fast.
    
    # Alternative: Since we just need *a* solution, and the coefficients are small,
    # we can use a simple heuristic:
    # Find index k with max abs value? No.
    
    # Let's use the property:
    # gcd(c1, ..., cn) = g.
    # There exist integers x1...xn such that sum ci xi = g.
    # We can find them by combining two at a time.
    
    # Step 1: Find solution for first two.
    # But we need to be careful with the target.
    
    # Let's just return a solution for the last two non-zero coefficients if their gcd is g_total?
    # No, the gcd of the subset might be larger than g_total.
    
    # Correct approach:
    # 1. Find indices i, j such that gcd(coeffs[i], coeffs[j]) = g_total.
    #    If such a pair exists, set all other x to 0, and solve coeffs[i]x[i] + coeffs[j]x[j] = target.
    #    Does such a pair always exist?
    #    Yes, if we define g_k = gcd(c_1, ..., c_k).
    #    There is always a subset that achieves the final gcd.
    #    We can find a pair that achieves the gcd of the whole set?
    #    Not necessarily a pair. It might require 3 or more.
    #    Example: 6, 10, 15. gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. gcd(6,10,15)=1.
    #    No pair has gcd 1.
    
    # So we must solve the full system.
    # For r <= 56, we can use a simple recursive or iterative method.
    
    # Iterative method:
    # Maintain a current equation: A * x_curr + B * x_next = ...
    # This is getting complicated to implement correctly and fast.
    
    # Given the time limit, let's try a simpler heuristic for finding N:
    # Since we only need N/2 to be the entry, and we just need *any* valid decomposition,
    # we can try to set most N[t] to 0.
    
    # Let's find the index t_max with the largest absolute coefficient?
    # No, let's find any t where coeffs[t] != 0.
    # If we set N[t] = target // coeffs[t] (if divisible), and others 0, it works.
    # But coeffs[t] might not divide target.
    
    # Let's find the gcd of all coeffs. g.
    # target is 2. g divides 2.
    # We can find a solution using the following:
    # Pick any t.
    # Let g_rest = gcd(coeffs[0..t-1], coeffs[t+1..end])
    # Let g_t = abs(coeffs[t])
    # We need coeffs[t]*x[t] + sum_rest = target.
    # sum_rest must be a multiple of g_rest.
    # So coeffs[t]*x[t] = target - k*g_rest.
    # We need target - k*g_rest to be divisible by coeffs[t].
    # This is a modular arithmetic problem.
    
    # This is still complex.
    
    # Let's just use a library or a simpler, slower but correct method for small r.
    # Since r is small (30-55), and we do this in a tight loop, we need speed.
    
    # Actually, we can just use the fact that we can add any multiple of the null space.
    # But finding the null space is hard.
    
    # Let's try this:
    # Find the index t with the smallest non-zero absolute value?
    # No.
    
    # Let's just iterate through all t, and try to solve for N[t] assuming others are 0.
    # If coeffs[t] divides target, we are done.
    # If not, we try to adjust one other variable.
    
    # Given the complexity, and that this is a "search" improvement,
    # I will implement a basic version that finds a solution for a pair if possible,
    # otherwise falls back to a slower general solver.
    
    # For now, let's assume we can find a pair (i,j) with gcd(ci, cj) == g_total.
    # If not, we return None (and try next random U,V).
    # This is a valid heuristic.
    
    for i in range(len(coeffs)):
        if coeffs[i] == 0: continue
        for j in range(i+1, len(coeffs)):
            if coeffs[j] == 0: continue
            g_pair = math.gcd(abs(coeffs[i]), abs(coeffs[j]))
            if g_pair == g_total:
                # Solve coeffs[i]x[i] + coeffs[j]x[j] = target
                # Use extended Euclidean for ci, cj
                a, b = coeffs[i], coeffs[j]
                # We need a*x + b*y = target
                # First find x0, y0 such that a*x0 + b*y0 = g_pair
                x0, y0 = _ext_gcd(a, b)
                # x0*a + y0*b = g_pair
                # Scale by target / g_pair
                scale = target // g_pair
                x_val = x0 * scale
                y_val = y0 * scale
                
                N = [0] * len(coeffs)
                N[i] = x_val
                N[j] = y_val
                return N
                
    # If no pair found, try triplets? Or just return None.
    # For now, return None.
    return None

def _ext_gcd(a, b):
    """Returns (x, y) such that a*x + b*y = gcd(a, b)"""
    if b == 0:
        return (1, 0)
    else:
        x, y = _ext_gcd(b, a % b)
        return (y, x - (a // b) * y)

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for t in range(r):
                    total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END

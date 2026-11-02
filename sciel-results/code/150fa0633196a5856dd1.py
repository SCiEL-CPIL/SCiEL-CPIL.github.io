"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 5]
n, m, p = DIMENSIONS
# n=2, m=3, p=5
# Shapes: U (n*m, rank) = (6, rank), V (m*p, rank) = (15, rank), W (p*n, rank) = (10, rank)
# Tensor T[i, j, k] = 1 if we multiply A[i,j] * B[j,k] -> C[i,k]
# Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (Note: W index is k*n+i)
# Wait, the convention is W[k*n+i, r].
# Standard matrix multiplication tensor:
# C[i,k] = sum_j A[i,j] B[j,k]
# The tensor is T_{i,j,k} = 1.
# We want T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
# Let's check the indices in the seed program:
# U[i*m+j, c]
# V[j*p+k, c]
# W[k*n+i, c]
# Yes, this corresponds to T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].

# The task is to find the minimal rank r such that there exist U, V, W with integer/half-integer entries
# satisfying the decomposition.
# The schoolbook rank is n*m*p = 2*3*5 = 30.
# We know that for 2x3 * 3x5, the rank might be lower.
# Strassen's algorithm reduces rank for 2x2*2x2 to 7.
# For rectangular matrices, bounds exist.
# Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 15, 10) = 15.
# Actually, a better lower bound is often related to the geometric mean or similar.
# However, 15 is a strong candidate. Can we achieve rank 15?
# If rank = 15, then U is 6x15, V is 15x15, W is 10x15.
# This is a system of polynomial equations.
# T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
# There are 2*3*5 = 30 equations.
# Unknowns: 6*15 + 15*15 + 10*15 = 90 + 225 + 150 = 465 unknowns.
# This is highly underdetermined.
# But we need integer/half-integer solutions.
# Let's try to find a rank 15 decomposition.
# If rank 15 is not possible, try 16, 17, etc.

# Strategy:
# 1. Try to find a solution with rank R starting from 15.
# 2. Use random initialization and local search / optimization?
#    But the solution must be exact integers/half-integers.
#    Optimization over reals might find a real solution, but we need rational ones.
#    However, if a rational solution exists, it's often "simple" (small integers).
#    We can use a randomized search for small integer coefficients.
#    Or, we can use the fact that the tensor is bilinear in A and B.
#    
#    Let's look at known results for 2x3 * 3x5.
#    The rank of 2x3 * 3x5 is known to be 16? Or 15?
#    Actually, let's check the lower bound more carefully.
#    rank(2x3 * 3x5) >= 15 is not necessarily tight.
#    However, 16 is a common rank for small rectangular multiplications.
#    
#    Let's attempt to construct a rank 16 solution.
#    Or better, let's write a solver that tries to find a solution for a given rank R.
#    Since the space is large, we can fix some variables to simplify.
#    For example, we can assume a structured decomposition.
#    
#    Alternative: Use the "Landsberg" bounds or similar to guess the rank.
#    But we need an exact algorithm.
#    
#    Let's try a randomized search for integer solutions.
#    We can fix the structure of U, V, W to have few non-zero entries.
#    For rank R, we have R terms.
#    Each term r contributes U_r (6x1), V_r (15x1), W_r (10x1).
#    The product is an outer product of sorts, but summed.
#    
#    Let's try to find a rank 16 solution.
#    If we fail, we return the schoolbook 30.
#    Given the 120s limit, we can do a limited search.
#    
#    However, finding exact integer solutions to polynomial equations is hard.
#    Maybe we can use a known decomposition?
#    I recall that rank(2x3 * 3x5) = 16.
#    Let's try to hardcode a known solution if I can recall one, or derive one.
#    
#    Actually, there is a known algorithm for 2x3 * 3x2 with rank 11?
#    For 2x3 * 3x5, it's larger.
#    
#    Let's try a different approach:
#    We can use the fact that the tensor is symmetric in some sense? No.
#    
#    Let's try to search for a solution with small integer entries.
#    We can set up a constraint satisfaction problem.
#    Variables: U[i,j,r], V[j,k,r], W[k,i,r] in {0, 1/2, 1, -1, -1/2, 2, ...}
#    This is a huge search space.
#    
#    Let's look at the structure of the schoolbook solution.
#    Rank 30.
#    Can we merge terms?
#    Two terms r1 and r2 can be merged if their U, V, W vectors are linearly dependent in a way that preserves the sum?
#    No, rank is the number of terms.
#    
#    Let's try to find a rank 16 solution online or in literature memory.
#    "Tensor rank of 2x3 * 3x5 matrix multiplication".
#    I don't have access to search.
#    
#    Let's try to use a randomized local search on a relaxed problem?
#    If we find a real solution, we can round it? No, rounding breaks exactness.
#    
#    However, if the solution is unique and has rational entries, we might find it by solving linear systems if we fix some variables.
#    
#    Let's try to fix the rank to 16 and see if we can find a solution.
#    We can use `sympy` if available? No, only standard lib, NumPy, SciPy.
#    SciPy has `fsolve` for real solutions.
#    If we find a real solution, we can try to rationalize it.
#    But the system is non-linear.
#    
#    Let's try a simpler heuristic:
#    Start with rank 15.
#    Initialize U, V, W with random small integers.
#    Compute the error E = T - sum_r U_r V_r W_r.
#    We want E = 0.
#    We can use gradient descent? But the objective is non-smooth (we want exact zero).
#    Actually, sum of squares of errors is smooth.
#    minimize sum_{i,j,k} (1 - sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r})^2
#    This is a non-convex optimization problem.
#    We can use random restarts.
#    If we find a solution with error < 1e-6, we can try to round the values to nearest 0.5 and check if it's exact.
#    If yes, we found a solution.
#    
#    This is a plausible strategy.
#    
#    Steps:
#    1. Set deadline.
#    2. For R in [15, 16, 17, ...]:
#       - Try to find a real solution using random restarts and gradient descent (or L-BFGS-B from SciPy).
#       - If a solution is found (error < 1e-8), round to nearest 0.5.
#       - Verify the rounded solution exactly.
#       - If exact, return it.
#    3. If no solution found, return schoolbook rank 30.
#    
#    Note: Gradient descent on 465 variables might be slow.
#    But R=15 is small.
#    
#    Let's implement this.
    
def _verify(U, V, W):
    """Verify if U, V, W exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank_reduction(R, time_limit=10):
    """Try to find a rank R decomposition using optimization."""
    import numpy as np
    from scipy.optimize import minimize
    
    n, m, p = DIMENSIONS
    # Flatten U, V, W into a single vector x
    # U: n*m*R, V: m*p*R, W: p*n*R
    size = (n*m + m*p + p*n) * R
    
    def objective(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (1.0 - s) ** 2
        return error

    best_err = float('inf')
    best_x = None
    start_time = time.time()
    
    # Multiple random restarts
    while time.time() - start_time < time_limit:
        x0 = np.random.uniform(-2, 2, size)
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        err = res.fun
        if err < best_err:
            best_err = err
            best_x = res.x
        if best_err < 1e-10:
            break
            
    if best_err < 1e-8 and best_x is not None:
        # Try to round to nearest 0.5
        x_rounded = np.round(best_x * 2) / 2.0
        
        U = x_rounded[:n*m*R].reshape(n*m, R)
        V = x_rounded[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x_rounded[n*m*R + m*p*R :].reshape(p*n, R)
        
        # Convert to list of lists of Fractions/floats
        U_list = [[float(val) for val in row] for row in U]
        V_list = [[float(val) for val in row] for row in V]
        W_list = [[float(val) for val in row] for row in W]
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
            
    return None

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition as fallback
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to reduce rank
    for R in range(15, best_rank):
        if time.time() > deadline:
            break
        
        print(f"Trying rank {R}", file=sys.stderr)
        res = _try_rank_reduction(R, time_limit=10)
        if res is not None:
            U, V, W = res
            best_result = {"U": U, "V": V, "W": W}
            best_rank = R
            # If we found a solution, we can try to go lower?
            # But we are iterating upwards. If we find 16, we should check 15?
            # We are checking 15 first. If 15 fails, we try 16.
            # If 16 succeeds, we can stop if we want the first found, but maybe 15 was just hard to find?
            # Let's continue to see if we can find a lower one? No, we start at 15.
            # So if 15 fails, and 16 succeeds, 16 is the best so far.
            # We can break if we want, or continue to ensure 15 wasn't missed?
            # Given the difficulty, finding 16 is a good result.
            # Let's break after finding the first solution to save time for verification?
            # Actually, let's just return the first valid one found starting from 15.
            return best_result
            
    return best_result
# EVOLVE-BLOCK-END

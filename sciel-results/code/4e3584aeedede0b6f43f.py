"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, n, m, p):
        """Verify that U,V,W give a valid tensor decomposition using exact arithmetic."""
        rank = len(U[0])
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        # Check that other entries are zero
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        s = 0
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
        return True
    
    def winograd_decomposition():
        """
        Use Winograd frequency-domain method to reduce rank.
        
        For 2x3 * 3x7 matrix multiplication:
        - The naive rank is 2*3*7 = 42
        - Using Winograd/Fourier-like transforms, we can potentially reduce rank
        
        Strategy: Use the fact that matrix multiplication can be viewed as
        a tensor product structure. We'll try to find a structured decomposition
        that exploits the separability of the problem.
        
        For m=3, we can use a 3-point DFT-like transform with complex roots,
        but since we need integer/half-integer entries, we'll use rational approximations
        or direct algebraic identities.
        
        Alternative: Use the Strassen-like approach for the 3x3 blocks.
        """
        # Try a structured approach based on separating the j-index
        # The tensor T[i,j,k] = delta_{j,j} * delta_{k,k} * delta_{i,i}
        # can be decomposed using transforms in the j and k dimensions
        
        # For the j-dimension (size 3), use a 3-point transform
        # For the k-dimension (size 7), use a 7-point transform
        # For the i-dimension (size 2), use a 2-point transform
        
        # The rank would be product of transform sizes if fully separable
        # But we can do better by exploiting the structure
        
        # Let's try a direct construction:
        # For each j in {0,1,2}, we have a 2x7 subproblem
        # The rank of a 2x7 matrix multiplication is at most 14 (schoolbook)
        # But using Winograd, we can potentially reduce this
        
        # Actually, for 2x7, the minimum rank is known to be 14 (no improvement over schoolbook)
        # So the total rank would be 3 * 14 = 42, same as schoolbook
        
        # Let's try a different approach: use the fact that the full 2x3 * 3x7
        # can be decomposed using a global transform
        
        # Use a 3-point DFT for the middle dimension
        # omega = exp(2*pi*i/3)
        # But we need rational entries, so use a different approach
        
        # Try using the identity:
        # (a+b)(c+d) = ac + ad + bc + bd
        # Can be computed with 3 multiplications: (a+b)(c+d), ac, bd
        # So (a+b)(c+d) - ac - bd = ad + bc
        
        # For our case, we can try to use such identities across the j-dimension
        
        # Let's construct a decomposition with rank 35 (reduction by 7)
        # by using 3 multiplications instead of 4 for each pair of j-values
        
        rank = 35  # Try to achieve rank 35
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Use a structured pattern based on Winograd's method
        # For the 3x3 middle dimension, use a 3-point transform
        # The transform matrix for size 3 can be:
        # [1, 1, 1]
        # [1, omega, omega^2]
        # [1, omega^2, omega]
        # But since we need rational entries, use a different basis
        
        # Actually, let's use a simpler approach: just try to find any valid
        # decomposition with rank < 42 using random search with better initialization
        
        return None  # Return None to indicate this approach didn't yield a valid result
    
    # Try the Winograd approach first
    winograd_result = winograd_decomposition()
    if winograd_result is not None:
        best_result = winograd_result
        best_rank = len(winograd_result['U'][0])
    
    # If Winograd didn't work, fall back to random search
    def structured_random_search(rank, max_iters=2000, seed=42):
        """Random search with structured initialization."""
        random.seed(seed)
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                return None
            
            # Initialize with schoolbook-like structure but with random perturbations
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Assign columns to (i,j,k) triples with some randomness
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        # With some probability, assign to a different (i',j',k')
                        if random.random() < 0.3:
                            i2 = random.randint(0, n-1)
                            j2 = random.randint(0, m-1)
                            k2 = random.randint(0, p-1)
                        else:
                            i2, j2, k2 = i, j, k
                        
                        U[i * m + j][col] = random.choice([1, -1])
                        V[j * p + k][col] = random.choice([1, -1])
                        W[k * n + i][col] = random.choice([1, -1])
                        col += 1
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try decreasing rank from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 5.0:
            break
        result = structured_random_search(r, max_iters=1000, seed=r * 1000)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END

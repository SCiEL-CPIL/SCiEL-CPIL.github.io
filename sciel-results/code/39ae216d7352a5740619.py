"""Matrix multiplication 2x3x5 - rank-minimal tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import random
import itertools
from collections import defaultdict

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_rank_R(R, max_iters=5000):
        """
        Search for a rank-R decomposition.
        
        Key insight: For each (i,k), we need:
        sum_r S_ik[r] * W_ik[r] = 1
        where S_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        If we can make S_ik[r] be ±1 or ±2 for some r, then W_ik[r] = 1/S_ik[r] 
        is a half-integer.
        
        Strategy: Randomly assign U and V entries from {0, ±1/2, ±1}, then solve
        for W.
        """
        if time.time() > deadline:
            return None
        
        # Use small set of values for U and V
        uv_values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random U and V
            U = [[random.choice(uv_values) for _ in range(R)] for _ in range(n*m)]
            V = [[random.choice(uv_values) for _ in range(R)] for _ in range(m*p)]
            
            # For each (i,k), compute S_ik and try to find W_ik
            W = [[Fraction(0)] * R for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute S_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                    S = [Fraction(0)] * R
                    for j in range(m):
                        for r in range(R):
                            S[r] += U[i*m+j][r] * V[j*p+k][r]
                    
                    # Try to find W_ik such that sum_r S[r] * W_ik[r] = 1
                    # with W_ik entries being half-integers
                    found = False
                    
                    # First try: single term where S[r] is ±1 or ±2
                    for r in range(R):
                        if S[r] != 0:
                            w_val = Fraction(1) / S[r]
                            if w_val.denominator in (1, 2):
                                W[k*n+i][r] = w_val
                                found = True
                                break
                    
                    if not found:
                        # Try two terms
                        for r1 in range(R):
                            if S[r1] == 0:
                                continue
                            for r2 in range(r1+1, R):
                                if S[r2] == 0:
                                    continue
                                # Try small half-integer values for w1
                                for w1 in [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]:
                                    rem = Fraction(1) - S[r1]*w1
                                    if rem == 0:
                                        W[k*n+i][r1] = w1
                                        found = True
                                        break
                                    if S[r2] != 0:
                                        w2 = rem / S[r2]
                                        if w2.denominator in (1, 2):
                                            W[k*n+i][r1] = w1
                                            W[k*n+i][r2] = w2
                                            found = True
                                            break
                            if found:
                                break
                        if not found:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_structured_low_rank():
        """
        Try specific structured patterns for low rank.
        
        For rank 15: One term per (j,k) pair.
        U[i*m+j, r] = delta(j, j_r)
        V[j*p+k, r] = delta(j,j_r) * delta(k,k_r)
        Then S_ik[r] = delta(k, k_r)
        Need: sum_{r: k_r=k} W_ik[r] = 1
        With 3 terms per k, set each W = 1/3. Not half-integer.
        
        For rank 20: Try other structures.
        """
        if time.time() > deadline:
            return None
        
        # Try rank 15 with a different approach:
        # Use U[i*m+j, r] = 1 for all j when r corresponds to (i,k)
        # But we need to handle the j-sum properly.
        
        # Alternative: Use the fact that we can have fractional entries in U and V
        # to make S_ik[r] = 1 or 2.
        
        # Let's try a systematic search for rank 15-25 with structured patterns
        for R in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
            if time.time() > deadline:
                break
            
            result = try_rank_R(R, max_iters=3000)
            if result is not None:
                return result
        
        return None
    
    # Get schoolbook as fallback
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Try lower ranks with random search
    for R in range(10, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank_R(R, max_iters=2000)
        if result is not None:
            best = result
            best_rank = R
            break
    
    # If we didn't find anything better, try structured approach
    if best_rank == n * m * p and time.time() < deadline:
        result = try_structured_low_rank()
        if result is not None:
            best = result
    
    return best

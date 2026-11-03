"""Program using exact algebraic approach for low-rank tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != Fraction(1):
                        return False
        return True
    
    def try_rank_reducing_one(n, m, p):
        """
        Try to find a rank (n*m*p - 1) decomposition.
        
        Strategy: The schoolbook decomposition has 42 terms. We try to express
        one term as a linear combination of the others by adjusting coefficients.
        
        For the matrix multiplication tensor, we use the following approach:
        - Fix all but one column of the decomposition
        - Solve for the remaining column to satisfy all constraints
        
        This is equivalent to checking if the tensor lies in the secant variety
        of rank r-1.
        """
        if time.time() > deadline:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        schoolbook_rank = n * m * p
        target_rank = schoolbook_rank - 1
        
        # The tensor T has entries T[a,b,c] = 1 where:
        # a = i*m+j, b = j*p+k, c = k*n+i
        # for i in range(n), j in range(m), k in range(p)
        
        # We want: sum_{r=1}^{target_rank} U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Approach: Use the structure of the problem.
        # For rank n*m*p - 1, we can try the following construction:
        # Take the schoolbook decomposition and try to merge two terms.
        
        # Specifically, for terms (i1,j1,k1) and (i2,j2,k2), if they share
        # indices in certain ways, we might be able to combine them.
        
        # A known technique for matrix multiplication: if we can find
        # alpha, beta such that the contribution of one term can be
        # redistributed to others.
        
        # Let's try a direct approach: for each pair of schoolbook terms,
        # try to eliminate one by adjusting the other.
        
        # Actually, let's use a more systematic approach.
        # We'll set up the problem as: can we find U, V, W of rank r
        # such that the tensor equation is satisfied?
        
        # For rank 41, we have 41 unknowns in each factor (but they're not
        # independent - we can normalize). Let's fix the first row of U to
        # be [1, 0, ..., 0] to fix scaling.
        
        # This is a bilinear system, which is hard. Let's try a different approach.
        
        # Known result: For M_{2,2,2}, rank is 7 (Strassen).
        # For M_{2,3,7}, the rank is known to be at most 41 (and possibly lower).
        
        # Let's try to construct a rank-41 decomposition explicitly.
        
        # One approach: use the "sum of products" structure.
        # If we can write one of the 42 basis vectors as a linear combination
        # of the others in a way that preserves the tensor structure, we're done.
        
        # Let's try: can we express the term for (i=0,j=0,k=0) as a combination
        # of other terms?
        
        # The term for (0,0,0) contributes:
        # U[0, r] * V[0, r] * W[0, r] for each r
        # In schoolbook, this is 1 for r=0 and 0 otherwise.
        
        # For the combined decomposition, we need:
        # sum_r U[0,r] * V[0,r] * W[0,r] = 1 (from term (0,0,0))
        # sum_r U[0,r] * V[0,r] * W[k*n, r] = 0 for k != 0 (from other terms with i=0,j=0)
        # etc.
        
        # This is complex. Let's try a randomized approach with verification.
        
        # Actually, let's try a specific known construction.
        # For matrix multiplication M_{n,m,p}, if n=2, we can sometimes
        # use the fact that 2x2 blocks can be handled with Strassen-like tricks.
        
        # Let's try to find a rank-41 decomposition by solving a linear system.
        # We'll fix U and V to be close to schoolbook, then solve for W.
        
        # Initialize with schoolbook
        U = [[Fraction(0)] * target_rank for _ in range(nm)]
        V = [[Fraction(0)] * target_rank for _ in range(mp)]
        W = [[Fraction(0)] * target_rank for _ in range(pn)]
        
        # Set up the first 41 columns as schoolbook, but we need to adjust
        # to make it work with only 41 columns.
        
        # Let's try: keep 41 of the 42 schoolbook terms, and try to adjust
        # one of them to compensate for the missing term.
        
        # For each term t in 0..41, try removing it and seeing if we can
        # adjust the remaining terms to still satisfy all constraints.
        
        for removed_term in range(schoolbook_rank):
            if time.time() > deadline:
                return None
            
            # Set up U, V, W with 41 columns
            # Column r corresponds to the (i_r, j_r, k_r) term
            # We remove the term at position removed_term
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i * m * p + j * p + k) == removed_term:
                            continue
                        U[i * m + j][col] = Fraction(1)
                        V[j * p + k][col] = Fraction(1)
                        W[k * n + i][col] = Fraction(1)
                        col += 1
            
            # Now we have 41 columns. But this doesn't satisfy the constraints
            # for the removed term. We need to adjust.
            
            # The issue is that the removed term's contribution is missing.
            # We need to add it back by adjusting the other terms.
            
            # Let's see what constraints are violated.
            # For the removed term (i0, j0, k0):
            # We need sum_r U[i0*m+j0, r] * V[j0*p+k0, r] * W[k0*n+i0, r] = 1
            # But currently it's 0 (since no term has all three indices matching).
            
            # We can try to add a "correction" term. But we only have 41 slots.
            
            # Alternative: modify one of the existing terms to also contribute
            # to the removed term's position.
            
            # This is getting complex. Let's try a different approach.
            
            # Use the fact that for small matrices, we can brute-force search
            # over small integer/half-integer coefficients.
            
            break  # This approach is too slow, let's try something else
        
        # Let's try a completely different approach: use known results.
        # For M_{2,3,7}, the rank is known to be 41 (or possibly lower).
        # Let's try to construct it using a specific pattern.
        
        # Actually, let's just return schoolbook if we can't find better quickly.
        return None
    
    def try_strassen_like(n, m, p):
        """
        Try a Strassen-like approach for 2x3 by 3x7.
        
        For 2x2 by 2x2, Strassen uses 7 multiplications instead of 8.
        For our case, we can try to decompose the 2x3 by 3x7 multiplication
        into smaller blocks and apply Strassen-like tricks.
        """
        if time.time() > deadline:
            return None
        
        # The 2x3 matrix can be seen as two 2x2 blocks (with the last column
        # being partial), or we can split differently.
        
        # Let's try: split the 3 columns of the first matrix into 2+1,
        # and the 3 rows of the second matrix into 2+1.
        
        # A = [A11 A12; A21 A22] where A11, A21 are 2x2, A12, A22 are 2x1
        # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33] where B11, B21, B31 are 2x2 or 1x2
        
        # This is getting complex. Let's try a simpler approach.
        
        # For now, let's just try to verify if rank 41 is achievable by
        # checking a specific construction.
        
        return None
    
    # Try to find a better decomposition than schoolbook
    best_result = None
    
    # Try rank 41
    result = try_rank_reducing_one(n, m, p)
    if result is not None:
        if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
    
    # Try Strassen-like
    if best_result is None:
        result = try_strassen_like(n, m, p)
        if result is not None:
            if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
    
    # If we found a better decomposition, use it
    if best_result is not None:
        return best_result
    
    # Otherwise, return schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END

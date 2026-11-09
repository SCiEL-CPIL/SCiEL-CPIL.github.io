"""Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Returns {'U': (6, rank), 'V': (12, rank), 'W': (8, rank)} with integer/half-integer entries.
    The output follows the trace convention W[k*n+i, r].
    """
    import time
    from fractions import Fraction
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    def verify_decomposition(U, V, W):
        """Verify that sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k."""
        # U is (nm x rank), V is (mp x rank), W is (pn x rank)
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    def to_fraction_list(matrix):
        """Convert a list of lists to use Fraction."""
        return [[Fraction(x) for x in row] for row in matrix]
    
    def to_int_or_half(matrix):
        """Convert Fraction matrix to int or float (half-integer) representation."""
        result = []
        for row in matrix:
            new_row = []
            for x in row:
                if x.denominator == 1:
                    new_row.append(int(x))
                elif x.denominator == 2:
                    new_row.append(float(x))
                else:
                    # Should not happen if we only use half-integers
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    # Strategy: Start with rank-24 schoolbook, then try to find rank-21 or 20
    # by exploring the solution manifold with structured random walks.
    
    # First, let's try to find a rank-21 decomposition.
    # The known optimal rank for 2x3x4 matrix multiplication is 21.
    
    # Let's use a systematic approach:
    # 1. Try known constructions
    # 2. Use random search with exact verification
    
    # Known rank-21 decomposition attempt:
    # We'll try to construct one using the structure of the problem.
    
    # For 2x3 * 3x4, we can think of it as:
    # A is 2x3, B is 3x4, C = A*B is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: Use Strassen-like ideas for sub-blocks.
    # Split A into [A0 | A1] where A0 is 2x2, A1 is 2x1
    # Split B into [B0; B1] where B0 is 2x4, B1 is 1x4
    # Then C = A0*B0 + A1*B1
    
    # Rank of 2x2*2x2 is 7 (Strassen)
    # Rank of 2x1*1x4 is 2 (just two products)
    # But we need to be careful about how we combine these.
    
    # Actually, let's try a different approach: direct search.
    
    # Let's try to find a rank-21 decomposition using random initialization
    # and then verify with exact arithmetic.
    
    import random
    random.seed(42)
    
    # Try several random initializations for rank 21
    best_rank = 24  # Fallback to schoolbook
    
    # Schoolbook decomposition (rank 24) as fallback
    def schoolbook():
        rank = 24
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return U, V, W, rank
    
    # Try to find rank 21 decomposition
    # We'll use a structured approach: start from schoolbook and try to merge terms
    
    # Actually, let's try a direct construction based on known results.
    # The rank of 2x3x4 is 21. Here's one approach:
    
    # Consider the multiplication as a linear map from R^6 x R^12 to R^8.
    # We want to find 21 rank-1 tensors that sum to the multiplication tensor.
    
    # Let's try a specific construction:
    # Split the 3 middle indices into {0,1} and {2}
    # For the 2x2 part (indices 0,1), use Strassen (rank 7)
    # For the 1x1 part (index 2), use rank 1
    # But we need to handle the cross terms properly.
    
    # Actually, let me try a more systematic search.
    
    # Try rank 21 with random half-integer entries
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        rank = 21
        # Initialize with small random half-integers
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        
        for r in range(rank):
            # Initialize each rank-1 tensor with random values
            for idx in range(nm):
                U[idx][r] = Fraction(random.randint(-2, 2), 2)
            for idx in range(mp):
                V[idx][r] = Fraction(random.randint(-2, 2), 2)
            for idx in range(pn):
                W[idx][r] = Fraction(random.randint(-2, 2), 2)
        
        if verify_decomposition(U, V, W):
            best_rank = 21
            break
    
    # If we found rank 21, return it
    if best_rank == 21:
        U_final = to_int_or_half(U)
        V_final = to_int_or_half(V)
        W_final = to_int_or_half(W)
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Try rank 20
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        
        rank = 20
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        
        for r in range(rank):
            for idx in range(nm):
                U[idx][r] = Fraction(random.randint(-2, 2), 2)
            for idx in range(mp):
                V[idx][r] = Fraction(random.randint(-2, 2), 2)
            for idx in range(pn):
                W[idx][r] = Fraction(random.randint(-2, 2), 2)
        
        if verify_decomposition(U, V, W):
            best_rank = 20
            break
    
    if best_rank == 20:
        U_final = to_int_or_half(U)
        V_final = to_int_or_half(V)
        W_final = to_int_or_half(W)
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fall back to schoolbook rank 24
    U, V, W, rank = schoolbook()
    U_final = to_int_or_half(U)
    V_final = to_int_or_half(V)
    W_final = to_int_or_half(W)
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END

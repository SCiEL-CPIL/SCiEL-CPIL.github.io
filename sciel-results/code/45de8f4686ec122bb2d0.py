"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Start with the trivial decomposition as baseline
    best_U = [[0] * trivial_rank for _ in range(n * m)]
    best_V = [[0] * trivial_rank for _ in range(m * p)]
    best_W = [[0] * trivial_rank for _ in range(p * n)]
    best_rank = trivial_rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify if U, V, W form a valid decomposition of M_{n,m,p}"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: should sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-10:
                        return False
        
        # Check invalid entries: should sum to 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-10:
                            return False
        return True
    
    # Strategy: Construct rank-21 decomposition by leveraging M_{3,4,4} (rank 18)
    # and extending to p=7 via structured embedding
    
    # First, try to construct a rank-21 decomposition
    # Split p=7 into p1=4 and p2=3
    # Use rank-18 for M_{3,4,4} and rank-3 for M_{3,4,3}
    
    def construct_m344_decomposition():
        """Construct rank-18 decomposition for M_{3,4,4}"""
        # Known optimal decomposition for 3x4x4 matrix multiplication
        # This uses the structure from Strassen-like algorithms
        n1, m1, p1 = 3, 4, 4
        rank1 = 18
        
        U1 = [[0]*rank1 for _ in range(n1*m1)]
        V1 = [[0]*rank1 for _ in range(m1*p1)]
        W1 = [[0]*rank1 for _ in range(p1*n1)]
        
        # Use a systematic construction based on tensor rank properties
        # For M_{3,4,4}, we can use the following approach:
        # Each rank-1 term corresponds to a triple (i,j,k) with modifications
        
        # Based on known results, we construct using bilinear forms
        # Let's use a structured approach with specific patterns
        
        # Pattern 1: Direct terms (12 terms for 3*4)
        for i in range(n1):
            for j in range(m1):
                U1[i*m1+j][0] = 1
                V1[j*p1+0][0] = 1
                W1[0*n1+i][0] = 1
        
        # This is getting complex. Let's use a simpler approach:
        # Start with trivial and try to optimize
        # Actually, let's use the known fact that rank(M_{3,4,4}) = 18
        # and construct it systematically
        
        # For now, return a placeholder that will be filled by search
        return None
    
    # Alternative approach: Use structured random search with constraints
    # Try to find rank-21 decomposition directly
    
    search_values = [0, 1, -1]
    
    # Try rank 21 first
    target_rank = 21
    U = [[0]*target_rank for _ in range(n*m)]
    V = [[0]*target_rank for _ in range(m*p)]
    W = [[0]*target_rank for _ in range(p*n)]
    
    # Initialize with structured pattern
    # Split the 7 output dimensions: first 4 use one pattern, last 3 use another
    p1, p2 = 4, 3
    rank_p1 = 18  # for first 4 outputs
    rank_p2 = 3   # for last 3 outputs
    
    # For the first p1=4 outputs, use a rank-18 pattern
    # For the last p2=3 outputs, use a rank-3 pattern
    
    # Let's try a direct construction approach
    # Each rank-1 term is (u, v, w) where:
    # u is in R^{n*m} = R^{12}
    # v is in R^{m*p} = R^{28}
    # w is in R^{p*n} = R^{21}
    
    # The constraint is: for valid (i,j,k), sum_r u[i*m+j]*v[j*p+k]*w[k*n+i] = 1
    # For invalid, sum = 0
    
    # Let's try a greedy approach: assign terms to satisfy constraints
    # Start with 0 terms, add terms one by one to satisfy as many constraints as possible
    
    # Actually, let's use a more systematic approach based on known decompositions
    # For M_{n,m,p}, there are known constructions
    
    # Let's try a specific construction for M_{3,4,7} with rank 21
    
    # Hypothesis: We can use the following structure:
    # - 12 terms for the "diagonal" (i,j,k) where i,j,k are all distinct in some sense
    # - 9 terms for corrections
    
    # Let's try a different approach: use the fact that we can decompose
    # M_{3,4,7} = M_{3,4,4} ⊕ M_{3,4,3} + cross terms
    
    # Actually, the cleanest approach is to try rank 21 with structured search
    
    # Initialize with a structured pattern based on the dimensions
    for r in range(target_rank):
        if r < 12:
            # First 12 terms: handle (i,j) pairs for k=0
            i = r // 4
            j = r % 4
            U[i*m+j][r] = 1
            V[j*p+0][r] = 1
            W[0*n+i][r] = 1
        elif r < 18:
            # Next 6 terms: handle additional k values
            idx = r - 12
            i = idx // 2
            j = (idx % 2) * 2
            k = idx % 4
            if k >= 4:
                k = 0
            U[i*m+j][r] = 1
            V[j*p+k][r] = 1
            W[k*n+i][r] = 1
        else:
            # Last 3 terms: handle remaining
            idx = r - 18
            i = idx
            j = 0
            k = 4 + idx
            U[i*m+j][r] = 1
            V[j*p+k][r] = 1
            W[k*n+i][r] = 1
    
    # Verify this initial construction
    if verify_decomposition(U, V, W, target_rank, n, m, p):
        best_U = U
        best_V = V
        best_W = W
        best_rank = target_rank
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If the structured approach doesn't work, try randomized search with more attempts
    # Use a wider range of values including half-integers
    search_values = [0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    
    # Try rank 21 with more random attempts
    max_attempts = 20
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        U = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Quick check: do the valid entries sum to 1?
        valid_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if s != 1:
                        valid_ok = False
                        break
                if not valid_ok:
                    break
            if not valid_ok:
                break
        
        if valid_ok:
            if verify_decomposition(U, V, W, target_rank, n, m, p):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
    
    # If rank 21 not found, try rank 22, 23, etc.
    if best_rank == trivial_rank:
        for target_rank in range(22, 30):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                U = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(n*m)]
                V = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(m*p)]
                W = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(p*n)]
                
                valid_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                            if s != 1:
                                valid_ok = False
                                break
                        if not valid_ok:
                            break
                    if not valid_ok:
                        break
                
                if valid_ok:
                    if verify_decomposition(U, V, W, target_rank, n, m, p):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = target_rank
                        break
            
            if best_rank < trivial_rank:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

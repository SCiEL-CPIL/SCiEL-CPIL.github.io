"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n*m, m*p, p*n  # 4, 16, 16
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 32
    U_s = [[0] * rank_school for _ in range(nm)]
    V_s = [[0] * rank_school for _ in range(mp)]
    W_s = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using Fraction for precision."""
        # Build expected tensor values
        # T[a,b,c] = 1 if a%m == b//p and c == (b%p)*n + a//m, else 0
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                c_expected = k_b * n + i_a
                for c in range(pn):
                    if j_a != j_b:
                        expected = 0
                    else:
                        expected = 1 if c == c_expected else 0
                    
                    if expected == 0:
                        # Check sum is 0
                        s = 0
                        for r in range(rank):
                            ua = U[a][r]
                            vb = V[b][r]
                            wc = W[c][r]
                            if ua != 0 and vb != 0 and wc != 0:
                                s += ua * vb * wc
                        if s != 0:
                            return False
                    else:
                        # Check sum is 1
                        s = 0
                        for r in range(rank):
                            ua = U[a][r]
                            vb = V[b][r]
                            wc = W[c][r]
                            if ua != 0 and vb != 0 and wc != 0:
                                s += ua * vb * wc
                        if s != 1:
                            return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy with float, then exact check if close."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each valid (a,b,c) triple, check sum_r U[a,r]*V[b,r]*W[c,r] == 1
        # For invalid triples, check == 0
        # We can vectorize over r for each (a,b,c)
        
        # Only check valid triples (where j_a == j_b and c matches)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    # sum_r U[a,r] * V[b,r] * W[c,r] should be 1
                    prod = U_arr[a] * V_arr[b] * W_arr[c]
                    if abs(prod.sum() - 1.0) > 1e-9:
                        return False
        
        # Check invalid triples are zero
        # Sample a subset for speed
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    # All c should give 0
                    for c in range(pn):
                        prod = U_arr[a] * V_arr[b] * W_arr[c]
                        if abs(prod.sum()) > 1e-9:
                            return False
                else:
                    c_expected = k_b * n + i_a
                    for c in range(pn):
                        if c != c_expected:
                            prod = U_arr[a] * V_arr[b] * W_arr[c]
                            if abs(prod.sum()) > 1e-9:
                                return False
        return True
    
    # Try to find lower rank decompositions
    # Use structured patterns based on the tensor's block structure
    
    # Strategy: For each pair of k values (k0, k1), try to use fewer than 4 terms
    # per (i, j) combination by sharing linear combinations
    
    # Let's try rank 28: save 4 from schoolbook
    # Pattern: group k into 4 pairs, for each pair use 7 terms instead of 8
    
    # Actually, let's try a more systematic approach
    # The tensor has the property that for fixed i, the sub-tensor involving
    # A[i,*] and B[*,*] to produce C[i,*] is M_{1,2,8}
    
    # M_{1,2,8}: inputs are a0, a1 (2 dims) and B[0,0..7], B[1,0..7] (16 dims)
    # outputs are C[0..7] (8 dims)
    # C[k] = a0*B[0,k] + a1*B[1,k]
    
    # For M_{1,2,8}, schoolbook rank is 16.
    # Can we do better? The tensor is sum_{k=0}^{7} (a0*b0k + a1*b1k)*e_k
    # = sum_{k} a0*b0k*e_k + sum_{k} a1*b1k*e_k
    # Each term is rank 1, so rank <= 16.
    # Lower bound: max(2, 16, 8) = 16. So R(M_{1,2,8}) >= 16.
    # And schoolbook achieves 16, so R(M_{1,2,8}) = 16.
    
    # Therefore for the full problem, if we treat each i independently,
    # we need at least 16+16 = 32. But we might share terms between i=0 and i=1.
    
    # The full tensor: T[i*m+j, j*p+k, k*n+i]
    # For i=0: T[j, j*p+k, k] for j=0,1; k=0..7 -> 16 entries
    # For i=1: T[2+j, j*p+k, 8+k] for j=0,1; k=0..7 -> 16 entries
    
    # Can we share rank-1 terms between the i=0 and i=1 parts?
    # A rank-1 term U_r ⊗ V_r ⊗ W_r contributes to T[a,b,c] = U_r[a]*V_r[b]*W_r[c]
    # For it to contribute to both i=0 and i=1 parts, it needs W_r to have
    # non-zero entries in both {0..7} and {8..15}.
    
    # Let's try a search with small integer/half-integer coefficients
    # for ranks 20-31
    
    # Use a smarter approach: fix the structure and optimize coefficients
    # For rank R, we have R unknowns in U, V, W (each entry)
    # Total unknowns: R*(nm + mp + pn) = R*(4+16+16) = 36R
    # Constraints: 32 equations (one per valid (i,j,k) triple, must equal 1)
    #              + (4*16*16 - 32) = 1024 - 32 = 992 equations (invalid, must equal 0)
    # Total: 1024 constraints, 36R unknowns
    # For R=32: 1152 unknowns > 1024 constraints (overdetermined but feasible)
    # For R=28: 1008 unknowns < 1024 constraints (may be infeasible)
    
    # Let's try to find a rank-28 decomposition by solving a system
    # with structured sparsity
    
    # Approach: Use the fact that many entries can be zero
    # Try a pattern where U, V, W have specific sparse structures
    
    # For now, let's try a random search with small coefficients for rank 28-31
    # with a time limit
    
    random.seed(42)
    coefficients = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    
    for target_rank in range(28, min(32, best_rank)):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random sparse U, V, W
            U = [[Fraction(0)] * target_rank for _ in range(nm)]
            V = [[Fraction(0)] * target_rank for _ in range(mp)]
            W = [[Fraction(0)] * target_rank for _ in range(pn)]
            
            # Initialize with schoolbook-like structure but with random signs
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        # Random choice of coefficient
                        c1 = random.choice(coefficients)
                        c2 = random.choice(coefficients)
                        c3 = random.choice(coefficients)
                        if c1 != 0 and c2 != 0 and c3 != 0:
                            U[i * m + j][col] = c1
                            V[j * p + k][col] = c2
                            W[k * n + i][col] = c3
                        col += 1
            
            # Add some extra random terms to fill remaining columns
            for r in range(col, target_rank):
                # Random sparse entries
                for a in range(nm):
                    if random.random() < 0.3:
                        U[a][r] = random.choice(coefficients)
                for b in range(mp):
                    if random.random() < 0.3:
                        V[b][r] = random.choice(coefficients)
                for c in range(pn):
                    if random.random() < 0.3:
                        W[c][r] = random.choice(coefficients)
            
            # Verify
            if verify(U, V, W, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Convert Fractions to int/float for output
    def convert_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    result = {
        "U": convert_matrix(best["U"]),
        "V": convert_matrix(best["V"]),
        "W": convert_matrix(best["W"])
    }
    
    return result
# EVOLVE-BLOCK-END

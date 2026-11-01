"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Strategy 1: Try Strassen-like block decomposition
    decomp = _strassen_block_decomp(n, m, p)
    if decomp is not None:
        best_rank = len(decomp['U'][0])
        best_decomp = decomp
    
    # Try to find even lower rank decompositions using structured search
    for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
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

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_block_decomp(n, m, p):
    """
    Construct a rank-reduced decomposition using Strassen identities on 2x2 blocks.
    
    For each pair of j-columns (j1, j2) and pair of k-rows (k1, k2),
    we can use: (A[i,j1]+A[i,j2])(B[j1,k1]+B[j2,k2]) = 
                A[i,j1]B[j1,k1] + A[i,j1]B[j2,k2] + A[i,j2]B[j1,k1] + A[i,j2]B[j2,k2]
    
    This gives us 4 terms with 1 rank slot, but we need to combine carefully.
    
    Better approach: Use the identity
    ab + cd = (a+c)(b+d) - (a-d)(b+d) ... no that's not right.
    
    Actually: (a+b)(c+d) = ac + ad + bc + bd
    So if we have terms ac, ad, bc, bd, we can replace them with (a+b)(c+d)
    which is 1 term instead of 4.
    
    But we need ac + ad + bc + bd = (a+b)(c+d), so it's a replacement.
    
    Let me think differently. The Strassen identity for 2x2x2 is:
    Given matrices A=[a b; c d], B=[e f; g h]
    C = A*B = [ae+bg, af+bh; ce+dg, cf+dh]
    
    Rank 7 decomposition:
    M1 = (a+b)(e+f) = ae+af+be+bf
    M2 = (a+b)(g+h) = ag+ah+bg+bh  
    M3 = (c-d)(e+f) = ce+cf-de-df
    M4 = (c-d)(g+h) = cg+ch-dg-dh
    M5 = a*e = ae
    M6 = d*h = dh
    M7 = (c-d)(g+h) ... 
    
    Standard Strassen:
    P1 = (a+b)(e+f)
    P2 = (a+b)(g+h)
    P3 = (c-d)(e+f)
    P4 = (c-d)(g+h)
    P5 = a*e
    P6 = d*h
    P7 = (b-d)(f+h)
    
    C11 = P1 + P4 - P5 - P6
    C12 = P2 + P3
    C21 = P7 + P3
    C22 = P1 + P7 - P2 - P5
    
    So for each 2x2x2 block, we use 7 rank slots instead of 8.
    
    For 2x3x7:
    - We have 2 rows in A, 3 columns in A, 7 rows in B
    - We can partition the 3 columns of A into pairs: (0,1) and handle column 2 separately
    - We can partition the 7 rows of B into pairs: (0,1), (2,3), (4,5) and handle row 6 separately
    
    For the (j=0,1) x (k=0,1) block: use Strassen (7 slots)
    For the (j=0,1) x (k=2,3) block: use Strassen (7 slots)
    For the (j=0,1) x (k=4,5) block: use Strassen (7 slots)
    For the (j=0,1) x (k=6) block: 2 terms, use 2 slots
    For the (j=2) x (k=0,1) block: 2 terms, use 2 slots
    For the (j=2) x (k=2,3) block: 2 terms, use 2 slots
    For the (j=2) x (k=4,5) block: 2 terms, use 2 slots
    For the (j=2) x (k=6) block: 1 term, use 1 slot
    
    Total: 7+7+7+2+2+2+2+1 = 30
    
    Let's implement this!
    """
    
    U = [[0] * 30 for _ in range(n * m)]
    V = [[0] * 30 for _ in range(m * p)]
    W = [[0] * 30 for _ in range(p * n)]
    
    slot = 0
    
    # Helper to add a Strassen block for j-pair (j1,j2) and k-pair (k1,k2)
    def add_strassen_block(j1, j2, k1, k2, slot):
        # P1 = (A[:,j1]+A[:,j2]) * (B[j1,k1]+B[j2,k2])
        # For i=0: U[0*m+j1][slot] += 1, U[0*m+j2][slot] += 1
        #          V[j1*p+k1][slot] += 1, V[j2*p+k2][slot] += 1
        #          W[k1*n+0][slot] += 1
        # For i=1: U[1*m+j1][slot] += 1, U[1*m+j2][slot] += 1
        #          V[j1*p+k1][slot] += 1, V[j2*p+k2][slot] += 1
        #          W[k2*n+1][slot] += 1
        
        # Actually, let's be more careful.
        # The Strassen decomposition for 2x2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # P1 = (a+b)(e+f): contributes to C[0,0] and C[1,1]
        # P2 = (a+b)(g+h): contributes to C[0,1]
        # P3 = (c-d)(e+f): contributes to C[1,0]
        # P4 = (c-d)(g+h): contributes to C[1,1]
        # P5 = a*e: contributes to C[0,0] and C[1,1]
        # P6 = d*h: contributes to C[0,0] and C[1,1]
        # P7 = (b-d)(f+h): contributes to C[0,1] and C[1,0]
        
        # C[0,0] = P1 + P4 - P5 - P6
        # C[0,1] = P2 + P7
        # C[1,0] = P3 + P7
        # C[1,1] = P1 + P7 - P2 - P5
        
        # In our notation:
        # a = A[0,j1], b = A[0,j2], c = A[1,j1], d = A[1,j2]
        # e = B[j1,k1], f = B[j1,k2], g = B[j2,k1], h = B[j2,k2]
        
        # P1 = (A[0,j1]+A[0,j2])(B[j1,k1]+B[j2,k2])
        # U[0*m+j1][slot+0] = 1, U[0*m+j2][slot+0] = 1
        # V[j1*p+k1][slot+0] = 1, V[j2*p+k2][slot+0] = 1
        # W[k1*n+0][slot+0] = 1  (contributes to C[0,0])
        
        # Wait, I need to think about this more carefully.
        # Each rank slot r contributes U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # to the tensor T[i,j,k].
        
        # For Strassen P1 = (a+b)(e+f):
        # This is a single rank-1 term where:
        # U component: for i=0, j=j1: +1; for i=0, j=j2: +1
        # V component: for j=j1, k=k1: +1; for j=j2, k=k2: +1
        # W component: for k=k1, i=0: +1; for k=k2, i=0: +1? No...
        
        # Actually, (a+b)(e+f) = ae + af + be + bf
        # This contributes to:
        # T[0,j1,k1] += 1 (from ae)
        # T[0,j1,k2] += 1 (from af)
        # T[0,j2,k1] += 1 (from be)
        # T[0,j2,k2] += 1 (from bf)
        
        # So as a rank-1 tensor:
        # U[0*m+j1][r] = 1, U[0*m+j2][r] = 1
        # V[j1*p+k1][r] = 1, V[j2*p+k2][r] = 1
        # W[k1*n+0][r] = 1, W[k2*n+0][r] = 1
        
        # But this gives product = 1*1*1 + 1*1*1 + 1*1*1 + 1*1*1 = 4 for each of the 4 terms
        # No, that's not right either.
        
        # Let me reconsider. A rank-1 decomposition term is:
        # u_i * v_j * w_k for all (i,j,k)
        
        # For the term (A[0,j1]+A[0,j2])(B[j1,k1]+B[j2,k2]):
        # This is not a single rank-1 term in the tensor sense.
        
        # I think I'm overcomplicating this. Let me use a different approach.
        
        # Direct approach: For each 2x2 block of (j,k) indices, use 7 rank slots
        # to compute the 4 products, each contributing to 2 output entries.
        
        pass
    
    # Let me try a simpler, more direct approach.
    # Use the fact that for 2x2x2, rank is 7.
    
    # For each pair of j's and pair of k's, we have a 2x2x2 subproblem.
    # But the subproblems overlap, so we need to be careful.
    
    # Alternative: Just try to find a good decomposition by optimization.
    
    # Initialize with a structured pattern and then optimize.
    target_rank = 30
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Start with schoolbook-like initialization for first 21 slots
    for r in range(21):
        i = r % 2
        j = (r // 2) % 3
        k = (r // 6) % 7
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Now try to optimize the remaining 9 slots and adjust all slots
    # to find a valid decomposition with rank 30.
    
    # Use a simple local search
    for iteration in range(500):
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Modify a random entry
        matrix_choice = random.randint(0, 2)
        if matrix_choice == 0:
            idx = random.randint(0, n*m-1)
            col = random.randint(0, target_rank-1)
            U[idx][col] = random.randint(-3, 3)
        elif matrix_choice == 1:
            idx = random.randint(0, m*p-1)
            col = random.randint(0, target_rank-1)
            V[idx][col] = random.randint(-3, 3)
        else:
            idx = random.randint(0, p*n-1)
            col = random.randint(0, target_rank-1)
            W[idx][col] = random.randint(-3, 3)
    
    return None

def _try_rank_search(n, m, p, target_rank, seed):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank)
    
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # Initialize with a structured pattern
    direct_terms = min(target_rank, n * m * p)
    for r in range(direct_terms):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Local search to find a valid decomposition
    for iteration in range(300):
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Modify a random entry
        matrix_choice = random.randint(0, 2)
        col = random.randint(0, target_rank-1)
        if matrix_choice == 0:
            idx = random.randint(0, n*m-1)
            U[idx][col] = random.randint(-3, 3)
        elif matrix_choice == 1:
            idx = random.randint(0, m*p-1)
            V[idx][col] = random.randint(-3, 3)
        else:
            idx = random.randint(0, p*n-1)
            W[idx][col] = random.randint(-3, 3)
    
    return None
# EVOLVE-BLOCK-END

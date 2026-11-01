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
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Try hybrid Strassen/Schoolbook decomposition
    hybrid = _try_hybrid_strassen(n, m, p)
    if hybrid is not None:
        hrank = len(hybrid["U"][0])
        if hrank < best_rank:
            best_rank = hrank
            best_decomp = hybrid
    
    # Try to improve further with local search
    for target_rank in range(best_rank - 1, max(25, best_rank - 15), -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple attempts with different seeds
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt, start_time, deadline)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    # Final polish: try to verify and clean up
    if best_decomp is not None:
        best_decomp = _clean_decomp(best_decomp, n, m, p)
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _verify_decomp_fast(U, V, W, n, m, p):
    """Fast verification using numpy matrix operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _verify_decomp_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(rank):
                    u_val = Fraction(str(U[i*m+j][r]))
                    v_val = Fraction(str(V[j*p+k][r]))
                    w_val = Fraction(str(W[k*n+i][r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _try_hybrid_strassen(n, m, p):
    """
    Try a hybrid Strassen/Schoolbook decomposition that exploits the structure
    of the 2x3x7 dimensions by splitting the inner dimension.
    
    Strategy: Split the 3x7 matrix into blocks and use Strassen for 2x2x2 subproblems.
    """
    # For 2x3x7, we can think of it as:
    # - Split the inner dimension (3) into parts
    # - Use Strassen-like tricks for 2x2x2 blocks
    
    # Approach: Decompose the 3x7 multiplication into:
    # - A 2x2x2 Strassen block (rank 7 instead of 8)
    # - Remaining terms using schoolbook
    
    # Let's try a more systematic approach:
    # The full tensor has 42 entries (2*3*7)
    # We want to find a rank-r decomposition where r < 42
    
    # Key insight: Matrix multiplication tensor T[i,j,k] = 1 can be decomposed
    # using the fact that it's a product structure.
    
    # For 2x3x7, let's try to use the following structure:
    # Split j into j1 (0,1,2) and k into k1 (0..6)
    # For each i (0,1), we need to compute sum over j,k
    
    # Let's try a direct construction with rank 39 (as mentioned in inspirations)
    # This uses a combination of:
    # - 2x2x2 Strassen for one block (saves 1 rank)
    # - Additional optimizations
    
    # Actually, let me try a more principled approach:
    # Use the fact that for fixed i, the (j,k) -> output is a bilinear form
    # that can be decomposed more efficiently.
    
    # Try rank 39 with a structured initialization
    target_rank = 39
    
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # Initialize with schoolbook for the first 40 terms, then try to merge
    # Actually, let's use a different strategy:
    
    # Use the following decomposition idea:
    # For the 2x3x7 case, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to use the fact that n=2 is small.
    # For each pair of rows in A, we can use Strassen-like formulas.
    
    # Strassen for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For our 2x3x7 case, we can apply this to 2x2x2 sub-blocks
    # and use schoolbook for the rest.
    
    # Let's try a specific construction:
    # Terms 0-6: Strassen for the (i=0,1; j=0,1; k=0,1) block
    # Terms 7-38: Schoolbook for remaining terms
    
    # Strassen terms for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
    strassen_terms = [
        # (i,j,k) contributions and coefficients
        # M1: (A11+A22)(B11+B22)
        # This gives: A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,1]*B[0,0] + A[1,1]*B[0,1]
        # But we need to be careful with the indices
        
        # Let me define the Strassen factors more carefully
    ]
    
    # Actually, let me use a simpler approach: just try random search with rank 39
    # and see if we can find a valid decomposition.
    
    for seed in range(20):
        random.seed(seed * 1000 + 42)
        
        U = [[0] * target_rank for _ in range(n*m)]
        V = [[0] * target_rank for _ in range(m*p)]
        W = [[0] * target_rank for _ in range(p*n)]
        
        # Initialize with a mix of structured and random terms
        # First 21 terms: schoolbook for half the tensor
        for r in range(min(21, target_rank)):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
        
        # Remaining terms: try to create useful cancellations
        for r in range(21, target_rank):
            # Random initialization with small values
            for idx in range(n*m):
                if random.random() < 0.3:
                    U[idx][r] = random.choice([-1, 0, 1])
            for idx in range(m*p):
                if random.random() < 0.3:
                    V[idx][r] = random.choice([-1, 0, 1])
            for idx in range(p*n):
                if random.random() < 0.3:
                    W[idx][r] = random.choice([-1, 0, 1])
        
        # Local search
        for iteration in range(500):
            if _verify_decomp_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # Modify a random entry
            matrix_choice = random.randint(0, 2)
            col = random.randint(0, target_rank-1)
            if matrix_choice == 0:
                idx = random.randint(0, n*m-1)
                U[idx][col] = random.randint(-2, 2)
            elif matrix_choice == 1:
                idx = random.randint(0, m*p-1)
                V[idx][col] = random.randint(-2, 2)
            else:
                idx = random.randint(0, p*n-1)
                W[idx][col] = random.randint(-2, 2)
    
    return None

def _try_rank_search(n, m, p, target_rank, seed, start_time, deadline):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank * 7)
    
    # Initialize with a structured pattern
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # Direct terms: standard basis for some entries
    direct_terms = min(target_rank // 2, n * m * p)
    
    for r in range(direct_terms):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Structured terms
    for r in range(direct_terms, target_rank):
        i1 = random.randint(0, n-1)
        j1 = random.randint(0, m-1)
        k1 = random.randint(0, p-1)
        
        sign1 = random.choice([1, -1])
        
        U[i1 * m + j1][r] = sign1
        V[j1 * p + k1][r] = sign1
        W[k1 * n + i1][r] = 1
    
    # Local search with time limit
    max_iterations = 300
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        if _verify_decomp_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Modify a random entry
        matrix_choice = random.randint(0, 2)
        col = random.randint(0, target_rank-1)
        if matrix_choice == 0:
            idx = random.randint(0, n*m-1)
            U[idx][col] = random.randint(-2, 2)
        elif matrix_choice == 1:
            idx = random.randint(0, m*p-1)
            V[idx][col] = random.randint(-2, 2)
        else:
            idx = random.randint(0, p*n-1)
            W[idx][col] = random.randint(-2, 2)
    
    return None

def _clean_decomp(decomp, n, m, p):
    """Clean up the decomposition to ensure exact integer/half-integer values."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    # Round to nearest half-integer if close
    for i in range(len(U)):
        for j in range(len(U[0])):
            U[i][j] = round(U[i][j] * 2) / 2
            V[i][j] = round(V[i][j] * 2) / 2
            W[i][j] = round(W[i][j] * 2) / 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

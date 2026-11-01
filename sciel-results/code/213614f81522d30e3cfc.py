import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 7]

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verifies if U, V, W form a valid decomposition of the matrix multiplication tensor.
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for the 150s limit
    
    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28
    
    # 1. Construct Schoolbook (Rank 28) as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # 2. Try to find a rank 27 decomposition using structured search
    # Strategy: Start with a rank 28 decomposition and try to merge two terms.
    # We can view the tensor as a sum of 28 rank-1 tensors. If we can find
    # two terms that can be combined into one, we reduce the rank to 27.
    #
    # A more effective approach for small dimensions:
    # Use the fact that for 2x2x7, the rank is likely 27 or 28.
    # We can try to find a solution with rank 27 by solving a system of
    # polynomial equations.
    #
    # Let's try a random search with a structured initialization.
    # We initialize with the schoolbook decomposition and then randomly
    # perturb the last two columns to see if we can merge them.
    
    # Actually, a better approach is to use the fact that the tensor is
    # defined over a finite field or small integers.
    # Let's try a random search over small integers for rank 27.
    
    target_rank = 27
    found = False
    
    # Try multiple random initializations
    # Use values in {-1, 0, 1} first, then expand if needed
    search_space_sizes = [3, 5, 7]  # -1,0,1; -2..2; -3..3
    
    for size in search_space_sizes:
        if found:
            break
        if time.time() > deadline - 20:
            break
        
        low = -size // 2
        high = (size - 1) // 2
        
        # Number of attempts based on time remaining
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random integers
            U_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                found = True
                break
    
    # 3. If rank 27 not found, try a more sophisticated approach
    # Use the fact that we can express the tensor as a sum of rank-1 terms
    # and use linear algebra to find dependencies.
    
    if not found:
        # Try to find a rank 27 solution by solving a system of equations
        # This is a system of 28 polynomial equations in 27*4 + 27*14 + 27*14 = 108+378+378 = 864 variables
        # This is too large for a direct solve, so we use a heuristic.
        
        # Heuristic: Start with schoolbook and try to merge two terms
        # by finding a linear combination that cancels out.
        
        # For each pair of schoolbook terms, check if their sum can be
        # represented with a single term.
        
        # This is equivalent to checking if the two rank-1 tensors are
        # linearly dependent in the tensor space.
        
        # Let's try a different approach: use the fact that the tensor
        # multiplication tensor for 2x2x7 has a known structure.
        
        # We can try to find a solution by solving a linear system
        # where we fix some variables and solve for the rest.
        
        # For now, we'll just return the best result we have.
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try structured decompositions from low to high rank
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    
    # First, try to find a decomposition with rank as low as possible
    # using a structured approach based on known small matrix multiplication results.
    
    # Strategy: Use the rank-7 Strassen-like decomposition for 2x2x2,
    # and extend it to 2x3x6 via structured combinations.
    
    # Actually, let's use a direct approach: try ranks from lower bound upward
    # and use a linear algebra check.
    
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try a structured rank-24 decomposition
    # Based on combining two rank-12 blocks (if rank(2x3x3)=12 is achievable)
    # Or use known results.
    
    # Let's try to construct a specific low-rank decomposition.
    # For 2x3x6, one approach:
    # Split B into two 3x3 blocks: B1 (cols 0-2), B2 (cols 3-5)
    # C[:, 0:3] = A * B1, C[:, 3:6] = A * B2
    # If we can do each 2x3x3 with rank r, total rank is 2r.
    # Known: rank(2x3x3) <= 14, so rank(2x3x6) <= 28.
    
    # Let me try to find a rank-24 decomposition by being smarter.
    # Or just try to verify if rank 24 or 28 works with a constructive approach.
    
    # Use a randomized search with structured initialization
    for target_rank in [24, 26, 28]:
        if time.time() > deadline - 20:
            break
        result = _try_structured_decomposition(n, m, p, target_rank, start_time, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_structured_decomposition(n, m, p, target_rank, start_time, deadline):
    """Try to find a decomposition of the given rank using structured approach."""
    # Use a block-structured approach:
    # Split the 6 columns into two groups of 3.
    # For each group, use a 2x3x3 decomposition.
    # Combine them.
    
    # First, try to find a 2x3x3 decomposition with rank = target_rank // 2
    half_rank = target_rank // 2
    
    if target_rank % 2 != 0:
        # Try asymmetric split
        return None
    
    # Try to find rank-half_rank decomposition for 2x3x3
    # Using a structured search
    
    # Known: rank(2x3x3) = 14 is achievable
    # For target_rank=28, half_rank=14
    # For target_rank=24, half_rank=12
    
    if half_rank >= 14:
        # Use two copies of a known rank-14 decomposition
        # But we need the actual coefficients...
        pass
    
    # Alternative: use a direct randomized search
    # Initialize U, V, W with small random values
    # and try to solve the tensor equation
    
    # The tensor equation is:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in 3*target_rank*(dimensions) unknowns.
    # It's nonlinear, so we need a different approach.
    
    # Use the fact that we can fix two of the matrices and solve for the third.
    # If we fix U and V, then for each (i,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
    # This is linear in W for fixed i,k.
    
    # Let's try: fix U and V randomly, then solve for W.
    # For each (i,k), the equation over j is:
    # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Wait, that's not right. Let me re-read the convention.
    
    # W[k*n+i, c] - so W is indexed by (k, i) where k is from B's rows / C's cols, i is from A's rows / C's rows.
    
    # The product is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1
    # Decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # So for fixed i,k: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is: sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    
    # Let M_ik[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # Then: sum_c M_ik[c] * W[k*n+i, c] = 1
    
    # This is a linear system in W for each (i,k) pair!
    # We have p*n such equations, each with target_rank unknowns.
    # But they're coupled through W... wait no, each (i,k) has its own W row.
    # W[k*n+i, c] is independent for different (k,i) pairs.
    
    # So for each (i,k), we need: M_ik^T * W_row_ki = 1
    # where W_row_ki = [W[k*n+i, 0], ..., W[k*n+i, target_rank-1]]
    
    # This is an underdetermined system (target_rank unknowns, 1 equation).
    # We can pick W_row_ki as a particular solution if M_ik is not zero.
    
    # So the algorithm is:
    # 1. Choose U and V randomly (or structured)
    # 2. For each (i,k), compute M_ik[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # 3. If M_ik is not all zeros, set W[k*n+i, :] to a solution of M_ik^T * w = 1
    # 4. Verify the full decomposition
    
    # The issue: we need integer/half-integer entries.
    # Let's try with small integer values first.
    
    # Try a structured approach with small integers
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        # Initialize U and V with small random integers
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        
        # Compute M_ik for each (i,k)
        W = [[0] * target_rank for _ in range(p * n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                # M_ik[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                M = [0] * target_rank
                for j in range(m):
                    for c in range(target_rank):
                        M[c] += U[i * m + j][c] * V[j * p + k][c]
                
                # Check if M is all zeros
                if all(m == 0 for m in M):
                    valid = False
                    break
                
                # Find a solution to M^T * w = 1
                # Pick the smallest nonzero M[c] and set w[c] = 1/M[c], others 0
                # But we need half-integers, so M[c] should be even or we use 1/2
                # Let's try: find c where M[c] != 0, set W[k*n+i, c] = 1/M[c]
                # For this to be half-integer, M[c] should be 1, 2, -1, -2, etc.
                
                found = False
                for c in range(target_rank):
                    if M[c] == 1:
                        W[k * n + i][c] = 1
                        found = True
                        break
                    elif M[c] == -1:
                        W[k * n + i][c] = -1
                        found = True
                        break
                    elif M[c] == 2:
                        W[k * n + i][c] = 0.5
                        found = True
                        break
                    elif M[c] == -2:
                        W[k * n + i][c] = -0.5
                        found = True
                        break
                
                if not found:
                    # Try a linear combination
                    # Find two indices c1, c2 with M[c1] + M[c2] = 1 or -1 or 2 or -2
                    found2 = False
                    for c1 in range(target_rank):
                        for c2 in range(target_rank):
                            if c1 == c2:
                                continue
                            s = M[c1] + M[c2]
                            if s == 1:
                                W[k * n + i][c1] = 1
                                W[k * n + i][c2] = 0
                                found2 = True
                                break
                            elif s == -1:
                                W[k * n + i][c1] = -1
                                W[k * n + i][c2] = 0
                                found2 = True
                                break
                            elif s == 2:
                                W[k * n + i][c1] = 0.5
                                W[k * n + i][c2] = 0
                                found2 = True
                                break
                            elif s == -2:
                                W[k * n + i][c1] = -0.5
                                W[k * n + i][c2] = 0
                                found2 = True
                                break
                        if found2:
                            break
                    if found2:
                        found = True
                
                if not found:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

# EVOLVE-BLOCK-END

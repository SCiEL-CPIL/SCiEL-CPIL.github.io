<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 3, 3, 6
    
    # Build the target tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix multiplication A(n,m) x B(m,p) -> C(n,p)
    # has entries T[i,j,k] = delta_{i,k} in the standard basis, but we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] for all i,j,k
    # Actually, the decomposition requires:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} (if we're doing C=A*B)
    # Wait, let me reconsider. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the "multiplication" sense)
    # No: T[i,j,k] = delta_{i,k} is wrong.
    # 
    # The correct formulation: We want to compute C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j,c} A[i,j] * B[j,k] * ... 
    # Actually, the tensor decomposition of matrix multiplication satisfies:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} * (something)
    # 
    # Let me think again. The matrix multiplication tensor is T in R^{n x m x p} where
    # T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Since every A[i,j]*B[j,k] contributes to C[i,k] (with i fixed and k fixed),
    # T[i,j,k] = 1 for all i,j,k.
    # 
    # No wait. The tensor T has entries T[i,j,k] which represents the coefficient
    # such that C[i,k] = sum_{j} sum_{c} ... 
    # 
    # Standard formulation: The multiplication tensor T_{n,m,p} has entries
    # T[i,j,k] = delta_{i,k} ... no.
    # 
    # Let me use the standard: We decompose the tensor T where
    # T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] always contributes to C[i,k])
    # But that's not right either because different (i,j,k) contribute to different C[i,k].
    #
    # Correct: The tensor T has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to C[i,k]. Since A[i,j]*B[j,k] contributes to C[i,k],
    # T[i,j,k] = 1 for all valid i,j,k.
    #
    # The rank-r decomposition means: there exist U (nm x r), V (mp x r), W (pn x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] * (contribution to W)
    # 
    # Actually the standard is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # No, that's not right.
    #
    # The correct tensor decomposition: T is a 3-tensor with T[i,j,k] and we want
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But T[i,j,k] for matrix mult is: T[i,j,k] = delta_{i,k}? No.
    #
    # Let me look at this differently. For 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}^3 (8 entries all 1).
    # The rank-7 Strassen decomposition: sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] = T[i,j,k]
    # where W[i,k,c] is indexed by output position.
    #
    # So for our case: T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,6)
    # We need: sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1 for all i,j,k
    #
    # This is a system of 3*3*6 = 54 equations.
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # First, let's return the schoolbook as fallback
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
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        if time.time() > deadline:
            return False
        rank = len(U[0])
        # Check all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_random_search(max_rank, num_trials=1000):
        """Try random search for a valid decomposition."""
        nonlocal best_result, best_rank
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            # Try random small integer/half-integer values
            scale = random.choice([1, 2, 3])
            U = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(n*m)]
            V = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(m*p)]
            W = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_result = {"U": U, "V": V, "W": W}
                return True
        return False
    
    # Try to find lower rank decompositions
    # Start from lower ranks and go up
    for rank in range(20, best_rank):
        if time.time() > deadline - 20.0:
            break
        if try_random_search(rank, num_trials=200):
            # Found a valid decomposition at this rank
            break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result
    
    # Otherwise return schoolbook
    return best_result

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 3, 3, 6
    
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
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        if time.time() > deadline:
            return False
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Build a rank-46 decomposition by applying Strassen (rank 7) to 3x3x3 blocks
    # and using schoolbook for the remaining 3 columns
    # For each pair of 3-column blocks of B, we can use the 3x3x3 structure
    # Actually, let's try a different approach: use the fact that 3x3x6 can be
    # decomposed as two 3x3x3 multiplications plus corrections
    
    # Method: Treat B as [B1 | B2] where B1, B2 are 3x3
    # C = A*B1 + A*B2
    # We can compute A*B1 using rank 23 (or 7 with Strassen) and A*B2 similarly
    # But the issue is that the U matrix would need to be shared or combined
    
    # Let's try: rank = 2 * 7 + 6 = 20? No, that's not how it works.
    # Actually for 3x3x6, we can use:
    # - 7 Strassen terms for the first 3 columns
    # - 7 Strassen terms for the next 3 columns  
    # But the U factors would be different...
    
    # Simpler: use rank 18 = 3*6 (factor through the middle dimension)
    # U[i*m+j, c] where c = j*p + k' ... no
    
    # Let me try a structured decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For rank r, we need U (9 x r), V (18 x r), W (18 x r)
    # such that sum_c U[i*3+j,c] * V[j*6+k,c] * W[k*3+i,c] = 1
    
    # A valid decomposition of rank 18:
    # c = j*3 + k where j in [0,3), k in [0,3) for first block, and j*3 + (k+3) for second
    # Actually simpler: c = j*6 + k for j in [0,3), k in [0,6) gives rank 18
    # U[i*3+j, j*6+k] = 1 if i==i (always), so U[i*3+j, j*6+k] = 1
    # V[j*6+k, j*6+k] = 1
    # W[k*3+i, j*6+k] = 1 if... no this doesn't work directly
    
    # Let me think differently. Rank 18 decomposition:
    # For each j in [0,3) and k in [0,6):
    #   U[i*3+j, j*6+k] = delta_{i,i} = 1 for all i? No...
    
    # Correct rank 18:
    # c = j*6 + k
    # U[i*3+j, j*6+k] = 1 for all i (this makes U have 1s in blocks)
    # V[j*6+k, j*6+k] = 1
    # W[k*3+i, j*6+k] = 1
    # Check: sum_c U[i*3+j',c] * V[j'*6+k',c] * W[k*3+i',c]
    # = sum_{j,k} U[i*3+j, j*6+k] * V[j*6+k, j*6+k] * W[k*3+i, j*6+k]
    # Wait, I need to be more careful.
    
    # Let's define:
    # U[i*3+j, j*6+k] = 1 for all i,j,k
    # V[j*6+k, j*6+k] = 1
    # W[k*3+i, j*6+k] = 1
    # Then sum_c U[i*3+j1,c] * V[j2*6+k2,c] * W[k3*3+i3,c]
    # c must equal j1*6+k1 = j2*6+k2 = ... 
    # This only gives nonzero when j1=j2 and k1=k2
    # And then U[i*3+j, j*6+k] * V[j*6+k, j*6+k] * W[k*3+i, j*6+k]
    # = 1 * 1 * 1 = 1 if i3=i and k3=k
    # But we need the sum to be 1 for specific (i,j,k) tuples...
    
    # I think I'm overcomplicating this. Let me just do a proper rank 18:
    # The identity decomposition has rank 54. We can do better.
    
    # Rank 18 approach:
    # For j in 0..2, k in 0..5:
    #   term c = j*6+k
    #   U[i*3+j, c] = 1 for all i  -> but this is wrong because U depends on i
    
    # Correct: U[i*3+j, j*6+k] should depend on i somehow
    # Actually U[i*3+j, c] is the coefficient for A[i,j] in term c
    # V[j*6+k, c] is the coefficient for B[j,k] in term c  
    # W[k*3+i, c] is the coefficient for C[i,k] in term c
    
    # For rank 18, let c = j*6 + k:
    # U[i*3+j, j*6+k] = 1 (coefficient of A[i,j])
    # V[j*6+k, j*6+k] = 1 (coefficient of B[j,k])
    # W[k*3+i, j*6+k] = 1 (coefficient of C[i,k])
    # 
    # Check: sum_c U[i*3+j1, c] * V[j2*6+k2, c] * W[k3*3+i3, c]
    # = sum_{j,k} U[i*3+j1, j*6+k] * V[j2*6+k2, j*6+k] * W[k3*3+i3, j*6+k]
    # U[i*3+j1, j*6+k] = 1 if j=j1, else 0
    # V[j2*6+k2, j*6+k] = 1 if j=j2 and k=k2, else 0
    # W[k3*3+i3, j*6+k] = 1 if k=k3 and i3=i3... wait W is indexed by [k*n+i, c]
    # W[k3*3+i3, j*6+k] = 1 if the output index k3*3+i3 matches... 
    # W[k*3+i, c] = 1 means this term contributes 1 to C[i,k]
    # So W[k3*3+i3, j*6+k] = 1 iff (k3,i3) = (k, i) for some i... 
    # This is getting confusing. Let me just verify computationally.
    
    def build_rank18():
        rank = 18
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(18)]
        W = [[0]*rank for _ in range(18)]
        for j in range(3):
            for k in range(6):
                c = j*6 + k
                for i in range(3):
                    U[i*3+j][c] = 1
                    W[k*3+i][c] = 1
                V[j*6+k][c] = 1
        return U, V, W
    
    U18, V18, W18 = build_rank18()
    if verify(U18, V18, W18):
        if 18 < best_rank:
            best_rank = 18
            best_result = {"U": U18, "V": V18, "W": W18}
    
    # Try rank 12: maybe we can do better
    # For 3x3x6, the theoretical lower bound is max(9, 18) = 18? No.
    # Lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # So rank 18 is the minimum possible!
    
    # But wait, that's just a trivial lower bound. The actual rank could be higher.
    # For 3x3x3, rank is 23 > max(9,9,9)=9
    # For 3x3x6, it could be more than 18.
    
    # Let's try to find if rank 18 works and if not, search for better
    if best_rank == 18:
        return best_result
    
    # If rank 18 didn't work, try random search for ranks 19-53
    def try_random_search(max_rank, num_trials=500):
        nonlocal best_result, best_rank
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            scale = random.choice([1, 2])
            U = [[random.choice([-scale, 0, 1]) for _ in range(max_rank)] for _ in range(n*m)]
            V = [[random.choice([-scale, 0, 1]) for _ in range(max_rank)] for _ in range(m*p)]
            W = [[random.choice([-scale, 0, 1]) for _ in range(max_rank)] for _ in range(p*n)]
            if verify(U, V, W):
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_result = {"U": U, "V": V, "W": W}
                return True
        return False
    
    for rank in range(19, best_rank):
        if time.time() > deadline - 20.0:
            break
        if try_random_search(rank, num_trials=100):
            break
    
    return best_result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

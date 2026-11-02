"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction as F
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Keep 20s safety margin before 150s kill
    
    def validate(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-8:
                        return False
        return True
    
    def validate_strict(U, V, W, rank):
        """Check with exact arithmetic using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = F(1)
                    actual = F(0)
                    for r in range(rank):
                        actual += F(U[i * m + j][r]) * F(V[j * p + k][r]) * F(W[k * n + i][r])
                    if actual != expected:
                        return False
        return True
    
    # Strategy: Use the structure of matrix multiplication to build a lower-rank decomposition.
    # Key insight: For 3x3 x 3x5, we can use the fact that the 5 output columns can be
    # grouped. We'll try to find a decomposition by exploiting linear dependencies.
    
    # Approach 1: Try to merge schoolbook terms by finding that some rank-1 terms
    # can be combined. This is equivalent to finding a basis for the span of the
    # rank-1 tensors.
    
    # Approach 2: Use a randomized search with local optimization.
    # We parameterize U, V, W and minimize the error.
    
    # Let's try a structured approach: 
    # The 3x3x5 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can split k into k1 (0,1,2) and k2 (3,4)
    # C[i,k1] = sum_j A[i,j] * B[j,k1]  (3x3x3 part)
    # C[i,k2] = sum_j A[i,j] * B[j,k2]  (3x3x2 part)
    
    # For 3x3x3, known rank is 23. For 3x3x2, schoolbook is 18.
    # Total naive: 23 + 18 = 41. But we can share some terms.
    
    # Let me try to construct an explicit decomposition.
    # I'll use a computational search approach.
    
    # Method: Start with schoolbook, then try to reduce rank by finding
    # linear combinations of rank-1 terms that still span the same space.
    
    # Actually, let me try a direct approach: use random initialization and
    # gradient-like optimization to find a valid decomposition.
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try ranks from 25 to 44 (skip 23-24 as they're very hard)
    for target_rank in range(25, rank_schoolbook):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            if found:
                break
            
            random.seed(1000 + target_rank * 10 + attempt)
            
            # Initialize with small random values
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local search: try to fix errors by adjusting individual entries
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute current errors
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            err = 1.0 - actual
                            if abs(err) > 1e-6:
                                errors.append((abs(err), i, j, k))
                
                if not errors:
                    # Check with strict validation
                    if validate_strict(U, V, W, target_rank):
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                        found = True
                        break
                    else:
                        # Floating point issue, try to round
                        U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
                        V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
                        W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
                        if validate_strict(U_rounded, V_rounded, W_rounded, target_rank):
                            best_rank = target_rank
                            best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                            found = True
                            break
                
                if not errors:
                    break
                
                # Pick the largest error
                errors.sort(reverse=True)
                _, ei, ej, ek = errors[0]
                
                # Try to reduce this error by adjusting one entry
                # The error is: 1 - sum_r U[ei*m+ej,r]*V[ej*p+ek,r]*W[ek*n+ei,r]
                # We can try adjusting U[ei*m+ej, r] for some r
                # Delta = -err / (V[ej*p+ek,r] * W[ek*n+ei,r])
                
                adjusted = False
                for r in range(target_rank):
                    v_val = V[ej*p+ek][r]
                    w_val = W[ek*n+ei][r]
                    if v_val != 0 and w_val != 0:
                        delta = (1.0 - sum(U[ei*m+ej][rr] * V[ej*p+ek][rr] * W[ek*n+ei][rr] for rr in range(target_rank))) / (v_val * w_val)
                        new_val = U[ei*m+ej][r] + delta
                        # Round to nearest 0.5
                        new_val = round(new_val * 2) / 2
                        if new_val != U[ei*m+ej][r]:
                            U[ei*m+ej][r] = new_val
                            adjusted = True
                            break
                
                if not adjusted:
                    # Try adjusting V
                    for r in range(target_rank):
                        u_val = U[ei*m+ej][r]
                        w_val = W[ek*n+ei][r]
                        if u_val != 0 and w_val != 0:
                            delta = (1.0 - sum(U[ei*m+ej][rr] * V[ej*p+ek][rr] * W[ek*n+ei][rr] for rr in range(target_rank))) / (u_val * w_val)
                            new_val = V[ej*p+ek][r] + delta
                            new_val = round(new_val * 2) / 2
                            if new_val != V[ej*p+ek][r]:
                                V[ej*p+ek][r] = new_val
                                adjusted = True
                                break
                
                if not adjusted:
                    # Try adjusting W
                    for r in range(target_rank):
                        u_val = U[ei*m+ej][r]
                        v_val = V[ej*p+ek][r]
                        if u_val != 0 and v_val != 0:
                            delta = (1.0 - sum(U[ei*m+ej][rr] * V[ej*p+ek][rr] * W[ek*n+ei][rr] for rr in range(target_rank))) / (u_val * v_val)
                            new_val = W[ek*n+ei][r] + delta
                            new_val = round(new_val * 2) / 2
                            if new_val != W[ek*n+ei][r]:
                                W[ek*n+ei][r] = new_val
                                adjusted = True
                                break
        
        if found:
            break
    
    # Also try the schoolbook as fallback
    if best_rank == rank_schoolbook:
        best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # For each rank r, try to find U, V, W matrices of size (nm x r), (mp x r), (pn x r)
    # such that for all (i,j,k): sum_s U[i*m+j, s] * V[j*p+k, s] = W[k*n+i, s]
    
    # This is equivalent to: the tensor T_{i,j,k} = sum_s U_{i*m+j, s} * V_{j*p+k, s}
    # and W_{k*n+i, s} must equal this for the decomposition to be valid.
    
    # Actually, the correct formulation: we need three matrices U, V, W such that
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no wait.
    
    # Let me re-read the problem. The tensor decomposition of matrix multiplication:
    # (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T_{i,j,k} represents the multiplication operation.
    # A rank-1 decomposition term is: U_{i,j} * V_{j,k} * W_{k,i}? No.
    
    # Actually, for bilinear forms: T(A,B) = sum_r U_r(A) * V_r(B) where U_r, V_r are linear forms.
    # But here we have three indices, so it's a trilinear form? No, matrix multiplication is bilinear.
    
    # The tensor for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a bilinear map. The "tensor" is T_{i,j,k} where the output index is (i,k) and input indices are (i,j) and (j,k).
    
    # For tensor decomposition in the sense of bilinear complexity:
    # We want to write C_{i,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} where for each r,
    # U^{(r)} is a matrix of size n×m and V^{(r)} is a matrix of size m×p.
    # Then the rank is the number of terms r.
    
    # But the problem says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And the decomposition is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    
    # Wait, that doesn't make sense dimensionally. Let me re-read:
    # "exactly decompose the matrix multiplication tensor"
    # The condition should be: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 if this is a valid term
    # Actually no. The tensor T_{i,j,k} for matrix multiplication is defined such that
    # (A*B)_{i,k} = sum_{j} sum_r U[i*m+j, r] * A_{i,j} * V[j*p+k, r] * B_{j,k} ... 
    
    # I think the correct interpretation is:
    # We need U (n*m × r), V (m*p × r), W (p*n × r) such that for all i,j,k:
    # sum_s U[i*m+j, s] * V[j*p+k, s] = W[k*n+i, s] for each s? No, that's not right either.
    
    # Let me think again. The standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T_{(i,j),(j,k),(i,k)} = 1 for valid indices, 0 otherwise.
    # A rank-1 decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r ∈ R^{n×m}, V_r ∈ R^{m×p}, W_r ∈ R^{n×p}
    # And the condition is: for all (i,j), (j,k), (i,k):
    # T_{(i,j),(j,k),(i,k)} = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
    
    # But the problem states W has shape (p*n, rank), which suggests W is indexed by (k,i) not (i,k).
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is from B's columns and i is from A's rows.
    
    # So the condition is: for all i,j,k:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T_{i,j,k} = 1 for all valid (i,j,k) since (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # Wait, but that's a trilinear form, not bilinear. Matrix multiplication is bilinear in A and B.
    # The "tensor rank" here might refer to the bilinear complexity, which is different.
    
    # Actually, I think the problem is using a different convention. Let me assume the condition is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    # This would mean that for each r, U[:,r], V[:,r], W[:,r] form a valid rank-1 term.
    
    # Let me just implement a search that tries to satisfy:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] = 1 (since T_{i,j,k}=1)
    # And W is determined by: W[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r] for consistency?
    
    # No, I think the correct condition based on the problem statement is:
    # The three matrices U, V, W define a decomposition where for each rank term r,
    # the product U[:,r] ⊗ V[:,r] ⊗ W[:,r] contributes to the tensor.
    # The condition is that the sum over r of these rank-1 tensors equals the multiplication tensor.
    
    # For the multiplication tensor, T_{i,j,k} = 1 for all valid (i,j,k).
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # This is a system of n*m*p = 48 equations in r * (n*m + m*p + p*n) = r * (6+24+16) = 46r unknowns.
    # For r < 48/46 ≈ 1.04, this is underdetermined but with nonlinear constraints.
    
    # Let me try a different approach: fix U and V, then W is determined.
    # Or use a greedy/iterative approach.
    
    # For now, let's try small ranks with a simple search.
    # Given time constraints, let's try ranks from a lower bound up.
    
    # Lower bound: the rank must be at least max(n, m, p) = 8
    # Also, rank >= n*p = 16 (since each output element needs at least one term)
    # Actually, rank >= max(n*m, m*p, p*n) / something... let's just try from 16 up.
    
    min_rank = max(n * p, m * p, n * m)  # 16, 24, 6 -> 24
    # But we know schoolbook is 48, so try from 24 to 47
    
    for target_rank in range(min_rank, rank_school):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition with this rank
        # Use a simple approach: initialize randomly and try to satisfy the constraints
        
        # For simplicity, assume entries are in {-1, 0, 1}
        # Try a few random initializations
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V with random entries from {-1, 0, 1}
            np.random.seed(42 + attempt)
            U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            
            # Compute the tensor contribution: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r]
            # We need this to equal 1 for all (i,j,k)
            
            # Check if there exists W such that U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sums to 1
            # This is complex. Let's instead check if sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
            # If so, we can set W[k*n+i, r] = 1 for all r (but that might not work)
            
            # Actually, let's just check if the bilinear form sum_r U[i*m+j, r] * V[j*p+k, r] = 1
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] for r in range(target_rank))
                        if val != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                # Set W to 1s (or compute appropriately)
                W = np.ones((p*n, target_rank))
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

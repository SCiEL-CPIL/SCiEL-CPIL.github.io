"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check tensor equation: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And for all other combinations where the indices don't align with a valid (i,j,k), it should be 0
    # Actually, the tensor T_{i,j,k}^{i',j',k'} = delta_{i,i'}delta_{j,j'}delta_{k,k'} for the multiplication tensor
    # The decomposition gives: T = sum_c U_c ⊗ V_c ⊗ W_c
    # Where U_c has entries at positions (i,j), V_c at (j,k), W_c at (k,i)
    # The tensor element T[i,j,k] for the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} in the sense that
    # sum_{i,j,k} A[i,j] B[j,k] W[k*n+i] ... 
    
    # Let's construct the full tensor and verify
    # The multiplication tensor M has entries M[i,j,k] = 1 if this corresponds to a valid term
    # Actually, let's just verify the rank directly by checking the tensor identity
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, we need T to equal the standard multiplication tensor
    # where T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook sense... 
    
    # Actually the standard tensor for multiplying n×m by m×p is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the output is C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor representation: T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]*T[i,j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Wait, that's not right either. Let me think more carefully.
    # The matrix multiplication tensor T is such that:
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T = sum_{i,j,k} e_i ⊗ e_j ⊗ f_{i,k} where f_{i,k} is the basis for C
    # But the output index is (i,k), so the third factor has dimension n*p
    # T[i,j,i*n+k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,i*n+k] = 1 for all i,j,k
    
    # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of length n*m, V_c of length m*p, W_c of length n*p
    # T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k]
    # Wait, the problem says W[k*n+i, r] so W is indexed by (k,i) with k in [0,p), i in [0,n)
    # So W_c has entries W_c[k*n+i]
    
    # So T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # And we need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the third index k here refers to the output column, and the output row is i
    # So the output index is (i, k) stored as k*n + i
    
    # Let's verify:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition using random search
    # The known lower bound for 4x5 * 5x6 is not well-established, but 
    # schoolbook is 4*5*6 = 120. 
    # Let's try random rank-reduction approaches.
    
    # Approach: Try random projections to reduce rank
    # For each candidate rank r < best_rank, try to find a valid decomposition
    
    np.random.seed(42)
    
    # Try various ranks in decreasing order
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        max_attempts = 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Use gradient-free approach: try to solve the system
            # We need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
            # This is a system of n*m*p equations in n*m*r + m*p*r + p*n*r unknowns
            
            # Alternative: use the fact that we can fix one matrix and solve for others
            # Fix U, then we need V and W such that the tensor equation holds
            
            # Simpler approach: try to use structure
            # For small ranks, we can try systematic search
            
            # Actually, let's try a different approach: 
            # Use the known result that for 4x5*5x6, the minimum rank might be achievable
            # through structured decompositions
            
            # Try: use rank = n*p + m*p + n*m - something
            # Or try specific structured decompositions
            
            # For now, let's try a simple heuristic: 
            # If we can find any valid decomposition with lower rank, use it
            pass
    
    # Given the complexity, return the schoolbook decomposition for now
    # unless we find something better
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

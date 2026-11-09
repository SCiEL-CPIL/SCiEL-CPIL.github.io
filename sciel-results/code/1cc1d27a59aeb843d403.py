"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The tensor is T[i,j,k] = delta_{j,k} for matrix multiplication
    # We need U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] summed over r
    
    # Known lower bounds: for 3x5x5, rank >= max(n*m, m*p, p*n) = 25
    # But actual lower bound is higher. Let's try ranks from 25 upward.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] should equal delta_{i,i'} * delta_{k,k'}
        # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
        # Wait, let me reconsider.
        
        # The matrix multiplication tensor M has entries M[i,j,k] where:
        # - i indexes rows of A (0..n-1)
        # - j indexes columns of A / rows of B (0..m-1)  
        # - k indexes columns of B (0..p-1)
        # M[i,j,k] = 1 for all valid i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k])
        
        # The decomposition says: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right either. Let me re-read the contract.
        
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) and W[k*n+i, r]
        
        # The standard tensor decomposition for matrix multiplication:
        # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where M[i,j,k] = 1 for all i,j,k (the "all ones" tensor for this structure)
        
        # Actually no. The matrix multiplication tensor is:
        # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to C[i,k]
        # Since every (i,j,k) triple has a unique contribution, T[i,j,k] = 1 for all valid indices.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Randomized search for lower rank decompositions
    # Strategy: try small ranks and use random initialization with local search
    
    # First, let's try a structured approach
    # For matrix multiplication, there are known decompositions using bilinear methods
    
    # Try ranks from a lower bound upward
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But also rank >= (n*m*p) / min(n,m,p) = 75/3 = 25
    
    # Let's try a few candidate ranks
    candidate_ranks = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline - 3.0:
                break
            
            # Random initialization with small integers
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            # Initialize with random values from {-1, 0, 1}
            for r in range(target_rank):
                # Choose which entries are non-zero
                u_nonzero = random.sample(range(n*m), min(target_rank, n*m))
                v_nonzero = random.sample(range(m*p), min(target_rank, m*p))
                w_nonzero = random.sample(range(p*n), min(target_rank, p*n))
                
                for idx in u_nonzero:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_nonzero:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_nonzero:
                    W[idx, r] = random.choice([-1, 1])
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, target_rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                print(f"Found rank {target_rank} decomposition!")
                break
        
        if best_rank < rank_school:
            break
    
    return best_result
# EVOLVE-BLOCK-END

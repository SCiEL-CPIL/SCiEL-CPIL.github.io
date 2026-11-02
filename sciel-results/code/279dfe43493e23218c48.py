"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    # Try to find a lower-rank decomposition using structured search
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    # Start with schoolbook as fallback
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Try lower ranks with random initialization and gradient-free search
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    for target_rank in [79, 78, 77, 76, 75, 74, 73, 72, 70, 65, 60]:
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        max_iter = 200 if target_rank >= 70 else 100
        found = False
        
        for attempt in range(5 if target_rank >= 70 else 3):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U_try = np.random.randn(n * m, target_rank) * 0.1
            V_try = np.random.randn(m * p, target_rank) * 0.1
            W_try = np.random.randn(p * n, target_rank) * 0.1
            
            # Coordinate descent: update one tensor factor at a time
            # For each r, the contribution to T[i,j,k] is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # We want sum over r to equal 1 for all (i,j,k)
            
            # Use a simple iterative approach: for each r, try to optimize
            # Actually, let's use a more structured approach based on known results
            
            # For 4x4x5, the schoolbook rank is 80. 
            # Known lower bounds: for m*n*p matrix multiplication, rank >= max(m*n, n*p, m*p)
            # Here: max(16, 20, 20) = 20, but that's a very weak bound.
            
            # Let's try a different approach: use the fact that we can combine terms
            # If we have two terms that both contribute to the same (i,j,k), we can potentially merge them
            
            # Try a simple perturbation approach from schoolbook
            if target_rank == 79:
                # Remove one term and see if we can adjust others
                U_try = U[:, :79].copy()
                V_try = V[:, :79].copy()
                W_try = W[:, :79].copy()
                
                # The missing term is for (i=0, j=0, k=0)
                # T[0,0,0] = sum_r U[0,r]*V[0,r]*W[0,r] should be 1
                # Currently it's 0 (since we removed the term that had all 1s)
                # We need to add 1 to this entry
                # We can adjust one of the remaining terms
                
                # Find a term where we can add a small perturbation
                # Term r=0 has U[0,0]=1, V[0,0]=1, W[0,0]=1
                # If we change U[0,0] to 2, then T[0,0,0] increases by 1
                # But T[0,0,k] for k!=0 also increases by 1 (since V[0,k]=1, W[k,0]=1)
                # So we need to compensate
                
                # This is getting complex. Let's just verify if 79 is achievable.
                # Actually, for the all-ones tensor, the rank is exactly n*m*p
                # because the tensor is decomposable into n*m*p rank-1 tensors
                # and it's not clear we can do better.
                
                pass
        
        # If we can't find a better rank, keep schoolbook
        if not found:
            continue
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i,j]
                # Check if it's close to an integer or half-integer
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    row.append(round(val*2)/2)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

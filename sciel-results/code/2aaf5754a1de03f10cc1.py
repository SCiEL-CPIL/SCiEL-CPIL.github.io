"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    schoolbook_rank = n * m * p  # 96

    # Build schoolbook solution
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    # Verify the schoolbook decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
        
        # T should be all ones
        return np.allclose(T_recon, 1.0, atol=tol)

    # Try to find a lower rank decomposition using the Kronecker product structure
    # m=4 = 2*2, p=8 = 2*4, n=3
    # We can try to exploit the tensor product structure

    # Strategy: Use Strassen's algorithm for 2x2 multiplication and combine with
    # the remaining dimensions
    
    # For a 2x2 by 2x2 multiplication, Strassen gives rank 7
    # We can decompose the 4x8 multiplication as follows:
    # View the 4x4 part as two 2x2 blocks, and the 8 columns as four 2-column blocks
    
    # Actually, let's try a different approach:
    # Decompose m=4 as 2*2 and p=8 as 2*4
    # Then we have a 3x2 by 2x4 and a 3x2 by 2x4 multiplication
    # Each can be done with rank 3*2*4 = 24, but we can do better
    
    # Let's try to use the fact that we can share computation between the two 2x2 blocks
    
    # Alternative approach: Try to find a rank-88 decomposition by removing 8 terms
    # and solving for adjustments
    
    # For now, let's try a greedy approach to reduce rank
    
    def try_greedy_reduction(U_arr, V_arr, W_arr, current_rank, max_iterations=50):
        """Try to reduce rank by greedily removing terms and adjusting others."""
        U_curr = U_arr.copy()
        V_curr = V_arr.copy()
        W_curr = W_arr.copy()
        r = current_rank
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            
            best_improvement = 0
            best_remove_idx = -1
            best_new_U = None
            best_new_V = None
            best_new_W = None
            
            # Try removing each term and see if we can compensate
            for c_remove in range(r):
                if time.time() > deadline - 10.0:
                    break
                
                # Create mask to remove term c_remove
                mask = [c for c in range(r) if c != c_remove]
                if not mask:
                    break
                
                U_reduced = U_curr[:, mask]
                V_reduced = V_curr[:, mask]
                W_reduced = W_curr[:, mask]
                
                # Check if the remaining terms can still reconstruct the tensor
                # This is a linear feasibility problem
                # For simplicity, just check if the rank is still sufficient
                
                # Compute the tensor from reduced terms
                T_recon = np.zeros((n, m, p))
                for c in range(len(mask)):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon[i, j, k] += U_reduced[i*m+j, c] * V_reduced[j*p+k, c] * W_reduced[k*n+i, c]
                
                # Check if we're close to all ones
                error = np.abs(T_recon - 1.0).max()
                if error < 1e-10:
                    # We found a valid reduction
                    best_improvement = 1
                    best_remove_idx = c_remove
                    best_new_U = U_reduced
                    best_new_V = V_reduced
                    best_new_W = W_reduced
                    break
            
            if best_improvement > 0:
                U_curr = best_new_U
                V_curr = best_new_V
                W_curr = best_new_W
                r -= 1
            else:
                break
        
        return U_curr, V_curr, W_curr, r

    # Try greedy reduction
    if time.time() < deadline - 30.0:
        U_reduced, V_reduced, W_reduced, new_rank = try_greedy_reduction(
            U_np, V_np, W_np, schoolbook_rank, max_iterations=20
        )
        if new_rank < best_rank:
            # Convert back to list format
            best_result = {
                "U": U_reduced.tolist(),
                "V": V_reduced.tolist(),
                "W": W_reduced.tolist()
            }
            best_rank = new_rank

    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END

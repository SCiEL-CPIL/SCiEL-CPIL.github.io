import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    rank = 54
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base, col] += 1
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[j_base * p + k_base, col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base, col] += 1
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        V[(j_base+1) * p + (k_base+1), col] -= 1
        W[(k_base+1) * n + i_base, col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1), col] += 1
        V[(j_base+1) * p + k_base, col] += 1
        V[j_base * p + k_base, col] -= 1
        W[k_base * n + i_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base, col] += 1
        U[i_base * m + (j_base+1), col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] -= 1
        W[(k_base+1) * n + i_base, col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base, col] += 1
        U[i_base * m + j_base, col] -= 1
        V[j_base * p + k_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1), col] += 1
        U[(i_base+1) * m + (j_base+1), col] -= 1
        V[(j_base+1) * p + k_base, col] += 1
        V[(j_base+1) * p + (k_base+1), col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # 2x1 * 1x2 -> 4 scalar mults
        U[i_base * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1
        
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + i_base, col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base+1), col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base, col] += 1
        V[j_base * p + (k_base+1), col] += 1
        W[(k_base+1) * n + (i_base+1), col] += 1
        col += 1

    # Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1),(2,3),(4,5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Naive blocks for j=4
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify initial decomposition
    def verify(U_arr, V_arr, W_arr, current_rank):
        """Check if decomposition is valid for all (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :current_rank], 
                                 V_arr[j*p+k, :current_rank] * W_arr[k*n+i, :current_rank])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Should not happen, but return as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Greedy elimination with numpy verification
    current_rank = rank
    
    # Precompute for speed: for each (i,j,k), store the index triple
    # and use vectorized dot products
    
    max_passes = 8
    for pass_num in range(max_passes):
        if time.time() > deadline - 20:
            break
        
        removed_in_pass = False
        for trial in range(current_rank):
            if time.time() > deadline - 15:
                break
            
            # Try removing term 'trial'
            # Zero out column 'trial'
            U[:, trial] = 0
            V[:, trial] = 0
            W[:, trial] = 0
            
            if verify(U, V, W, current_rank):
                # Term removed successfully, shift columns
                current_rank -= 1
                # Remove the column by shifting
                U[:, trial:] = U[:, trial+1:current_rank+1]
                V[:, trial:] = V[:, trial+1:current_rank+1]
                W[:, trial:] = W[:, trial+1:current_rank+1]
                removed_in_pass = True
                break  # Restart with new rank
        
        if not removed_in_pass:
            break
    
    # Convert to lists
    final_U = U[:, :current_rank].tolist()
    final_V = V[:, :current_rank].tolist()
    final_W = W[:, :current_rank].tolist()
    
    return {"U": final_U, "V": final_V, "W": final_W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Schoolbook as fallback: rank = n*m*p = 48
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    def verify(U_mat, V_mat, W_mat):
        """Check if decomposition exactly reproduces matrix multiplication."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def verify_fast(U_mat, V_mat, W_mat):
        """Faster verification using numpy einsum."""
        n, m, p = DIMENSIONS
        # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
        U_r = U_mat.reshape(n, m, U_mat.shape[1])
        V_r = V_mat.reshape(m, p, V_mat.shape[1])
        W_r = W_mat.reshape(p, n, W_mat.shape[1])
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check if all entries are 1
        return np.allclose(T, 1.0, atol=1e-9)

    # Strategy: Use known algebraic identities to reduce rank
    # For 3x4x4, we can try to use the structure of 4x4 matmul
    # 4x4 matmul can be done in rank 25 (Lavin-Shokrollahi) or even lower with careful construction
    # But for 3x4x4, we need to multiply a 3x4 by 4x4, which is different
    
    # Let's try a systematic search with structured patterns
    # Key insight: We can use the fact that some entries can be shared
    
    # Try rank values from 44 upward
    for target_rank in [44, 45, 46, 47]:
        if time.time() > deadline:
            break
        
        # Try different structured approaches
        # Approach A: Random sparse with small integer/half-integer entries
        entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                # Sparse pattern with 2-4 nonzeros per column
                u_nonzero = np.random.choice(n*m, size=np.random.randint(2,5), replace=False)
                v_nonzero = np.random.choice(m*p, size=np.random.randint(2,5), replace=False)
                w_nonzero = np.random.choice(p*n, size=np.random.randint(2,5), replace=False)
                
                for idx in u_nonzero:
                    U_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in v_nonzero:
                    V_mat[idx, c] = entries[np.random.randint(len(entries))]
                for idx in w_nonzero:
                    W_mat[idx, c] = entries[np.random.randint(len(entries))]
            
            # Quick check with numpy
            if verify_fast(U_mat, V_mat, W_mat):
                best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                best_rank = target_rank
                break
        
        if best_rank < rank_school:
            break
    
    # If we found a better rank, try to optimize further
    if best_rank < rank_school and time.time() < deadline:
        current_rank = best_rank
        while current_rank > 40 and time.time() < deadline:
            new_rank = current_rank - 1
            found = False
            attempts = 0
            entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
            
            while attempts < 2000 and not found and time.time() < deadline:
                attempts += 1
                U_mat = np.zeros((n*m, new_rank))
                V_mat = np.zeros((m*p, new_rank))
                W_mat = np.zeros((p*n, new_rank))
                
                for c in range(new_rank):
                    u_nonzero = np.random.choice(n*m, size=np.random.randint(2,5), replace=False)
                    v_nonzero = np.random.choice(m*p, size=np.random.randint(2,5), replace=False)
                    w_nonzero = np.random.choice(p*n, size=np.random.randint(2,5), replace=False)
                    
                    for idx in u_nonzero:
                        U_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in v_nonzero:
                        V_mat[idx, c] = entries[np.random.randint(len(entries))]
                    for idx in w_nonzero:
                        W_mat[idx, c] = entries[np.random.randint(len(entries))]
                
                if verify_fast(U_mat, V_mat, W_mat):
                    best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                    best_rank = new_rank
                    current_rank = new_rank
                    found = True
            
            if not found:
                break
    
    return best_result
# EVOLVE-BLOCK-END

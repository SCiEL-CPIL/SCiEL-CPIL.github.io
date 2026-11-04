"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Strategy: Use block decomposition to exploit structure
    # A is 4x5, B is 5x6, C = A*B is 4x6
    # Split A into [A1 (4x4) | A2 (4x1)]
    # Split B into [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4x4 multiplication
    # A2*B2: 4x1x6 multiplication (outer product, rank 4*6=24)
    #
    # For 4x4x4: use Strassen's algorithm (rank 49) or schoolbook (rank 64)
    # Let's try to use a known rank-49 decomposition for 4x4x4
    # Actually, let's start with a simpler approach:
    # 
    # For 4x4x4, the schoolbook rank is 64.
    # Strassen gives rank 49 for 4x4x4.
    # 
    # Let me implement Strassen for 4x4x4.
    # Actually, let me just use schoolbook for both blocks first:
    # A1*B1: 4*4*4 = 64
    # A2*B2: 4*1*6 = 24
    # Total: 88
    #
    # That's already better than 120!
    #
    # Let me verify the tensor decomposition convention:
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (30, rank)  
    # W has shape (p*n, rank) = (24, rank)
    #
    # The decomposition satisfies:
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Wait, that's not right. Let me re-read the convention.
    #
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of C, i is row of C
    # U[i*m+j, r] - U is indexed by (i, j) where i is row of A, j is column of A
    # V[j*p+k, r] - V is indexed by (j, k) where j is row of B, k is column of B
    #
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, this doesn't look like a standard tensor decomposition. Let me think again.
    #
    # Actually, the standard matrix multiplication tensor T is defined by:
    # T_{i,j,k} = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    # 
    # A rank-1 decomposition would be:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)_{i,j,k} = U_r[i] * V_r[j] * W_r[k]
    #
    # But here we have a different indexing. Let me re-interpret:
    #
    # The evaluator checks: for each (i,j,k), the contribution should be 1 if
    # A[i,j]*B[j,k] contributes to C[i,k].
    #
    # Looking at the schoolbook in the original code:
    # U_school[i*m+j, col] = 1.0
    # V_school[j*p+k, col] = 1.0
    # W_school[k*n+i, col] = 1.0
    #
    # So for each (i,j,k) triple, we have one rank-1 term where:
    # U has 1 at position (i*m+j), V has 1 at position (j*p+k), W has 1 at position (k*n+i)
    #
    # The check is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for each (i,j,k) that corresponds to a valid multiplication term.
    #
    # Wait, but that would mean for each (i,j,k), we need the sum over r to be 1.
    # But in schoolbook, each (i,j,k) has exactly one r where all three are 1.
    #
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is equivalent to saying the tensor T with entries T[i*m+j, j*p+k, k*n+i] = 1
    # has a rank-r decomposition.
    #
    # OK so my block decomposition idea:
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    #
    # For the first sum (j=0..3): this is a 4x4x4 multiplication
    # For the second term: A[i,4]*B[4,k] for i in 0..3, k in 0..5
    #
    # For the 4x4x4 part:
    # We need: sum_r U1[i*5+j, r] * V1[j*6+k, r] * W1[k*4+i, r] = 1 for i in 0..3, j in 0..3, k in 0..5
    # Wait, but j only goes up to 3, and k goes up to 5.
    #
    # Hmm, this is getting complicated. Let me just implement the straightforward
    # block decomposition with schoolbook for each block.
    #
    # Block 1: A1 (4x4) * B1 (4x6) -> contributes to C (4x6)
    #   For i in 0..3, j in 0..3, k in 0..5:
    #     U[i*5+j, r] = 1, V[j*6+k, r] = 1, W[k*4+i, r] = 1
    #   Number of terms: 4*4*6 = 96
    #
    # Block 2: A2 (4x1) * B2 (1x6) -> contributes to C (4x6)
    #   For i in 0..3, j=4, k in 0..5:
    #     U[i*5+4, r] = 1, V[4*6+k, r] = 1, W[k*4+i, r] = 1
    #   Number of terms: 4*1*6 = 24
    #
    # Total: 96 + 24 = 120. That's the same as schoolbook!
    #
    # Oh right, because the schoolbook already does all 4*5*6 = 120 terms.
    #
    # To get a lower rank, I need to actually reduce the rank of one of the blocks.
    #
    # For the 4x4x6 part (A1*B1), can I do better than 4*4*6=96?
    # The rank of the 4x4x6 multiplication tensor... 
    #
    # Actually, I think the key insight is that for fixed i,k, we have:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # This is a bilinear form. The tensor rank is the minimum number of rank-1
    # bilinear forms needed.
    #
    # For 4x5x6, I don't know the exact minimal rank, but it's definitely less than 120.
    #
    # Let me try a different approach: use known fast matrix multiplication algorithms.
    #
    # Actually, let me just try to use a random search with gradient descent more effectively.
    # The previous code had a good framework but maybe the learning rate schedule was off.
    
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def smooth_loss(U, V, W, n, m, p):
        rank = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    total_err += (val - 1.0) ** 2
        return total_err
    
    def compute_gradients(U, V, W, n, m, p):
        rank = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = 2.0 * (val - 1.0)
                    dU[i*m+j, :] += err * v_row * w_row
                    dV[j*p+k, :] += err * u_row * w_row
                    dW[k*n+i, :] += err * u_row * v_row
        
        return dU, dV, dW
    
    # Try various target ranks with better optimization
    target_ranks = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        n_restarts = 3
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(42 + restart * 7919 + target_rank)
            
            # Better initialization: start near schoolbook and project
            # Initialize with random values scaled appropriately
            scale = 1.0 / np.sqrt(target_rank)
            U_try = np.random.randn(n*m, target_rank) * scale
            V_try = np.random.randn(m*p, target_rank) * scale
            W_try = np.random.randn(p*n, target_rank) * scale
            
            # Use Adam-like optimizer
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            mU = np.zeros_like(U_try)
            vU = np.zeros_like(U_try)
            mV = np.zeros_like(V_try)
            vV = np.zeros_like(V_try)
            mW = np.zeros_like(W_try)
            vW = np.zeros_like(W_try)
            
            best_loss = smooth_loss(U_try, V_try, W_try, n, m, p)
            patience = 0
            max_patience = 100
            t = 0
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                    
                loss = smooth_loss(U_try, V_try, W_try, n, m, p)
                
                if loss < 1e-10:
                    if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-8):
                        best_U = U_try.copy()
                        best_V = V_try.copy()
                        best_W = W_try.copy()
                        best_rank = target_rank
                        found = True
                        break
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                    if patience > max_patience:
                        break
                
                dU, dV, dW = compute_gradients(U_try, V_try, W_try, n, m, p)
                
                t += 1
                mU = beta1 * mU + (1 - beta1) * dU
                vU = beta2 * vU + (1 - beta2) * dU**2
                mhatU = mU / (1 - beta1**t)
                vhatU = vU / (1 - beta2**t)
                
                mV = beta1 * mV + (1 - beta1) * dV
                vV = beta2 * vV + (1 - beta2) * dV**2
                mhatV = mV / (1 - beta1**t)
                vhatV = vV / (1 - beta2**t)
                
                mW = beta1 * mW + (1 - beta1) * dW
                vW = beta2 * vW + (1 - beta2) * dW**2
                mhatW = mW / (1 - beta1**t)
                vhatW = vW / (1 - beta2**t)
                
                U_try -= lr * mhatU / (np.sqrt(vhatU) + eps)
                V_try -= lr * mhatV / (np.sqrt(vhatV) + eps)
                W_try -= lr * mhatW / (np.sqrt(vhatW) + eps)
                
                # Decay learning rate
                lr = 0.01 * (0.999 ** step)
            
            if found:
                break
    
    # Convert to list of lists
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(best_U))
    V_out = clean_list(to_list(best_V))
    W_out = clean_list(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END


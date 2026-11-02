"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
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
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Block Strassen - 12 independent 2x2x2 Strassen products
    # A: 4x4 -> 2x2 blocks indexed by (a,c) where a=row_block, c=col_block
    # B: 4x6 -> 2x2 blocks indexed by (c,b) where c=row_block, b=col_block
    # C: 4x6 -> 2x2 blocks indexed by (a,b) where a=row_block, b=col_block
    # C[a,b] = sum_c A[a,c] * B[c,b]
    # Total: 2*2*3 = 12 products, each rank 7 -> rank 84
    
    # Strassen 2x2x2 decomposition terms
    # Each term: (A_coefs, B_coefs, C_coefs)
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4 = A11(B01-B00)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 x 84
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 x 84
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(3):  # col block of B/C
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side: A_block(a,c) = A[2a:2a+2, 2c:2c+2]
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar  # global row in A
                        gj = 2*c + ac  # global col in A
                        U1[gi * m + gj][col] += coef
                    
                    # B-side: B_block(c,b) = B[2c:2c+2, 2b:2b+2]
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br  # global row in B
                        gj = 2*b + bc  # global col in B
                        V1[gi * p + gj][col] += coef
                    
                    # C-side: C_block(a,b) = C[2a:2a+2, 2b:2b+2]
                    # W[k*n+i] where k is row of C (=col of B), i is col of C (=row of A)
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc  # k (row of C)
                        gj = 2*a + cr  # i (col of C)
                        W1[gi * n + gj][col] += coef
                    
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a better decomposition using numerical optimization
    # We'll try to find a rank-60 decomposition (better than 84)
    
    # The idea: use a randomized initialization and gradient descent to find
    # a low-rank decomposition of the matrix multiplication tensor.
    
    # The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,6)
    # We want: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    
    # Let's try rank 60, 55, 50, 45, 40, 35, 30
    # Start with a reasonable rank and try to reduce it
    
    # Use numpy for efficient computation
    def try_rank(rank, max_iters=2000, lr=0.01, seed=42):
        """Try to find a rank-`rank` decomposition using gradient descent."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U_np = np.random.randn(n * m, rank) * 0.1
        V_np = np.random.randn(m * p, rank) * 0.1
        W_np = np.random.randn(p * n, rank) * 0.1
        
        # The target tensor is all ones
        # T[i,j,k] = 1 for all i,j,k
        
        # We want to minimize: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        
        # This is a non-convex problem, but we can use gradient descent
        
        # Precompute some indices
        # For each (i,j,k), we need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # Let's use a simple gradient descent with momentum
        
        U_grad = np.zeros_like(U_np)
        V_grad = np.zeros_like(V_np)
        W_grad = np.zeros_like(W_np)
        
        vU = np.zeros_like(U_np)
        vV = np.zeros_like(V_np)
        vW = np.zeros_like(W_np)
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute forward pass
            # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is expensive: 4*4*6 = 96 entries, each a dot product of length `rank`
            # Total: 96 * rank multiplications per iteration
            
            # Let's vectorize using numpy
            # For each r, we have U[:, r] (16,), V[:, r] (24,), W[:, r] (24,)
            # The contribution to (i,j,k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let's build the tensor product efficiently
            # For each r, the tensor is outer(U[:, r], V[:, r], W[:, r])
            # But we can't easily do 3D outer in numpy without einsum
            
            # Use einsum:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a bit tricky because the indices are mapped differently.
            # Let's just do it directly for small sizes.
            
            # Compute the current tensor
            current = np.zeros((n, m, p))
            for r in range(rank):
                # U[:, r] is (16,), indexed by i*m+j
                # V[:, r] is (24,), indexed by j*p+k
                # W[:, r] is (24,), indexed by k*n+i
                for i in range(n):
                    for j in range(m):
                        u_val = U_np[i*m+j, r]
                        if abs(u_val) < 1e-15:
                            continue
                        for k in range(p):
                            v_val = V_np[j*p+k, r]
                            if abs(v_val) < 1e-15:
                                continue
                            w_val = W_np[k*n+i, r]
                            current[i, j, k] += u_val * v_val * w_val
            
            # Compute loss
            loss = np.sum((current - 1.0) ** 2)
            
            # Compute gradients
            # dL/dU[i*m+j, r] = 2 * (current[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # But we need to sum over k for each (i,j,r)
            
            # Let's compute gradients efficiently
            U_grad = np.zeros_like(U_np)
            V_grad = np.zeros_like(V_np)
            W_grad = np.zeros_like(W_np)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = current[i, j, k] - 1.0
                        if abs(diff) < 1e-15:
                            continue
                        for r in range(rank):
                            u_val = U_np[i*m+j, r]
                            v_val = V_np[j*p+k, r]
                            w_val = W_np[k*n+i, r]
                            U_grad[i*m+j, r] += 2 * diff * v_val * w_val
                            V_grad[j*p+k, r] += 2 * diff * u_val * w_val
                            W_grad[k*n+i, r] += 2 * diff * u_val * v_val
            
            # Update with Adam
            t += 1
            mU = beta1 * mU + (1 - beta1) * U_grad
            mV = beta1 * mV + (1 - beta1) * V_grad
            mW = beta1 * mW + (1 - beta1) * W_grad
            vU = beta2 * vU + (1 - beta2) * U_grad ** 2
            vV = beta2 * vV + (1 - beta2) * V_grad ** 2
            vW = beta2 * vW + (1 - beta2) * W_grad ** 2
            
            mU_hat = mU / (1 - beta1 ** t)
            mV_hat = mV / (1 - beta1 ** t)
            mW_hat = mW / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            
            U_np -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V_np -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W_np -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if it % 100 == 0:
                pass  # Could print progress, but we don't want to slow things down
        
        # Check if the decomposition is valid
        # Convert to lists
        U_list = U_np.tolist()
        V_list = V_np.tolist()
        W_list = W_np.tolist()
        
        if verify(U_list, V_list, W_list, rank, n, m, p):
            return U_list, V_list, W_list, True
        return None, None, None, False
    
    # Try different ranks
    for rank in [60, 55, 50, 45, 40, 35, 30, 25]:
        if time.time() > deadline - 30:
            break
        
        U_try, V_try, W_try, success = try_rank(rank, max_iters=1000, lr=0.05)
        if success:
            best_rank = rank
            best_U = U_try
            best_V = V_try
            best_W = W_try
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

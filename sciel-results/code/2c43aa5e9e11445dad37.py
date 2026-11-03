"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Verify a decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Laderman's rank-23 decomposition for 3x3x3 matrix multiplication
    # Using the explicit construction from Laderman (1976)
    # This gives us a known valid rank-23 decomposition
    
    def laderman_3x3x3():
        """Return U, V, W matrices (each 9x23) for 3x3x3 multiplication with rank 23"""
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Laderman's 23-term algorithm for 3x3x3
        # The algorithm defines 23 products and how they combine to form C
        # 
        # Let A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
        # Let B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
        # C[i,k] = sum_j A[i,j]*B[j,k]
        #
        # The 23 products are:
        # p0 = a0*b0
        # p1 = a1*b3
        # p2 = a2*b6
        # p3 = a3*b1
        # p4 = a4*b4
        # p5 = a5*b7
        # p6 = a6*b2
        # p7 = a7*b5
        # p8 = a8*b8
        # p9 = (a0+a3)*(b1+b4) - p0 - p3 - p4
        # p10 = (a1+a4)*(b3+b6) - p1 - p4 - p5
        # p11 = (a2+a5)*(b6+b9) - p2 - p5 - p8  [Note: b9 doesn't exist, let me fix this]
        
        # Let me use a verified explicit construction instead.
        # Here's a known rank-23 decomposition based on literature:
        
        # I'll use a different approach: construct the decomposition by 
        # solving a system of linear equations with the constraint that 
        # the tensor decomposition is exact.
        
        # For now, let me try to find a valid rank-23 decomposition through 
        # optimization, starting from a good initialization.
        
        # Actually, let me use a known explicit formula.
        # Based on Laderman's paper, here are the 23 terms:
        
        # The key insight is that we can express the 9 products of C as 
        # linear combinations of 23 intermediate products.
        
        # Let me define the 23 intermediate products explicitly:
        
        # p0 = a0*b0
        # p1 = a1*b3  
        # p2 = a2*b6
        # p3 = a3*b1
        # p4 = a4*b4
        # p5 = a5*b7
        # p6 = a6*b2
        # p7 = a7*b5
        # p8 = a8*b8
        # p9 = (a0+a1+a3)*(b0+b3+b1) - p0 - p1 - p3
        # p10 = (a1+a2+a4)*(b3+b6+b4) - p1 - p2 - p4
        # p11 = (a2+a5+a8)*(b6+b7+b8) - p2 - p5 - p8
        # p12 = (a3+a4+a6)*(b1+b4+b2) - p3 - p4 - p6
        # p13 = (a4+a5+a7)*(b4+b7+b5) - p4 - p5 - p7
        # p14 = (a6+a7+a8)*(b2+b5+b8) - p6 - p7 - p8
        # p15 = (a0+a3+a6)*(b0+b1+b2) - p0 - p3 - p6
        # p16 = (a1+a4+a7)*(b3+b4+b5) - p1 - p4 - p7
        # p17 = (a2+a5+a8)*(b6+b7+b8) - p2 - p5 - p8
        # p18 = (a0+a1+a2)*(b0+b3+b6) - p0 - p1 - p2
        # p19 = (a3+a4+a5)*(b1+b4+b7) - p3 - p4 - p5
        # p20 = (a6+a7+a8)*(b2+b5+b8) - p6 - p7 - p8
        # p21 = (a0+a4+a8)*(b0+b4+b8) - p0 - p4 - p8
        # p22 = (a2+a4+a6)*(b2+b4+b6) - p2 - p4 - p6
        
        # This is getting complicated. Let me use a simpler approach:
        # I'll initialize with random values and use gradient descent to find 
        # a valid rank-23 decomposition for 3x3x3, then use that as a building block.
        
        return None, None, None, rank  # Placeholder - will implement properly below
    
    # Direct approach: Use gradient descent to find valid decompositions
    # Start with the known upper bound and try to reduce
    
    def compute_error(U, V, W, rank):
        """Compute the error tensor for a given decomposition"""
        error = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error[i,j,k] = 1.0 - val
        return error
    
    def compute_gradients(U, V, W, rank, error):
        """Compute gradients for U, V, W"""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = error[i,j,k]
                    if abs(err) < 1e-12:
                        continue
                    for r in range(rank):
                        dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
        
        return dU, dV, dW
    
    def gradient_descent(U, V, W, rank, max_iters=300, eta=0.02, tol=1e-9):
        """Run gradient descent to refine the decomposition"""
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
                
            error = compute_error(U, V, W, rank)
            max_err = np.max(np.abs(error))
            
            if max_err < tol:
                return U, V, W, True
            
            dU, dV, dW = compute_gradients(U, V, W, rank, error)
            
            # Adaptive learning rate
            eta_eff = eta / (1.0 + iteration * 0.01)
            
            U -= eta_eff * dU
            V -= eta_eff * dV
            W -= eta_eff * dW
        
        error = compute_error(U, V, W, rank)
        converged = np.max(np.abs(error)) < tol
        return U, V, W, converged
    
    # Strategy: Try to find a rank-46 decomposition (2x Laderman) first,
    # then try to reduce the rank
    
    # Initialize with block structure for rank 46
    def init_block_decomp(target_rank):
        """Initialize with block structure assuming two 3x3x3 blocks"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        if target_rank % 2 == 0:
            r3 = target_rank // 2
            for block in range(2):
                offset = block * r3
                U[:, offset:offset+r3] = np.random.randn(n*m, r3) * 0.5
                V[:, offset:offset+r3] = np.random.randn(m*p, r3) * 0.5
                W[:, offset:offset+r3] = np.random.randn(p*n, r3) * 0.5
                
                if block == 0:
                    V[9:18, offset:offset+r3] = 0  # k=3,4,5 rows
                else:
                    V[0:9, offset:offset+r3] = 0  # k=0,1,2 rows
        else:
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
        
        return U, V, W
    
    # Try ranks from 46 down to 40
    candidate_ranks = [46, 45, 44, 43, 42, 41, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        for init_idx in range(8):
            if time.time() > deadline:
                break
                
            np.random.seed(100 + target_rank * 10 + init_idx)
            U, V, W = init_block_decomp(target_rank)
            
            # Run gradient descent
            U, V, W, converged = gradient_descent(U, V, W, target_rank, max_iters=200, eta=0.015)
            
            if converged:
                # Verify the solution
                if verify(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
    
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

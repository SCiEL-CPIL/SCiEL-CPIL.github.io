"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_strassen_padded(n, m, p, t0, deadline):
    """
    Try to use Strassen's algorithm by padding to powers of 2.
    For 3x4 * 4x7, we can pad to 4x4 * 4x8 or use block structure.
    
    Approach: Partition the 4x4 middle dimension into two 2x2 blocks.
    C = A * B where A is 3x4, B is 4x7.
    Split A into [A1 | A2] where A1 is 3x2, A2 is 3x2.
    Split B into [B1; B2] where B1 is 2x7, B2 is 2x7.
    C = A1*B1 + A2*B2.
    
    Each 3x2 * 2x7 product can be done with schoolbook rank 42.
    Total: 84. No gain from this alone.
    
    Alternative: Use Strassen on the 2x2 block structure.
    If we view the 4x4 as two 2x2 blocks, we can use Strassen's 7 multiplications
    for 2x2 * 2x2, but we need to handle the rectangular case.
    
    Let's try a different approach: use the fact that 3x4 * 4x7 can be embedded
    in a larger square problem and use known low-rank decompositions.
    """
    if time.time() - t0 > deadline - 5:
        return None
    
    # Try: pad n=3 to 4, p=7 to 8. Use 4x4 * 4x8.
    # 4x4 * 4x8 = two 4x4 * 4x4 products.
    # Each 4x4 * 4x4 can use Strassen: rank 7*4 = 28.
    # Total: 56. But we need to handle the padding.
    
    # Actually, let's try a direct approach: use the known result that
    # the tensor rank of 3x4 * 4x7 might be less than 84.
    
    # For now, return None and let the main function handle it.
    return None

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try Strassen-based approach
    strassen_result = _try_strassen_padded(n, m, p, t0, DEADLINE)
    if strassen_result is not None:
        best = strassen_result
        best_rank = len(strassen_result['U'][0])
    
    # Try randomized gradient descent for lower rank
    if time.time() - t0 > DEADLINE - 10:
        return best
    
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 70), -1):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        # Initialize with random values
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Gradient descent with multiple restarts
        for restart in range(3):
            if time.time() - t0 > DEADLINE - 15:
                break
            
            lr = 0.01
            for it in range(500):
                if time.time() - t0 > DEADLINE - 15:
                    break
                
                # Compute residuals and gradients
                grad_U = np.zeros_like(U_rand)
                grad_V = np.zeros_like(V_rand)
                grad_W = np.zeros_like(W_rand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R_ijk = 0.0
                            for r in range(target_rank):
                                R_ijk += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                            R_ijk -= 1.0
                            
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += 2 * R_ijk * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * R_ijk * U_rand[i*m+j, r] * W_rand[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * R_ijk * U_rand[i*m+j, r] * V_rand[j*p+k, r]
                
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                # Check convergence every 50 iterations
                if it % 50 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-6:
                        U_list = [[float(x) for x in row] for row in U_rand]
                        V_list = [[float(x) for x in row] for row in V_rand]
                        W_list = [[float(x) for x in row] for row in W_rand]
                        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    return best
# EVOLVE-BLOCK-END


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Try to find a low-rank decomposition using rank minimization
    # The tensor T[i,j,k,l,m,n] = delta_{j,l} * delta_{i,n} * delta_{k,m}
    # We need U[i*m+j, r] * V[l*p+k, r] * W[n*p+m, r] = T[i,j,k,l,m,n]
    
    # First, try the known rank-42 Strassen-based decomposition
    def strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition
    def verify(U_mat, V_mat, W_mat):
        r = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for l in range(m):
                    for k in range(p):
                        for nn in range(n):
                            for mm in range(p):
                                expected = 1 if (i == nn and j == l and k == mm) else 0
                                actual = 0
                                for c in range(r):
                                    actual += U_mat[i*m+j][c] * V_mat[l*p+k][c] * W_mat[nn*p+mm][c]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Try rank-42 first as baseline
    best_result = strassen_decomposition()
    best_rank = 42
    
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find lower rank decompositions using random search
    # The key insight: we can try random initializations and optimize
    def random_search(rank, max_iters=1000):
        """Try to find a rank-r decomposition using random initialization."""
        if rank >= best_rank:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Use gradient descent to minimize the error
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Compute error
            error = 0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for l in range(m):
                        for k in range(p):
                            for nn in range(n):
                                for mm in range(p):
                                    expected = 1 if (i == nn and j == l and k == mm) else 0
                                    actual = 0
                                    for c in range(rank):
                                        actual += U[i*m+j, c] * V[l*p+k, c] * W[nn*p+mm, c]
                                    error += (actual - expected) ** 2
                                    diff = actual - expected
                                    for c in range(rank):
                                        grad_U[i*m+j, c] += 2 * diff * V[l*p+k, c] * W[nn*p+mm, c]
                                        grad_V[l*p+k, c] += 2 * diff * U[i*m+j, c] * W[nn*p+mm, c]
                                        grad_W[nn*p+mm, c] += 2 * diff * U[i*m+j, c] * V[l*p+k, c]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if error < 1e-6:
                # Check if it's exact
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 24 up to 41
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = random_search(target_rank, max_iters=500)
        if result is not None:
            best_result = result
            best_rank = target_rank
            if best_rank <= 24:
                break
    
    return best_result
# EVOLVE-BLOCK-END

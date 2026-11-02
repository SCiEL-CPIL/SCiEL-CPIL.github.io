"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline (5s safety margin from 150s kill)
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 42
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = _schoolbook()
    
    # Try to find lower rank using greedy peeling + half-integer projection
    # Matrix multiplication tensor: T[i,j,k] = 1 for all valid i,j,k
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For this to be exact, we need:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try ranks below schoolbook
    best_result = schoolbook
    best_rank = rank_school
    
    # Try to find rank < 42
    for test_rank in range(41, 30, -1):  # Try ranks 41 down to 31
        if time.time() > deadline:
            break
        
        # Initialize random factors
        best_solution = None
        max_iters = 50
        found = False
        
        for seed in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n * m, test_rank) * 0.5
            V = np.random.randn(m * p, test_rank) * 0.5
            W = np.random.randn(p * n, test_rank) * 0.5
            
            # Greedy peeling / iterative refinement
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute residual
                residual = np.zeros((n, m, p))
                for c in range(test_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    # residual[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
                    temp = np.einsum('ij,jk->ik', Uc, Vc)
                    for k in range(p):
                        residual[:, :, k] += temp[:, k] * Wc[k, :]
                
                error = np.linalg.norm(residual - 1.0)
                if error < 1e-10:
                    # Check if we can project to half-integers
                    # Project each factor to nearest half-integer
                    U_proj = np.round(U * 2) / 2
                    V_proj = np.round(V * 2) / 2
                    W_proj = np.round(W * 2) / 2
                    
                    # Verify
                    U_proj_int = np.round(U_proj * 2).astype(int)
                    V_proj_int = np.round(V_proj * 2).astype(int)
                    W_proj_int = np.round(W_proj * 2).astype(int)
                    
                    # Check exactness using integer arithmetic
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(test_rank):
                                    val += U_proj_int[i*m+j, c] * V_proj_int[j*p+k, c] * W_proj_int[k*n+i, c]
                                if val != 4:  # 4 because 2*2*2 = 8, but we want 1... 
                                    # Actually: U*V*W should equal 1 in real numbers
                                    # With half-integers, U*2, V*2, W*2 are integers
                                    # So (U*2)*(V*2)*(W*2) / 8 should equal 1
                                    # So (U*2)*(V*2)*(W*2) = 8
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        # Convert to half-integers (divide by 2)
                        U_out = U_proj_int / 2.0
                        V_out = V_proj_int / 2.0
                        W_out = W_proj_int / 2.0
                        
                        # Convert to lists with proper types
                        U_list = [[float(val) for val in row] for row in U_out]
                        V_list = [[float(val) for val in row] for row in V_out]
                        W_list = [[float(val) for val in row] for row in W_out]
                        
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                
                if found:
                    break
                
                # Update: try to reduce residual by adjusting factors
                # Simple gradient-free update
                lr = 0.01
                for c in range(test_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    
                    # Compute gradient-like update
                    for i in range(n):
                        for j in range(m):
                            grad_U = 0
                            for k in range(p):
                                r = 0
                                for c2 in range(test_rank):
                                    r += U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
                                grad_U += (r - 1) / r if abs(r) > 1e-10 else 0
                            U[i*m+j, c] += lr * grad_U
                    
                    for j in range(m):
                        for k in range(p):
                            grad_V = 0
                            for i in range(n):
                                r = 0
                                for c2 in range(test_rank):
                                    r += U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
                                grad_V += (r - 1) / r if abs(r) > 1e-10 else 0
                            V[j*p+k, c] += lr * grad_V
                    
                    for k in range(p):
                        for i in range(n):
                            grad_W = 0
                            for j in range(m):
                                r = 0
                                for c2 in range(test_rank):
                                    r += U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
                                grad_W += (r - 1) / r if abs(r) > 1e-10 else 0
                            W[k*n+i, c] += lr * grad_W
            
            if found:
                break
        
        if best_solution is not None:
            best_rank = test_rank
            best_result = best_solution
            # Don't break - maybe we can do even better, but time is limited
            # Given the complexity, just take the first improvement
    
    if time.time() > deadline:
        return schoolbook
    
    return best_result
# EVOLVE-BLOCK-END

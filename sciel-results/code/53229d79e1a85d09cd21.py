import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Embed 3x4x5 into 4x4x5, then use structured decomposition
    # Actually, let's try a direct structured approach using Kronecker products
    # and known low-rank decompositions.
    
    # First, let's try the schoolbook rank as a baseline
    rank_schoolbook = n * m * p  # 60
    
    # Try to find a better rank using structured methods
    # The multiplication tensor for (n,m,p) has entries T[i,j,k] = delta_{i,j,k} 
    # where the tensor is defined by (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The rank is the tensor rank.
    
    # Let's try a greedy randomized approach with better heuristics
    # and also try structured decompositions.
    
    def get_target_tensor():
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i, j, k)] = 1.0
        return T
    
    def random_sparse_vector(length, max_nonzero=4, sparsity=0.3):
        vec = [0.0] * length
        num_nonzero = max(1, int(length * sparsity))
        num_nonzero = min(num_nonzero, max_nonzero, length)
        indices = random.sample(range(length), num_nonzero)
        for idx in indices:
            vec[idx] = random.choice([-1.0, 1.0])
        return vec
    
    def compute_inner_product(R, u, v, w):
        num = 0.0
        den = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_val = R.get((i, j, k), 0.0)
                    prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                    num += r_val * prod
                    den += prod * prod
        return num, den
    
    def try_find_decomposition(r, max_iterations=2000):
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            R = get_target_tensor()
            U_cols = []
            V_cols = []
            W_cols = []
            
            success = True
            for c in range(r):
                if time.time() > deadline:
                    return None
                
                # Generate random vectors with varying sparsity
                sparsity = random.choice([0.2, 0.3, 0.4, 0.5])
                u = random_sparse_vector(n * m, max_nonzero=5, sparsity=sparsity)
                v = random_sparse_vector(m * p, max_nonzero=5, sparsity=sparsity)
                w = random_sparse_vector(p * n, max_nonzero=5, sparsity=sparsity)
                
                num, den = compute_inner_product(R, u, v, w)
                
                if den == 0:
                    continue
                
                alpha = num / den
                alpha_rounded = round(alpha * 2) / 2.0
                
                if abs(alpha - alpha_rounded) > 1e-5:
                    continue
                
                # Update R
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            if prod != 0:
                                current_val = R.get((i, j, k), 0.0)
                                new_val = current_val - alpha_rounded * prod
                                if abs(new_val) < 1e-9:
                                    if (i, j, k) in R:
                                        del R[(i, j, k)]
                                else:
                                    R[(i, j, k)] = new_val
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
            
            if len(R) == 0:
                U = [[0.0] * r for _ in range(n * m)]
                V = [[0.0] * r for _ in range(m * p)]
                W = [[0.0] * r for _ in range(p * n)]
                
                for c in range(r):
                    for idx, val in enumerate(U_cols[c]):
                        if val != 0:
                            U[idx][c] = val
                    for idx, val in enumerate(V_cols[c]):
                        if val != 0:
                            V[idx][c] = val
                    for idx, val in enumerate(W_cols[c]):
                        if val != 0:
                            W[idx][c] = val
                
                # Verify
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1.0) > 1e-9:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    return U, V, W
        
        return None
    
    # Try structured decomposition: use Kronecker structure
    # For 3x4x5, we can try to decompose as a sum of Kronecker products
    # or use the fact that 4 = 2*2, 5 = 2*3+1, etc.
    
    # Let's try ranks from 58 down to 52
    best_result = None
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W
    
    # Try ranks 58, 57, 56, 55, 54, 53, 52
    for r in range(58, 51, -1):
        if time.time() > deadline:
            break
        
        max_iter = 3000 if r >= 56 else 2000
        res = try_find_decomposition(r, max_iter)
        if res:
            best_result = res
            break
    
    if best_result:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = get_schoolbook()
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

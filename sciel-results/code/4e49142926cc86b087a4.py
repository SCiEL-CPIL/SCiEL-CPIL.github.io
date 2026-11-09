import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp_fast(n, m, p, U, V, W):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        # Build expected tensor
        # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) with matching indices
        # a = i*m+j, b = j*p+k, c = k*n+i
        # Check: for each (i,j,k), U[i*m+j] . V[j*p+k] . W[k*n+i] should be 1
        # And for all other (a,b,c) combinations, should be 0
        
        # Efficient check: for each rank element r, compute outer products
        # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This is a 3-tensor. Check against expected.
        
        # Expected tensor
        T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute actual tensor
        T_actual = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            T_actual += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        return np.allclose(T_actual, T_expected, atol=1e-9)
    
    # Strategy: Use block structure. 3x9 = 3x(3x3)
    # Decompose as (2,3,3) ⊗ (1,1,3) with rank = rank_233 * 3
    # Also try direct (2,3,9) search with structured patterns
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 54
    
    # Method 1: Try to find good (2,3,3) decomposition, then Kronecker with (1,1,3)
    n2, m2, p2 = 2, 3, 3
    r_school_233 = n2 * m2 * p2  # 18
    
    # Try structured searches for (2,3,3)
    # Use the fact that we can try polynomial-based constructions
    
    found_233 = None
    
    # Try ranks from lower bound up
    # Lower bound for (2,3,3): at least max(2*3, 3*3, 3*2) = 9 by dimension argument
    # But more carefully, rank >= n*m = 6, >= m*p = 9, >= p*n = 6, so >= 9
    
    for target_rank in range(9, 19):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations with different structures
        for trial in range(200):
            if time.time() > deadline - 20:
                break
            
            U = np.zeros((n2*m2, target_rank))
            V = np.zeros((m2*p2, target_rank))
            W = np.zeros((p2*n2, target_rank))
            
            # Try different initialization strategies
            strategy = trial % 4
            
            if strategy == 0:
                # Random {-1,0,1}
                for r in range(target_rank):
                    U[:, r] = np.random.choice([-1, 0, 1], size=n2*m2)
                    V[:, r] = np.random.choice([-1, 0, 1], size=m2*p2)
                    W[:, r] = np.random.choice([-1, 0, 1], size=p2*n2)
            elif strategy == 1:
                # Sparse: each row has exactly 1 nonzero
                for r in range(target_rank):
                    pos_u = np.random.randint(0, n2*m2)
                    pos_v = np.random.randint(0, m2*p2)
                    pos_w = np.random.randint(0, p2*n2)
                    U[pos_u, r] = np.random.choice([-1, 1])
                    V[pos_v, r] = np.random.choice([-1, 1])
                    W[pos_w, r] = np.random.choice([-1, 1])
            elif strategy == 2:
                # Block structured: group by i, j, k
                for r in range(target_rank):
                    # Assign each (i,j,k) triple to some pattern
                    for i in range(n2):
                        for j in range(m2):
                            if np.random.random() < 0.5:
                                U[i*m2+j, r] = np.random.choice([-1, 0, 1])
                    for j in range(m2):
                        for k in range(p2):
                            if np.random.random() < 0.5:
                                V[j*p2+k, r] = np.random.choice([-1, 0, 1])
                    for k in range(p2):
                        for i in range(n2):
                            if np.random.random() < 0.5:
                                W[k*n2+i, r] = np.random.choice([-1, 0, 1])
            else:
                # Evaluation/interpolation style
                # Use distinct values
                for r in range(target_rank):
                    val = r - target_rank // 2
                    for i in range(n2):
                        for j in range(m2):
                            U[i*m2+j, r] = 1.0
                    for j in range(m2):
                        for k in range(p2):
                            V[j*p2+k, r] = val
                    for k in range(p2):
                        for i in range(n2):
                            W[k*n2+i, r] = 1.0 / val if val != 0 else 0.0
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomp_fast(n2, m2, p2, U_list, V_list, W_list):
                found_233 = (U_list, V_list, W_list, target_rank)
                break
        
        if found_233 is not None:
            break
    
    if found_233 is not None:
        U233, V233, W233, r233 = found_233
        total_rank = r233 * 3
        
        if total_rank < best_rank:
            # Kronecker product with (1,1,3)
            # (2,3,9) = (2,3,3) ⊗ (1,1,3)
            # U: (2*3) x (r233*3), V: (3*9) x (r233*3), W: (9*2) x (r233*3)
            # Indexing: a = i*3+j for U, b = j*9+k for V, c = k*2+i for W
            # For Kronecker: U[i,j; r1,r2] = U233[i,j;r1] * U113[0,0;r2]
            #               V[j,k; r1,r2] = V233[j,k_local;r1] * V113[0,k_local;r2] where k=k_block*3+k_local
            #               W[k,i; r1,r2] = W233[k_local,i;r1] * W113[k_block,0;r2]
            # U113: 1x1x3, V113: 1x3x3, W113: 3x1x3
            # U113[0][r2] = 1, V113[0][k_local][r2] = 1 if k_local==r2 else 0, W113[k_block][0][r2] = 1 if k_block==r2 else 0
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            for r1 in range(r233):
                for r2 in range(3):
                    r = r1 * 3 + r2
                    # U: U[i*m+j][r] = U233[i*m2+j][r1] * 1
                    for i in range(n):
                        for j in range(m):
                            a = i * m + j
                            U[a][r] = U233[i*m2+j][r1]
                    
                    # V: V[j*p+k][r] = V233[j*p2+k_local][r1] * (1 if k_block==r2 else 0)
                    for j in range(m):
                        for k in range(p):
                            k_block = k // 3
                            k_local = k % 3
                            b = j * p + k
                            if k_block == r2:
                                V[b][r] = V233[j*p2+k_local][r1]
                    
                    # W: W[k*n+i][r] = W233[k_local*n2+i][r1] * (1 if k_block==r2 else 0)
                    for k in range(p):
                        k_block = k // 3
                        k_local = k % 3
                        if k_block == r2:
                            for i in range(n):
                                c = k * n + i
                                W[c][r] = W233[k_local*n2+i][r1]
            
            if verify_decomp_fast(n, m, p, U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
    
    # Method 2: Direct search on (2,3,9) with lower ranks
    # Try a few structured approaches directly
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, 30, -1):
            if time.time() > deadline - 10:
                break
            
            found_direct = False
            for trial in range(100):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Structured: use block diagonal patterns
                for r in range(target_rank):
                    # Each rank element handles a subset of (i,j,k)
                    block = r % 3  # which 3x3 block of the 3x9
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.6:
                                U[i*m+j, r] = np.random.choice([-1, 0, 1])
                        for k in range(p):
                            k_local = k % 3
                            k_block = k // 3
                            if k_block == block and np.random.random() < 0.6:
                                V[j*p+k, r] = np.random.choice([-1, 0, 1])
                            for ii in range(n):
                                c = k * n + ii
                                if np.random.random() < 0.6:
                                    W[c, r] = np.random.choice([-1, 0, 1])
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify_decomp_fast(n, m, p, U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found_direct = True
                    break
            
            if found_direct:
                break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()

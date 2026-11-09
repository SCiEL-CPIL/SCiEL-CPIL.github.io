import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Construct the sparse tensor T over F2 (integers 0/1)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    T = np.zeros((nm, mp, np_), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, tol=1e-9):
        # Check if U, V, W exactly decompose T
        # Use random matrices to verify
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        U_mat = np.array(U).reshape(nm, -1)
        V_mat = np.array(V).reshape(mp, -1)
        W_mat = np.array(W).reshape(np_, -1)
        
        A_flat = A.flatten() # (nm,)
        B_flat = B.flatten() # (mp,)
        
        M1 = U_mat.T @ A_flat # (R,)
        M2 = V_mat.T @ B_flat # (R,)
        
        prod = M1 * M2
        
        C_flat = W_mat.T @ prod # (np_,)
        
        C_calc = C_flat.reshape(p, n).T
        
        if np.allclose(C_true, C_calc, atol=tol):
            return True
        return False

    start_time = time.time()
    deadline = 110.0
    
    best_rank = n * m * p
    best_result = None

    # Schoolbook fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((np_, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    if verify(U_school, V_school, W_school):
        best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Greedy Peeling over F2
    # We try to find a decomposition with rank < 105
    # We use a randomized greedy approach
    
    def find_decomposition_greedy(seed_offset=0):
        np.random.seed(seed_offset)
        T_cur = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            
            if not np.any(T_cur):
                break
            
            found_improvement = False
            best_u = None
            best_v = None
            best_w = None
            best_weight = np.sum(T_cur)
            
            # Try a few random sparse u, v
            for _ in range(20):
                if time.time() > deadline:
                    return None
                
                # Generate sparse u
                u = np.zeros(nm, dtype=np.int8)
                num_ones_u = np.random.randint(1, 4)
                indices_u = np.random.choice(nm, num_ones_u, replace=False)
                u[indices_u] = 1
                
                # Generate sparse v
                v = np.zeros(mp, dtype=np.int8)
                num_ones_v = np.random.randint(1, 4)
                indices_v = np.random.choice(mp, num_ones_v, replace=False)
                v[indices_v] = 1
                
                # Compute optimal w
                # M[a,b] = u[a] * v[b]
                M = np.outer(u, v)
                
                w = np.zeros(np_, dtype=np.int8)
                for c in range(np_):
                    # Count matches and mismatches
                    # T_cur[:,:,c] is the slice
                    # M is the mask
                    # If w_c=0, cost = sum(T_cur[:,:,c])
                    # If w_c=1, cost = sum(T_cur[:,:,c] != M)
                    
                    t_slice = T_cur[:, :, c]
                    # Cost if w_c=0
                    cost_0 = np.sum(t_slice)
                    
                    # Cost if w_c=1
                    # Mismatches where M=1 and T=0, and M=0 and T=1
                    # sum(T != M)
                    cost_1 = np.sum(t_slice != M)
                    
                    if cost_1 < cost_0:
                        w[c] = 1
                
                # Calculate new weight
                # New T = T_cur XOR (u outer v outer w)
                # term = np.einsum('a,b,c->abc', u, v, w)
                # We can compute weight without full array if possible, but array is small (15*35*21 = 11025)
                term = np.einsum('a,b,c->abc', u, v, w)
                T_new = T_cur ^ term
                new_weight = np.sum(T_new)
                
                if new_weight < best_weight:
                    best_weight = new_weight
                    best_u = u
                    best_v = v
                    best_w = w
                    found_improvement = True
            
            if found_improvement:
                T_cur = T_cur ^ np.einsum('a,b,c->abc', best_u, best_v, best_w)
                U_list.append(best_u)
                V_list.append(best_v)
                W_list.append(best_w)
            else:
                # Stuck. Break or try a different strategy?
                # For now, break to save time
                break
        
        if np.any(T_cur):
            return None
        
        # Construct matrices
        R = len(U_list)
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((np_, R))
        
        for i in range(R):
            U[:, i] = U_list[i]
            V[:, i] = V_list[i]
            W[:, i] = W_list[i]
            
        return U, V, W

    # Try a few seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        res = find_decomposition_greedy(seed)
        if res is not None:
            U, V, W = res
            R = U.shape[1]
            if R < best_rank:
                # Verify
                if verify(U, V, W):
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    # If we found a very good rank, we might stop early?
                    # But let's try to find even better
                    if R < 50:
                        break

    return best_result


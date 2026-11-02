import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target tensor for 3x4x4
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def objective(params, size_u, size_v, size_w, R, target):
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        T = tensor_contraction(U, V, W)
        return np.sum((T - target)**2)

    # Schoolbook solution
    best_rank = n * m * p
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

    # Try to find a rank 11 decomposition for M_{2,2,3}
    # This will allow us to construct a rank 44 solution for M_{3,4,4}
    
    n_sub, m_sub, p_sub = 2, 2, 3
    R_sub = 11
    su, sv, sw = n_sub * m_sub, m_sub * p_sub, p_sub * n_sub
    # su=4, sv=6, sw=6? No.
    # M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3.
    # U: 2*2=4 rows. V: 2*3=6 rows. W: 3*2=6 rows.
    # Wait, the indices in the prompt are:
    # U: i*m+j. V: j*p+k. W: k*n+i.
    # For M_{2,2,3}: n=2, m=2, p=3.
    # U: 2*2=4. V: 2*3=6. W: 3*2=6.
    # Target tensor T_{ijk} = 1 for i in 0..1, j in 0..1, k in 0..2.
    
    TARGET_SUB = np.zeros((su, sv, sw))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u = i * m_sub + j
                v = j * p_sub + k
                w = k * n_sub + i
                TARGET_SUB[u, v, w] = 1.0

    def objective_sub(params):
        U = params[:su * R_sub].reshape(su, R_sub)
        V = params[su * R_sub : su * R_sub + sv * R_sub].reshape(sv, R_sub)
        W = params[su * R_sub + sv * R_sub :].reshape(sw, R_sub)
        T = tensor_contraction(U, V, W)
        return np.sum((T - TARGET_SUB)**2)

    found_sub = False
    U_sub = V_sub = W_sub = None
    
    # Try to find sub-decomposition
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        np.random.seed(seed)
        params = np.random.randn(su * R_sub + sv * R_sub + sw * R_sub) * 0.5
        try:
            res = minimize(objective_sub, params, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
            if res.fun < 1e-6:
                U_sub = res.x[:su * R_sub].reshape(su, R_sub)
                V_sub = res.x[su * R_sub : su * R_sub + sv * R_sub].reshape(sv, R_sub)
                W_sub = res.x[su * R_sub + sv * R_sub :].reshape(sw, R_sub)
                
                # Snap to half-integers
                U_snap = np.round(U_sub * 2) / 2
                V_snap = np.round(V_sub * 2) / 2
                W_snap = np.round(W_sub * 2) / 2
                
                T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                if np.allclose(T_snap, TARGET_SUB, atol=1e-9):
                    U_sub, V_sub, W_sub = U_snap, V_snap, W_snap
                    found_sub = True
                    break
        except Exception:
            pass

    if found_sub:
        # Construct M_{3,4,4} from M_{2,2,3}
        # We need to map M_{2,2,3} to M_{3,2,2}
        # M_{2,2,3} is A(2x2) * B(2x3) = C(2x3).
        # We want M_{3,2,2} which is A(3x2) * B(2x2) = C(3x2).
        # They are isomorphic. 
        # Let's just use the found U_sub, V_sub, W_sub directly for the sub-blocks?
        # No, the dimensions must match.
        # The sub-problem we solved is for n=2, m=2, p=3.
        # The blocks in the main problem are 3x2 * 2x2.
        # This is n=3, m=2, p=2.
        # We can permute the indices of the solution to fit n=3, m=2, p=2.
        # Or we can just solve for n=3, m=2, p=2 directly with rank 11.
        
        # Let's restart the search for n=3, m=2, p=2 with rank 11.
        # It's the same problem, just permuted.
        # n=3, m=2, p=2.
        # su=6, sv=4, sw=6.
        
        n_sub2, m_sub2, p_sub2 = 3, 2, 2
        R_sub2 = 11
        su2, sv2, sw2 = n_sub2 * m_sub2, m_sub2 * p_sub2, p_sub2 * n_sub2
        # su2=6, sv2=4, sw2=6.
        
        TARGET_SUB2 = np.zeros((su2, sv2, sw2))
        for i in range(n_sub2):
            for j in range(m_sub2):
                for k in range(p_sub2):
                    u = i * m_sub2 + j
                    v = j * p_sub2 + k
                    w = k * n_sub2 + i
                    TARGET_SUB2[u, v, w] = 1.0

        def objective_sub2(params):
            U = params[:su2 * R_sub2].reshape(su2, R_sub2)
            V = params[su2 * R_sub2 : su2 * R_sub2 + sv2 * R_sub2].reshape(sv2, R_sub2)
            W = params[su2 * R_sub2 + sv2 * R_sub2 :].reshape(sw2, R_sub2)
            T = tensor_contraction(U, V, W)
            return np.sum((T - TARGET_SUB2)**2)

        found_sub2 = False
        U_sub2 = V_sub2 = W_sub2 = None
        
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            np.random.seed(seed + 100)
            params = np.random.randn(su2 * R_sub2 + sv2 * R_sub2 + sw2 * R_sub2) * 0.5
            try:
                res = minimize(objective_sub2, params, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
                if res.fun < 1e-6:
                    U_sub2 = res.x[:su2 * R_sub2].reshape(su2, R_sub2)
                    V_sub2 = res.x[su2 * R_sub2 : su2 * R_sub2 + sv2 * R_sub2].reshape(sv2, R_sub2)
                    W_sub2 = res.x[su2 * R_sub2 + sv2 * R_sub2 :].reshape(sw2, R_sub2)
                    
                    U_snap = np.round(U_sub2 * 2) / 2
                    V_snap = np.round(V_sub2 * 2) / 2
                    W_snap = np.round(W_sub2 * 2) / 2
                    
                    T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, TARGET_SUB2, atol=1e-9):
                        U_sub2, V_sub2, W_sub2 = U_snap, V_snap, W_snap
                        found_sub2 = True
                        break
            except Exception:
                pass

        if found_sub2:
            # Build 3x4x4 solution
            # A is 3x4. Split into A1, A2 (3x2).
            # B is 4x4. Split into B1, B2 (2x4).
            # B1 split into B11, B12 (2x2).
            # B2 split into B21, B22 (2x2).
            # C = A1 B11 + A1 B12 + A2 B21 + A2 B22.
            
            R_total = 4 * R_sub2 # 44
            
            U_final = np.zeros((size_u, R_total))
            V_final = np.zeros((size_v, R_total))
            W_final = np.zeros((size_w, R_total))
            
            # We need to map the indices.
            # The sub-decomposition is for A_sub (3x2), B_sub (2x2), C_sub (3x2).
            # U_sub2 has 6 rows (3x2). V_sub2 has 4 rows (2x2). W_sub2 has 6 rows (3x2).
            
            # Block 1: A1 * B11
            # A1 rows: i in 0..2, j in 0..1. Index in U_final: i*4 + j.
            # B11 rows: j in 0..1, k in 0..1. Index in V_final: j*4 + k.
            # C11 rows: k in 0..1, i in 0..2. Index in W_final: k*3 + i.
            
            # The sub U_sub2 row index is i*2 + j_sub.
            # The sub V_sub2 row index is j_sub*2 + k_sub.
            # The sub W_sub2 row index is k_sub*3 + i.
            
            # For A1 * B11:
            # i is 0..2, j_sub is 0..1 (corresponds to j=0,1 in main).
            # k_sub is 0..1 (corresponds to k=0,1 in main).
            
            offset = 0
            for i in range(3):
                for j in range(2): # j_sub
                    u_sub_idx = i * 2 + j
                    u_main_idx = i * 4 + j
                    U_final[u_main_idx, offset:offset+R_sub2] = U_sub2[u_sub_idx, :]
            
            for j in range(2): # j_sub
                for k in range(2): # k_sub
                    v_sub_idx = j * 2 + k
                    v_main_idx = j * 4 + k
                    V_final[v_main_idx, offset:offset+R_sub2] = V_sub2[v_sub_idx, :]
            
            for k in range(2): # k_sub
                for i in range(3):
                    w_sub_idx = k * 3 + i
                    w_main_idx = k * 3 + i
                    W_final[w_main_idx, offset:offset+R_sub2] = W_sub2[w_sub_idx, :]
            
            offset += R_sub2
            
            # Block 2: A1 * B12
            # A1 is same.
            # B12: j_sub in 0..1, k_sub in 0..1. But k in main is 2..3.
            # So k_main = 2 + k_sub.
            # v_main_idx = j * 4 + (2 + k_sub).
            
            for i in range(3):
                for j in range(2):
                    u_sub_idx = i * 2 + j
                    u_main_idx = i * 4 + j
                    U_final[u_main_idx, offset:offset+R_sub2] = U_sub2[u_sub_idx, :]
            
            for j in range(2):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = j * 4 + (2 + k)
                    V_final[v_main_idx, offset:offset+R_sub2] = V_sub2[v_sub_idx, :]
            
            for k in range(2):
                for i in range(3):
                    w_sub_idx = k * 3 + i
                    w_main_idx = (2 + k) * 3 + i
                    W_final[w_main_idx, offset:offset+R_sub2] = W_sub2[w_sub_idx, :]
            
            offset += R_sub2
            
            # Block 3: A2 * B21
            # A2: i in 0..2, j_sub in 0..1. j in main is 2..3.
            # j_main = 2 + j_sub.
            # u_main_idx = i * 4 + (2 + j_sub).
            # B21: j_sub in 0..1, k_sub in 0..1. k in main is 0..1.
            # v_main_idx = (2 + j_sub) * 4 + k_sub.
            # C21: k_sub in 0..1, i in 0..2.
            # w_main_idx = k_sub * 3 + i.
            
            for i in range(3):
                for j in range(2):
                    u_sub_idx = i * 2 + j
                    u_main_idx = i * 4 + (2 + j)
                    U_final[u_main_idx, offset:offset+R_sub2] = U_sub2[u_sub_idx, :]
            
            for j in range(2):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = (2 + j) * 4 + k
                    V_final[v_main_idx, offset:offset+R_sub2] = V_sub2[v_sub_idx, :]
            
            for k in range(2):
                for i in range(3):
                    w_sub_idx = k * 3 + i
                    w_main_idx = k * 3 + i
                    W_final[w_main_idx, offset:offset+R_sub2] = W_sub2[w_sub_idx, :]
            
            offset += R_sub2
            
            # Block 4: A2 * B22
            # A2: j_main = 2 + j_sub.
            # B22: j_sub in 0..1, k_sub in 0..1. k_main = 2 + k_sub.
            # v_main_idx = (2 + j_sub) * 4 + (2 + k_sub).
            # C22: k_main = 2 + k_sub.
            # w_main_idx = (2 + k_sub) * 3 + i.
            
            for i in range(3):
                for j in range(2):
                    u_sub_idx = i * 2 + j
                    u_main_idx = i * 4 + (2 + j)
                    U_final[u_main_idx, offset:offset+R_sub2] = U_sub2[u_sub_idx, :]
            
            for j in range(2):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = (2 + j) * 4 + (2 + k)
                    V_final[v_main_idx, offset:offset+R_sub2] = V_sub2[v_sub_idx, :]
            
            for k in range(2):
                for i in range(3):
                    w_sub_idx = k * 3 + i
                    w_main_idx = (2 + k) * 3 + i
                    W_final[w_main_idx, offset:offset+R_sub2] = W_sub2[w_sub_idx, :]
            
            # Verify
            T_final = tensor_contraction(U_final, V_final, W_final)
            if np.allclose(T_final, TARGET_TENSOR, atol=1e-9):
                best_rank = R_total
                best_solution = {
                    "U": U_final.tolist(),
                    "V": V_final.tolist(),
                    "W": W_final.tolist()
                }

    return best_solution

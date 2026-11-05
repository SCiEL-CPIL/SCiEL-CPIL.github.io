import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Best result so far
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Strategy: Use Strassen-like algorithms on sub-tensors
    
    # Attempt Rank 44 via Strassen on 2x2x2 sub-tensor (i,j,k in {0,1})
    U_new = []
    V_new = []
    W_new = []
    
    # Add the 37 non-overlapping schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                U_new.append(u)
                V_new.append(v)
                W_new.append(w)
    
    # Standard Strassen terms for 2x2x2
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 0, 0])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        # M6
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    # Mappings for sub-tensor to global indices
    # a = 3*i + j. i,j in {0,1}
    # 00->0, 01->1, 10->3, 11->4
    map_a = {0: 0, 1: 1, 2: 3, 3: 4}
    # b = 5*j + k. j,k in {0,1}
    # 00->0, 01->1, 10->5, 11->6
    map_b = {0: 0, 1: 1, 2: 5, 3: 6}
    # c = 3*k + i. k,i in {0,1}
    # 00->0, 01->1, 10->3, 11->4
    map_c = {0: 0, 1: 1, 2: 3, 3: 4}
    
    for u_sub, v_sub, w_sub in strassen_terms:
        u_global = np.zeros(nm)
        v_global = np.zeros(mp)
        w_global = np.zeros(pn)
        
        for s, g in map_a.items():
            u_global[g] = u_sub[s]
        for s, g in map_b.items():
            v_global[g] = v_sub[s]
        for s, g in map_c.items():
            w_global[g] = w_sub[s]
            
        U_new.append(u_global)
        V_new.append(v_global)
        W_new.append(w_global)
        
    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    if np.allclose(T_recon, T, atol=1e-10):
        best_U = U_final
        best_V = V_final
        best_W = W_final
        best_rank = len(U_new)
    
    # Try to find even lower rank using a search approach
    # For 3x3 * 3x5, known lower bounds suggest rank could be lower
    # Let's try a more systematic approach with different sub-tensor choices
    
    # Try Strassen on different 2x2x2 sub-tensors
    for i_start in range(n - 1):
        for j_start in range(m - 1):
            for k_start in range(p - 1):
                if time.time() > deadline:
                    break
                
                # Build sub-tensor indices
                sub_a_indices = [i_start * m + j_start, i_start * m + j_start + 1,
                                (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1]
                sub_b_indices = [j_start * p + k_start, j_start * p + k_start + 1,
                                (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1]
                sub_c_indices = [k_start * n + i_start, k_start * n + i_start + 1,
                                (k_start + 1) * n + i_start, (k_start + 1) * n + i_start + 1]
                
                # Count terms not in sub-tensor
                non_sub_terms = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            if a not in sub_a_indices or b not in sub_b_indices or c not in sub_c_indices:
                                non_sub_terms += 1
                
                # Total rank would be non_sub_terms + 7 (Strassen)
                total_rank = non_sub_terms + 7
                if total_rank >= best_rank:
                    continue
                
                # Build decomposition
                U_curr = []
                V_curr = []
                W_curr = []
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            if a in sub_a_indices and b in sub_b_indices and c in sub_c_indices:
                                continue
                            u = np.zeros(nm)
                            v = np.zeros(mp)
                            w = np.zeros(pn)
                            u[a] = 1
                            v[b] = 1
                            w[c] = 1
                            U_curr.append(u)
                            V_curr.append(v)
                            W_curr.append(w)
                
                # Add Strassen terms with proper mapping
                for u_sub, v_sub, w_sub in strassen_terms:
                    u_global = np.zeros(nm)
                    v_global = np.zeros(mp)
                    w_global = np.zeros(pn)
                    
                    for s, g in enumerate(sub_a_indices):
                        u_global[g] = u_sub[s]
                    for s, g in enumerate(sub_b_indices):
                        v_global[g] = v_sub[s]
                    for s, g in enumerate(sub_c_indices):
                        w_global[g] = w_sub[s]
                        
                    U_curr.append(u_global)
                    V_curr.append(v_global)
                    W_curr.append(w_global)
                
                U_test = np.column_stack(U_curr)
                V_test = np.column_stack(V_curr)
                W_test = np.column_stack(W_curr)
                
                T_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
                if np.allclose(T_test, T, atol=1e-10):
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank = len(U_curr)
    
    # Try rank 43, 42, etc. using a more aggressive search
    # Use a greedy approach to find minimal rank
    
    # For now, return the best found
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

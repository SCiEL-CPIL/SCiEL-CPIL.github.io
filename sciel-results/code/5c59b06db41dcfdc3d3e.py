import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    
                    u_vec = [(I, 1)]
                    v_vec = [(J, 1)]
                    w_vec = [(K, 1)]
                    add_term(u_vec, v_vec, w_vec)
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        def get_I(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def get_J(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def get_K(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        # M1
        add_term(
            [(get_I(0,0), 1), (get_I(1,1), 1)],
            [(get_J(0,0), 1), (get_J(1,1), 1)],
            [(get_K(0,0), 1), (get_K(1,1), 1)]
        )
        
        # M2
        add_term(
            [(get_I(1,0), 1), (get_I(1,1), 1)],
            [(get_J(0,0), 1)],
            [(get_K(0,1), 1), (get_K(1,1), -1)]
        )
        
        # M3
        add_term(
            [(get_I(0,0), 1)],
            [(get_J(0,1), 1), (get_J(1,1), -1)],
            [(get_K(1,0), 1), (get_K(1,1), 1)]
        )
        
        # M4
        add_term(
            [(get_I(1,1), 1)],
            [(get_J(1,0), 1), (get_J(0,0), -1)],
            [(get_K(0,0), 1), (get_K(0,1), 1)]
        )
        
        # M5
        add_term(
            [(get_I(0,0), 1), (get_I(0,1), 1)],
            [(get_J(1,1), 1)],
            [(get_K(0,0), -1), (get_K(1,0), 1)]
        )
        
        # M6
        add_term(
            [(get_I(1,0), 1), (get_I(0,0), -1)],
            [(get_J(0,0), 1), (get_J(0,1), 1)],
            [(get_K(1,1), 1)]
        )
        
        # M7
        add_term(
            [(get_I(0,1), 1), (get_I(1,1), -1)],
            [(get_J(1,0), 1), (get_J(1,1), 1)],
            [(get_K(0,0), 1)]
        )

    # Sub-problem 1: A11 B11 (2x2x2) -> Strassen
    strassen_2x2x2(0, 0, 0)
    
    # Sub-problem 2: A12 B21 (2x3x2) -> Strassen(2x2x2) + Schoolbook(2x1x2)
    # i in {0,1}, j in {2,3,4}, k in {0,1}
    # Split j: {2,3} and {4}
    strassen_2x2x2(0, 2, 0)  # i in {0,1}, j in {2,3}, k in {0,1}
    schoolbook(2, 1, 2, 0, 4, 0) # i in {0,1}, j in {4}, k in {0,1}
    
    # Sub-problem 3: A11 B12 (2x2x3) -> Strassen(2x2x2) + Schoolbook(2x2x1)
    # i in {0,1}, j in {0,1}, k in {2,3,4}
    # Split k: {2,3} and {4}
    strassen_2x2x2(0, 0, 2)  # i in {0,1}, j in {0,1}, k in {2,3}
    schoolbook(2, 2, 1, 0, 0, 4) # i in {0,1}, j in {0,1}, k in {4}
    
    # Sub-problem 4: A12 B22 (2x3x3) -> Strassen(2x2x2) + Schoolbook(2x1x2) + Schoolbook(2x3x1)
    # i in {0,1}, j in {2,3,4}, k in {2,3,4}
    # Split k: {2,3} and {4}
    # Block 1: k in {2,3} -> 2x3x2
    #   Split j: {2,3} and {4}
    strassen_2x2x2(0, 2, 2) # i in {0,1}, j in {2,3}, k in {2,3}
    schoolbook(2, 1, 2, 0, 4, 2) # i in {0,1}, j in {4}, k in {2,3}
    # Block 2: k in {4} -> 2x3x1
    schoolbook(2, 3, 1, 0, 2, 4) # i in {0,1}, j in {2,3,4}, k in {4}
    
    # Sub-problem 5: A21 B11 (2x2x2) -> Strassen
    strassen_2x2x2(2, 0, 0)
    
    # Sub-problem 6: A22 B21 (2x3x2) -> Strassen(2x2x2) + Schoolbook(2x1x2)
    # i in {2,3}, j in {2,3,4}, k in {0,1}
    strassen_2x2x2(2, 2, 0)  # i in {2,3}, j in {2,3}, k in {0,1}
    schoolbook(2, 1, 2, 2, 4, 0) # i in {2,3}, j in {4}, k in {0,1}
    
    # Sub-problem 7: A21 B12 (2x2x3) -> Strassen(2x2x2) + Schoolbook(2x2x1)
    # i in {2,3}, j in {0,1}, k in {2,3,4}
    strassen_2x2x2(2, 0, 2)  # i in {2,3}, j in {0,1}, k in {2,3}
    schoolbook(2, 2, 1, 2, 0, 4) # i in {2,3}, j in {0,1}, k in {4}
    
    # Sub-problem 8: A22 B22 (2x3x3) -> Strassen(2x2x2) + Schoolbook(2x1x2) + Schoolbook(2x3x1)
    # i in {2,3}, j in {2,3,4}, k in {2,3,4}
    # Split k: {2,3} and {4}
    # Block 1: k in {2,3} -> 2x3x2
    #   Split j: {2,3} and {4}
    strassen_2x2x2(2, 2, 2) # i in {2,3}, j in {2,3}, k in {2,3}
    schoolbook(2, 1, 2, 2, 4, 2) # i in {2,3}, j in {4}, k in {2,3}
    # Block 2: k in {4} -> 2x3x1
    schoolbook(2, 3, 1, 2, 2, 4) # i in {2,3}, j in {2,3,4}, k in {4}
    
    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()

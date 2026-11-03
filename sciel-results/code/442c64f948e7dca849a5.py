import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)

        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)

        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)

        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Naive for 3rd column
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)

    # Check time
    if time.time() > deadline:
        # Return whatever we have so far (should be empty if this happens at start, but we do it after construction)
        pass

    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    if time.time() > deadline:
         # If we are still here, we have finished construction, so return
         pass

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
# EVOLVE-BLOCK-END

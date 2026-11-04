import time

N = 3
M = 4
P = 6

def construct():
    n, m, p = N, M, P
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # i_off: offset in i (0 or 1)
    # j_off: offset in j (0 or 2)
    # k_off: offset in k (0, 2, 4)
    # base_r: offset in rank
    def add_strassen(i_off, j_off, k_off, base_r):
        # 7 terms of Strassen
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # We define the coefficients for U, V, W for each of the 7 ranks
        # U[i_off+i', j_off+j', base_r+r']
        # V[j_off+j', k_off+k', base_r+r']
        # W[k_off+k', i_off+i', base_r+r']
        
        # Term 1: (a+d)(e+h)
        # U: a(0,0)=1, d(1,1)=1 -> U[i_off+0, j_off+0] += 1, U[i_off+1, j_off+1] += 1
        # V: e(0,0)=1, h(1,1)=1 -> V[j_off+0, k_off+0] += 1, V[j_off+1, k_off+1] += 1
        # W: C00 += m1, C11 += m1 -> W[k_off+0, i_off+0] += 1, W[k_off+1, i_off+1] += 1
        
        # It's easier to define the 7 vectors explicitly.
        # Let's use a list of (u_indices, v_indices, w_indices, coeffs)
        # Or just hardcode the additions.
        
        # r = 0: m1
        U[i_off*2 + 0][base_r + 0] += 1
        U[i_off*2 + 1][base_r + 0] += 1
        V[j_off*2 + 0][base_r + 0] += 1
        V[j_off*2 + 1][base_r + 0] += 1
        W[(k_off*2 + 0)*3 + i_off][base_r + 0] += 1
        W[(k_off*2 + 1)*3 + i_off+1][base_r + 0] += 1
        
        # r = 1: m2 = (c+d)e
        # U: c(1,0)=1, d(1,1)=1
        # V: e(0,0)=1
        # W: C10 += m2
        U[i_off*2 + 1][base_r + 1] += 1
        U[i_off*2 + 1][base_r + 1] += 1 # Wait, c is (1,0), d is (1,1).
        # Correction:
        # c is A[1,0], d is A[1,1].
        U[i_off*2 + 1][base_r + 1] = 0 # Reset? No, we are adding to existing zeros.
        # Let's be precise.
        # U[i_off+1, j_off+0] += 1 (c)
        # U[i_off+1, j_off+1] += 1 (d)
        # V[j_off+0, k_off+0] += 1 (e)
        # W[k_off+0, i_off+1] += 1 (C10)
        
        # I will write a generic helper for adding a term to be safe.
        
    # Let's rewrite construct to use a cleaner addition method.
    
    def add_term(u_idx, v_idx, w_idx, r):
        U[u_idx[0]][u_idx[1]] # No, U is 2D list [row][col]
        # U[row][rank]
        # row = i*m + j
        # rank = r
        
        U[u_idx][r] += 1
        V[v_idx][r] += 1
        W[w_idx][r] += 1

    # This is getting complicated with indices. 
    # Let's stick to the previous logic but implemented carefully.
    
    pass

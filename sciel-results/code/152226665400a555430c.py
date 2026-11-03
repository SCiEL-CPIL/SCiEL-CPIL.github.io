import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 88
    # 4 blocks of 2x4x2 (rank 14 each) for rows 0,1
    # 1 block of 1x4x8 (rank 32) for row 2
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Winograd 2x2x2 decomposition coefficients
    # m1=ad, m2=bc, m3=(a+b)(e+g), m4=(b+d)(f+h), m5=(a+c)(e+g), m6=(c+d)(f+h), m7=(a+b+c+d)(e+f+g+h)
    # C11 = m1 + m3 - m5 + m7 - m4
    # C12 = m4 - m1 - m6 + m7
    # C21 = m2 - m3 + m5 + m6 - m7
    # C22 = m2 + m6 - m4 + m7
    
    # Helper to add a 2x2x2 Winograd block
    # a_idx: index in U for A[0,0]
    # b_idx: index in U for A[0,1]
    # c_idx: index in U for A[1,0]
    # d_idx: index in U for A[1,1]
    # e_idx: index in V for B[0,0]
    # f_idx: index in V for B[0,1]
    # g_idx: index in V for B[1,0]
    # h_idx: index in V for B[1,1]
    # k_idx: index in W for C[0,0]
    # l_idx: index in W for C[0,1]
    # m_idx: index in W for C[1,0]
    # n_idx: index in W for C[1,1]
    # offset: current column in U,V,W
    
    def add_winograd_2x2x2(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, k_idx, l_idx, m_idx, n_idx, offset):
        # m1 = a*d
        U[a_idx][offset] += 1
        V[e_idx][offset] += 1
        W[k_idx][offset] += 1
        W[l_idx][offset] -= 1
        W[m_idx][offset] += 0
        W[n_idx][offset] += 0
        # Wait, W coefficients depend on the specific C element.
        # C11 gets +1 from m1.
        # C12 gets -1 from m1.
        # C21 gets 0 from m1.
        # C22 gets 0 from m1.
        
        # Let's do it term by term to be safe.
        # m1 = a*d
        U[a_idx][offset] += 1
        V[e_idx][offset] += 1
        W[k_idx][offset] += 1
        W[l_idx][offset] -= 1
        
        # m2 = b*c
        U[b_idx][offset] += 1
        V[g_idx][offset] += 1
        W[m_idx][offset] += 1
        W[n_idx][offset] += 1
        
        # m3 = (a+b)(e+g)
        U[a_idx][offset] += 1
        U[b_idx][offset] += 1
        V[e_idx][offset] += 1
        V[g_idx][offset] += 1
        W[k_idx][offset] += 1
        W[l_idx][offset] += 0
        W[m_idx][offset] -= 1
        W[n_idx][offset] += 0
        
        # m4 = (b+d)(f+h)
        U[b_idx][offset] += 1
        U[d_idx][offset] += 1
        V[f_idx][offset] += 1
        V[h_idx][offset] += 1
        W[k_idx][offset] -= 1
        W[l_idx][offset] += 1
        W[m_idx][offset] += 0
        W[n_idx][offset] -= 1
        
        # m5 = (a+c)(e+g)
        U[a_idx][offset] += 1
        U[c_idx][offset] += 1
        V[e_idx][offset] += 1
        V[g_idx][offset] += 1
        W[k_idx][offset] -= 1
        W[l_idx][offset] += 0
        W[m_idx][offset] += 1
        W[n_idx][offset] += 0
        
        # m6 = (c+d)(f+h)
        U[c_idx][offset] += 1
        U[d_idx][offset] += 1
        V[f_idx][offset] += 1
        V[h_idx][offset] += 1
        W[k_idx][offset] += 0
        W[l_idx][offset] -= 1
        W[m_idx][offset] += 1
        W[n_idx][offset] += 1
        
        # m7 = (a+b+c+d)(e+f+g+h)
        U[a_idx][offset] += 1
        U[b_idx][offset] += 1
        U[c_idx][offset] += 1
        U[d_idx][offset] += 1
        V[e_idx][offset] += 1
        V[f_idx][offset] += 1
        V[g_idx][offset] += 1
        V[h_idx][offset] += 1
        W[k_idx][offset] += 1
        W[l_idx][offset] += 1
        W[m_idx][offset] -= 1
        W[n_idx][offset] += 1
        
        return offset + 7

    # Part 1: Rows 0, 1
    # 4 column pairs: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k_start = pair_idx * 2
        # A rows 0,1 (indices 0,1 in A, which are 0,1 in U)
        # A is 3x4. U index = i*m + j.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 4
        # i=1, j=1 -> 5
        # i=0, j=2 -> 2
        # i=0, j=3 -> 3
        # i=1, j=2 -> 6
        # i=1, j=3 -> 7
        
        # B cols k_start, k_start+1.
        # B is 4x8. V index = j*p + k.
        # j=0, k=k_start -> k_start
        # j=0, k=k_start+1 -> k_start+1
        # j=1, k=k_start -> 8+k_start
        # j=1, k=k_start+1 -> 8+k_start+1
        # j=2, k=k_start -> 16+k_start
        # j=2, k=k_start+1 -> 16+k_start+1
        # j=3, k=k_start -> 24+k_start
        # j=3, k=k_start+1 -> 24+k_start+1
        
        # C rows 0,1. Cols k_start, k_start+1.
        # W index = k*n + i.
        # i=0, k=k_start -> k_start*3 + 0
        # i=0, k=k_start+1 -> (k_start+1)*3 + 0
        # i=1, k=k_start -> k_start*3 + 1
        # i=1, k=k_start+1 -> (k_start+1)*3 + 1
        
        # Split inner dimension 4 into two 2s.
        # First half: j=0,1
        # A: i=0, j=0 (idx 0), i=0, j=1 (idx 1), i=1, j=0 (idx 4), i=1, j=1 (idx 5)
        # B: j=0, k=k_start (idx k_start), j=0, k=k_start+1 (idx k_start+1)
        #    j=1, k=k_start (idx 8+k_start), j=1, k=k_start+1 (idx 8+k_start+1)
        # C: i=0, k=k_start (idx 3*k_start), i=0, k=k_start+1 (idx 3*(k_start+1))
        #    i=1, k=k_start (idx 3*k_start+1), i=1, k=k_start+1 (idx 3*(k_start+1)+1)
        
        a00 = 0
        a01 = 1
        a10 = 4
        a11 = 5
        
        b00 = k_start
        b01 = k_start + 1
        b10 = 8 + k_start
        b11 = 8 + k_start + 1
        
        c00 = 3 * k_start
        c01 = 3 * (k_start + 1)
        c10 = 3 * k_start + 1
        c11 = 3 * (k_start + 1) + 1
        
        col = add_winograd_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, col)
        
        # Second half: j=2,3
        # A: i=0, j=2 (idx 2), i=0, j=3 (idx 3), i=1, j=2 (idx 6), i=1, j=3 (idx 7)
        # B: j=2, k=k_start (idx 16+k_start), j=2, k=k_start+1 (idx 16+k_start+1)
        #    j=3, k=k_start (idx 24+k_start), j=3, k=k_start+1 (idx 24+k_start+1)
        
        a00 = 2
        a01 = 3
        a10 = 6
        a11 = 7
        
        b00 = 16 + k_start
        b01 = 16 + k_start + 1
        b10 = 24 + k_start
        b11 = 24 + k_start + 1
        
        # C indices are the same
        col = add_winograd_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, col)

    # Part 2: Row 2
    # Schoolbook 1x4x8
    # A row 2: indices 8, 9, 10, 11
    # B all cols: 0..31
    # C row 2: indices 3*k + 2 for k=0..7
    
    for k in range(8):
        for j in range(4):
            U[8 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (28, R)
    # W: (p*n, R) = (14, R)
    
    R = 50
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Helper to add a rank-1 tensor
    # u_idx: list of (index, value) for U
    # v_idx: list of (index, value) for V
    # w_idx: list of (index, value) for W
    def add_tensor(u_idx, v_idx, w_idx):
        nonlocal col
        for idx, val in u_idx:
            U[idx, col] += val
        for idx, val in v_idx:
            V[idx, col] += val
        for idx, val in w_idx:
            W[idx, col] += val
        col += 1
        
    # Strassen block
    # A indices: a,b,c,d -> flat indices
    # B indices: e,f,g,h -> flat indices
    # C indices: c11, c12, c21, c22 -> flat indices
    def add_strassen(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c11_idx, c12_idx, c21_idx, c22_idx):
        # M1 = (a+d)(e+h)
        add_tensor([(a_idx, 1), (d_idx, 1)], [(e_idx, 1), (h_idx, 1)], [(c11_idx, 1), (c22_idx, 1)])
        # M2 = (c+d)e
        add_tensor([(c_idx, 1), (d_idx, 1)], [(e_idx, 1)], [(c21_idx, 1), (c22_idx, -1)])
        # M3 = a(f-h)
        add_tensor([(a_idx, 1)], [(f_idx, 1), (h_idx, -1)], [(c12_idx, 1), (c22_idx, 1)])
        # M4 = d(g-e)
        add_tensor([(d_idx, 1)], [(g_idx, 1), (e_idx, -1)], [(c11_idx, 1), (c21_idx, 1)])
        # M5 = (a+b)h
        add_tensor([(a_idx, 1), (b_idx, 1)], [(h_idx, 1)], [(c11_idx, -1), (c12_idx, 1)])
        # M6 = (c-a)(e+f)
        add_tensor([(c_idx, 1), (a_idx, -1)], [(e_idx, 1), (f_idx, 1)], [(c22_idx, 1)])
        # M7 = (b-d)(g+h)
        add_tensor([(b_idx, 1), (d_idx, -1)], [(g_idx, 1), (h_idx, 1)], [(c11_idx, 1)])

    # Single block (2x2x1)
    # A indices: a,b,c,d
    # B indices: e,g (e is top, g is bottom)
    # C indices: c1, c2
    def add_single(a_idx, b_idx, c_idx, d_idx, e_idx, g_idx, c1_idx, c2_idx):
        add_tensor([(a_idx, 1)], [(e_idx, 1)], [(c1_idx, 1)])
        add_tensor([(b_idx, 1)], [(g_idx, 1)], [(c1_idx, 1)])
        add_tensor([(c_idx, 1)], [(e_idx, 1)], [(c2_idx, 1)])
        add_tensor([(d_idx, 1)], [(g_idx, 1)], [(c2_idx, 1)])

    # A indices
    # A is 2x4.
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    a_left = [0, 1, 2, 3]
    a_right = [4, 5, 6, 7]
    
    # B indices
    # B is 4x7.
    # B[j, k] -> j*7 + k
    # Top rows: j=0,1
    # Bottom rows: j=2,3
    
    # C indices
    # C is 2x7.
    # C[i, k] -> i*7 + k
    # i=0,1
    
    # Process C1 = A_left * B_top
    # B_top has 7 columns.
    # Columns 0,1 -> Strassen
    # Columns 2,3 -> Strassen
    # Columns 4,5 -> Strassen
    # Column 6 -> Single
    
    # Strassen 1: B cols 0,1
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    add_strassen(a_left[0], a_left[1], a_left[2], a_left[3], 
                 0*7+0, 0*7+1, 1*7+0, 1*7+1,
                 0*7+0, 0*7+1, 1*7+0, 1*7+1)
                 
    # Strassen 2: B cols 2,3
    add_strassen(a_left[0], a_left[1], a_left[2], a_left[3], 
                 0*7+2, 0*7+3, 1*7+2, 1*7+3,
                 0*7+2, 0*7+3, 1*7+2, 1*7+3)
                 
    # Strassen 3: B cols 4,5
    add_strassen(a_left[0], a_left[1], a_left[2], a_left[3], 
                 0*7+4, 0*7+5, 1*7+4, 1*7+5,
                 0*7+4, 0*7+5, 1*7+4, 1*7+5)
                 
    # Single: B col 6
    # e=B[0,6], g=B[1,6]
    add_single(a_left[0], a_left[1], a_left[2], a_left[3],
               0*7+6, 1*7+6,
               0*7+6, 1*7+6)
               
    # Process C2 = A_right * B_bottom
    # B_bottom has 7 columns.
    # Rows 2,3.
    # e=B[2,k], g=B[3,k]
    
    # Strassen 1: B cols 0,1
    add_strassen(a_right[0], a_right[1], a_right[2], a_right[3], 
                 2*7+0, 2*7+1, 3*7+0, 3*7+1,
                 0*7+0, 0*7+1, 1*7+0, 1*7+1)
                 
    # Strassen 2: B cols 2,3
    add_strassen(a_right[0], a_right[1], a_right[2], a_right[3], 
                 2*7+2, 2*7+3, 3*7+2, 3*7+3,
                 0*7+2, 0*7+3, 1*7+2, 1*7+3)
                 
    # Strassen 3: B cols 4,5
    add_strassen(a_right[0], a_right[1], a_right[2], a_right[3], 
                 2*7+4, 2*7+5, 3*7+4, 3*7+5,
                 0*7+4, 0*7+5, 1*7+4, 1*7+5)
                 
    # Single: B col 6
    add_single(a_right[0], a_right[1], a_right[2], a_right[3],
               2*7+6, 3*7+6,
               0*7+6, 1*7+6)

    return {"U": U, "V": V, "W": W}

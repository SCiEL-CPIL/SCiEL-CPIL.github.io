import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # We aim for rank 98
    # Blocks:
    # A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # C11 (2x2), C12 (2x3), C21 (2x2), C22 (2x3)
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Ranks:
    # A11 B11: Strassen 7
    # A12 B21: Schoolbook 12
    # A11 B12: Schoolbook 12
    # A12 B22: Schoolbook 18
    # A21 B11: Strassen 7
    # A22 B21: Schoolbook 12
    # A21 B12: Schoolbook 12
    # A22 B22: Schoolbook 18
    # Total: 98
    
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen_2x2x2(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal col
        # Strassen terms for 2x2x2
        # U indices: (a_i_off+di)*m + (a_j_off+dj)
        # V indices: (b_j_off+dj)*p + (b_k_off+dk)
        # W indices: (c_k_off+dk)*n + (c_i_off+di)
        
        # M1
        U[(a_i_off+0)*m + (a_j_off+0), col] = 1
        U[(a_i_off+1)*m + (a_j_off+1), col] = 1
        V[(b_j_off+0)*p + (b_k_off+0), col] = 1
        V[(b_j_off+1)*p + (b_k_off+1), col] = 1
        W[(c_k_off+0)*n + (c_i_off+0), col] = 1
        W[(c_k_off+1)*n + (c_i_off+1), col] = 1
        col += 1
        
        # M2
        U[(a_i_off+1)*m + (a_j_off+0), col] = 1
        U[(a_i_off+1)*m + (a_j_off+1), col] = 1
        V[(b_j_off+0)*p + (b_k_off+0), col] = 1
        W[(c_k_off+0)*n + (c_i_off+1), col] = 1
        col += 1
        
        # M3
        U[(a_i_off+0)*m + (a_j_off+0), col] = 1
        V[(b_j_off+0)*p + (b_k_off+1), col] = 1
        V[(b_j_off+1)*p + (b_k_off+1), col] = -1
        W[(c_k_off+1)*n + (c_i_off+0), col] = 1
        col += 1
        
        # M4
        U[(a_i_off+1)*m + (a_j_off+1), col] = 1
        V[(b_j_off+1)*p + (b_k_off+0), col] = 1
        V[(b_j_off+0)*p + (b_k_off+0), col] = -1
        W[(c_k_off+0)*n + (c_i_off+0), col] = -1
        W[(c_k_off+1)*n + (c_i_off+1), col] = 1
        col += 1
        
        # M5
        U[(a_i_off+0)*m + (a_j_off+0), col] = 1
        U[(a_i_off+0)*m + (a_j_off+1), col] = 1
        V[(b_j_off+1)*p + (b_k_off+1), col] = 1
        W[(c_k_off+1)*n + (c_i_off+0), col] = 1
        col += 1
        
        # M6
        U[(a_i_off+1)*m + (a_j_off+0), col] = 1
        U[(a_i_off+0)*m + (a_j_off+0), col] = -1
        V[(b_j_off+0)*p + (b_k_off+0), col] = 1
        V[(b_j_off+0)*p + (b_k_off+1), col] = 1
        W[(c_k_off+0)*n + (c_i_off+0), col] = 1
        col += 1
        
        # M7
        U[(a_i_off+0)*m + (a_j_off+1), col] = 1
        U[(a_i_off+1)*m + (a_j_off+1), col] = -1
        V[(b_j_off+1)*p + (b_k_off+0), col] = 1
        V[(b_j_off+1)*p + (b_k_off+1), col] = 1
        W[(c_k_off+1)*n + (c_i_off+1), col] = 1
        col += 1

    def add_schoolbook(a_i_off, a_j_off, a_i_dim, a_j_dim, b_j_off, b_k_off, b_j_dim, b_k_dim, c_i_off, c_k_off):
        nonlocal col
        for di in range(a_i_dim):
            for dj in range(a_j_dim):
                for dk in range(b_k_dim):
                    U[(a_i_off+di)*m + (a_j_off+dj), col] = 1
                    V[(b_j_off+dj)*p + (b_k_off+dk), col] = 1
                    W[(c_k_off+dk)*n + (c_i_off+di), col] = 1
                    col += 1

    # 1. C11 = A11 B11 (Strassen)
    # A11: i=0..1, j=0..1
    # B11: j=0..1, k=0..1
    # C11: i=0..1, k=0..1
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # 2. C11 += A12 B21 (Schoolbook)
    # A12: i=0..1, j=2..4 (dim 2x3)
    # B21: j=2..4, k=0..1 (dim 3x2)
    # C11: i=0..1, k=0..1
    add_schoolbook(0, 2, 2, 3, 2, 0, 3, 2, 0, 0)
    
    # 3. C12 = A11 B12 (Schoolbook)
    # A11: i=0..1, j=0..1 (dim 2x2)
    # B12: j=0..1, k=2..4 (dim 2x3)
    # C12: i=0..1, k=2..4
    add_schoolbook(0, 0, 2, 2, 0, 2, 2, 3, 0, 2)
    
    # 4. C12 += A12 B22 (Schoolbook)
    # A12: i=0..1, j=2..4 (dim 2x3)
    # B22: j=2..4, k=2..4 (dim 3x3)
    # C12: i=0..1, k=2..4
    add_schoolbook(0, 2, 2, 3, 2, 2, 3, 3, 0, 2)
    
    # 5. C21 = A21 B11 (Strassen)
    # A21: i=2..3, j=0..1
    # B11: j=0..1, k=0..1
    # C21: i=2..3, k=0..1
    add_strassen_2x2x2(2, 0, 0, 0, 2, 0)
    
    # 6. C21 += A22 B21 (Schoolbook)
    # A22: i=2..3, j=2..4 (dim 2x3)
    # B21: j=2..4, k=0..1 (dim 3x2)
    # C21: i=2..3, k=0..1
    add_schoolbook(2, 2, 2, 3, 2, 0, 3, 2, 2, 0)
    
    # 7. C22 = A21 B12 (Schoolbook)
    # A21: i=2..3, j=0..1 (dim 2x2)
    # B12: j=0..1, k=2..4 (dim 2x3)
    # C22: i=2..3, k=2..4
    add_schoolbook(2, 0, 2, 2, 0, 2, 2, 3, 2, 2)
    
    # 8. C22 += A22 B22 (Schoolbook)
    # A22: i=2..3, j=2..4 (dim 2x3)
    # B22: j=2..4, k=2..4 (dim 3x3)
    # C22: i=2..3, k=2..4
    add_schoolbook(2, 2, 2, 3, 2, 2, 3, 3, 2, 2)
    
    return {"U": U, "V": V, "W": W}

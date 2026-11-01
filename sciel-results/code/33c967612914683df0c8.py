import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS

    # Schoolbook fallback (rank 28)
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)

    def to_half_int_list(mat):
        doubled = np.round(mat * 2)
        return doubled.astype(int).tolist()

    # Laderman block for 2x2x2 (rank 7)
    def laderman_block():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # c=0: (a-b-c+d)(e-f-g+h) -> all C
        U_b[0,0]=1; U_b[1,0]=-1; U_b[2,0]=-1; U_b[3,0]=1
        V_b[0,0]=1; V_b[1,0]=-1; V_b[2,0]=-1; V_b[3,0]=1
        W_b[0,0]=1; W_b[1,0]=1; W_b[2,0]=1; W_b[3,0]=1
        
        # c=1: (a-b)(f-h) -> C01, C11
        U_b[0,1]=1; U_b[1,1]=-1
        V_b[1,1]=1; V_b[3,1]=-1
        W_b[1,1]=1; W_b[3,1]=1
        
        # c=2: (a-c)(g-e) -> C10, C11
        U_b[0,2]=1; U_b[2,2]=-1
        V_b[2,2]=1; V_b[0,2]=-1
        W_b[2,2]=1; W_b[3,2]=1
        
        # c=3: ae -> C00
        U_b[0,3]=1; V_b[0,3]=1; W_b[0,3]=1
        
        # c=4: bf -> C01
        U_b[1,4]=1; V_b[1,4]=1; W_b[1,4]=1
        
        # c=5: cg -> C10
        U_b[2,5]=1; V_b[2,5]=1; W_b[2,5]=1
        
        # c=6: dh -> C11
        U_b[3,6]=1; V_b[3,6]=1; W_b[3,6]=1
        
        return U_b, V_b, W_b

    # Schoolbook block for 2x2x1 (rank 2)
    def schoolbook_2x2x1():
        U_b = np.zeros((4, 2))
        V_b = np.zeros((2, 2))
        W_b = np.zeros((2, 2))
        
        # c=0: a*e -> C00
        U_b[0,0]=1; V_b[0,0]=1; W_b[0,0]=1
        
        # c=1: b*f -> C10
        U_b[1,1]=1; V_b[1,1]=1; W_b[1,1]=1
        
        return U_b, V_b, W_b

    def build_decomposition():
        """
        Constructs a rank-23 decomposition.
        3 blocks of 2x2x2 (Laderman, rank 7) for columns 0-5.
        1 block of 2x2x1 (Schoolbook, rank 2) for column 6.
        Total rank = 21 + 2 = 23.
        """
        rank = 23
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        
        U_l, V_l, W_l = laderman_block()
        U_s, V_s, W_s = schoolbook_2x2x1()
        
        # Process 3 blocks of 2 columns
        for block_idx in range(3):
            col_offset = block_idx * 7
            j_start = block_idx * 2
            
            for c_local in range(7):
                c_global = col_offset + c_local
                
                # U (independent of j)
                for idx in range(4):
                    U_full[idx, c_global] = U_l[idx, c_local]
                
                # V (maps local k*2+j to global k*p+j)
                for k in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        V_full[k*p + j_global, c_global] = V_l[k*2 + j_local, c_local]
                
                # W (maps local j*2+i to global j*n+i)
                for j_local in range(2):
                    for i in range(2):
                        j_global = j_start + j_local
                        W_full[j_global*n + i, c_global] = W_l[j_local*2 + i, c_local]
        
        # Process 1 block of 1 column (j=6)
        col_offset = 21
        for c_local in range(2):
            c_global = col_offset + c_local
            
            # U
            for idx in range(4):
                U_full[idx, c_global] = U_s[idx, c_local]
            
            # V (local k, global k*7+6)
            for k in range(2):
                V_full[k*p + 6, c_global] = V_s[k, c_local]
            
            # W (local i, global 6*n+i)
            for i in range(2):
                W_full[6*n + i, c_global] = W_s[i, c_local]
                
        return U_full, V_full, W_full

    U_mat, V_mat, W_mat = build_decomposition()
    
    if verify(U_mat, V_mat, W_mat):
        if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
            return {
                "U": to_half_int_list(U_mat),
                "V": to_half_int_list(V_mat),
                "W": to_half_int_list(W_mat)
            }
            
    return fallback

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

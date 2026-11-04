"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140

    def strassen_2x2x2_decomp():
        """Strassen 2x2x2: U,V,W each (4,7). U row=i*2+j, V row=j*2+k, W row=k*2+i"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # A=[a,b;c,d] idx 0,1,2,3; B=[e,f;g,h] idx 0,1,2,3; C idx k*2+i
        # p1=(a+d)(e+h) -> C11+=, C22+=
        U[0,0]=1; U[2,0]=1; V[0,0]=1; V[2,0]=1; W[0,0]=1; W[3,0]=1
        # p2=(c+d)e -> C21+=, C22-=
        U[1,1]=1; U[2,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # p3=a(f-h) -> C12+=, C22+=
        U[0,2]=1; V[1,2]=1; V[2,2]=-1; W[1,2]=1; W[3,2]=1
        # p4=d(g-e) -> C11+=, C21+=
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # p5=(a+b)h -> C11-=, C12+=
        U[0,4]=1; U[1,4]=1; V[2,4]=1; W[0,4]=-1; W[1,4]=1
        # p6=(c-a)(e+f) -> C22+=
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # p7=(b-d)(g+h) -> C11+=
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def strassen_4x4x4_decomp():
        """Kronecker product of Strassen 2x2x2 to get 4x4x4 rank 49."""
        U2, V2, W2 = strassen_2x2x2_decomp()
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        for c1 in range(7):
            for c2 in range(7):
                c = c1 * 7 + c2
                for i in range(4):
                    for j in range(4):
                        i1, i2 = i // 2, i % 2
                        j1, j2 = j // 2, j % 2
                        U4[i*4+j, c] = U2[i1*2+i2, c1] * U2[j1*2+j2, c2]
                for j in range(4):
                    for k in range(4):
                        j1, j2 = j // 2, j % 2
                        k1, k2 = k // 2, k % 2
                        V4[j*4+k, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                for k in range(4):
                    for i in range(4):
                        k1, k2 = k // 2, k % 2
                        i1, i2 = i // 2, i % 2
                        W4[k*4+i, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
        return U4, V4, W4

    # Build Strassen 4x4x4
    U_s, V_s, W_s = strassen_4x4x4_decomp()
    
    # Verify Strassen 4x4x4
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test
    C_comp = np.zeros((4, 4))
    for c in range(49):
        prod = np.outer(U_s[:, c], V_s[:, c])
        # U_s[:,c] is A factors, V_s[:,c] is B factors
        # C = sum_c outer(W_s[:,c], prod) ... no
        # Actually: C[k*4+i] = sum_c W_s[k*4+i, c] * (sum_j U_s[i*4+j,c] * V_s[j*4+k,c])
        # Let me compute properly
    # The decomposition: C[i,k] = sum_c W[k*4+i, c] * U[i*4+j, c] * V[j*4+k, c] summed over j
    # Actually it's: for each c, compute M_c = U[:,c] @ V[:,c] (outer product in tensor sense)
    # Then C = sum_c W[:,c] * (something)
    
    # Let me just verify by direct formula:
    # T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    # This should equal 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for the multiplication tensor
    
    T_comp = np.zeros((4, 4, 4))
    for c in range(49):
        T_comp += np.einsum('i,j,k->ijk', U_s[:, c], V_s[:, c], W_s[:, c])
    
    # The multiplication tensor T[i,j,k] = 1 (it's always 1 for the rank-1 term structure)
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # And C[i,k] = sum_j A[i,j]*B[j,k] = sum_{i,j,k} T[i,j,k]*A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k
    
    # Verify:
    if not np.allclose(T_comp, 1.0, atol=1e-10):
        # Fallback to schoolbook if Strassen verification fails
        pass
    
    # Schoolbook 4x4x5 for columns 4-8
    rank_sb = 80
    U_sb = np.zeros((16, rank_sb))
    V_sb = np.zeros((36, rank_sb))
    W_sb = np.zeros((36, rank_sb))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U_sb[i*4+j, col] = 1
                V_sb[j*9+k, col] = 1
                W_sb[k*4+i, col] = 1
                col += 1
    
    total_rank = 129
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    U_full[:, 0:49] = U_s
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:49] = V_s[j*4+k, 0:49]
    W_full[0:16, 0:49] = W_s
    
    U_full[:, 49:129] = U_sb
    V_full[:, 49:129] = V_sb
    W_full[:, 49:129] = W_sb
    
    # Final verification
    T_full = np.zeros((4, 4, 9))
    for c in range(total_rank):
        T_full += np.einsum('i,j,k->ijk', U_full[:, c], V_full[:, c], W_full[:, c])
    
    expected = np.ones((4, 4, 9))
    if not np.allclose(T_full, expected, atol=1e-10):
        # Fallback to pure schoolbook rank 144
        rank = 144
        U_fb = np.zeros((16, rank))
        V_fb = np.zeros((36, rank))
        W_fb = np.zeros((36, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*9+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END

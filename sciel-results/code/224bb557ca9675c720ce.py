"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # A[i,j] indexed by i*m+j in U
    # B[j,k] indexed by j*p+k in V
    # C[i,k] indexed by k*n+i in W
    
    # Strategy: Split middle dimension j into {0,1} and {2,3}
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # = A1 @ B1 + A2 @ B2
    # where A1 is 2x2, B1 is 2x5, A2 is 2x2, B2 is 2x5
    # Split each B into 2x2 and 2x3 parts
    # B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x3
    # B2 = [B2a | B2b] where B2a is 2x2, B2b is 2x3
    # C[:,0:2] = A1@B1a + A2@B2a (each 2x2*2x2, Strassen: 7 each = 14)
    # C[:,2:5] = A1@B1b + A2@B2b (each 2x2*2x3, schoolbook: 6 each = 12)
    # Total: 26
    
    rank = 26
    U = [[0]*rank for _ in range(n*m)]   # 8 x 26
    V = [[0]*rank for _ in range(m*p)]   # 20 x 26
    W = [[0]*rank for _ in range(p*n)]   # 10 x 26
    
    col = 0
    
    # Helper: add a Strassen decomposition for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B, output indices: C[0,0], C[0,1], C[1,0], C[1,1]
    # u_base: index of a00 in U (i*m+j)
    # v_base: index of b00 in V (j*p+k)
    # w_base: index of C[0,0] in W (k*n+i)
    def strassen_2x2(u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11):
        nonlocal col
        # p1 = (a00+a11)*(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        # p2 = (a10+a11)*b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        # p3 = a00*(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1
        col += 1
        # p4 = a11*(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        # p5 = (a00+a01)*b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        # p6 = (a10-a00)*(b00+b01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w10][col] += -1; W[w11][col] += 1
        col += 1
        # p7 = (a01-a11)*(b10+b11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
    
    # Helper: add schoolbook for 2x2 * 2x3
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # 6 multiplications
    def schoolbook_2x2x2x3(u00, u01, u10, u11, v00, v01, v02, v10, v11, v12, w00, w01, w02, w10, w11, w12):
        nonlocal col
        # a00*b00
        U[u00][col] += 1; V[v00][col] += 1; W[w00][col] += 1; col += 1
        # a00*b01
        U[u00][col] += 1; V[v01][col] += 1; W[w01][col] += 1; col += 1
        # a00*b02
        U[u00][col] += 1; V[v02][col] += 1; W[w02][col] += 1; col += 1
        # a01*b10
        U[u01][col] += 1; V[v10][col] += 1; W[w00][col] += 1; col += 1
        # a01*b11
        U[u01][col] += 1; V[v11][col] += 1; W[w01][col] += 1; col += 1
        # a01*b12
        U[u01][col] += 1; V[v12][col] += 1; W[w02][col] += 1; col += 1
        # a10*b00
        U[u10][col] += 1; V[v00][col] += 1; W[w10][col] += 1; col += 1
        # a10*b01
        U[u10][col] += 1; V[v01][col] += 1; W[w11][col] += 1; col += 1
        # a10*b02
        U[u10][col] += 1; V[v02][col] += 1; W[w12][col] += 1; col += 1
        # a11*b10
        U[u11][col] += 1; V[v10][col] += 1; W[w10][col] += 1; col += 1
        # a11*b11
        U[u11][col] += 1; V[v11][col] += 1; W[w11][col] += 1; col += 1
        # a11*b12
        U[u11][col] += 1; V[v12][col] += 1; W[w12][col] += 1; col += 1
    
    # Now let's map indices:
    # A[i,j]: i in {0,1}, j in {0,1,2,3}
    #   A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    #   A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    # B[j,k]: j in {0,1,2,3}, k in {0,1,2,3,4}
    #   B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3], B[0,4]=V[4]
    #   B[1,0]=V[5], B[1,1]=V[6], B[1,2]=V[7], B[1,3]=V[8], B[1,4]=V[9]
    #   B[2,0]=V[10], B[2,1]=V[11], B[2,2]=V[12], B[2,3]=V[13], B[2,4]=V[14]
    #   B[3,0]=V[15], B[3,1]=V[16], B[3,2]=V[17], B[3,3]=V[18], B[3,4]=V[19]
    # C[i,k]: i in {0,1}, k in {0,1,2,3,4}
    #   W[k*n+i] = W[k*2+i]
    #   C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3], C[0,4]=W[4]
    #   C[1,0]=W[2], wait no. W[k*n+i], n=2
    #   W[0*2+0]=W[0]=C[0,0], W[1*2+0]=W[2]=C[0,1], W[2*2+0]=W[4]=C[0,2]
    #   W[3*2+0]=W[6]=C[0,3], W[4*2+0]=W[8]=C[0,4]
    #   W[0*2+1]=W[1]=C[1,0], W[1*2+1]=W[3]=C[1,1], W[2*2+1]=W[5]=C[1,2]
    #   W[3*2+1]=W[7]=C[1,3], W[4*2+1]=W[9]=C[1,4]
    
    # A1 = A[:,0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[U[0],U[1]],[U[4],U[5]]]
    # A2 = A[:,2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] = [[U[2],U[3]],[U[6],U[7]]]
    # B1 = B[0:2,:] = [[B[0,0],B[0,1],B[0,2],B[0,3],B[0,4]],[B[1,0],B[1,1],B[1,2],B[1,3],B[1,4]]]
    #   B1a = B1[:,0:2] = [[V[0],V[1]],[V[5],V[6]]]
    #   B1b = B1[:,2:5] = [[V[2],V[3],V[4]],[V[7],V[8],V[9]]]
    # B2 = B[2:4,:] = [[B[2,0],B[2,1],B[2,2],B[2,3],B[2,4]],[B[3,0],B[3,1],B[3,2],B[3,3],B[3,4]]]
    #   B2a = B2[:,0:2] = [[V[10],V[11]],[V[15],V[16]]]
    #   B2b = B2[:,2:5] = [[V[12],V[13],V[14]],[V[17],V[18],V[19]]]
    
    # C[:,0:2] = A1@B1a + A2@B2a
    #   C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    #   Wait, let me recheck: W[k*n+i] where n=2
    #   C[0,0]: k=0,i=0 -> W[0]
    #   C[0,1]: k=1,i=0 -> W[2]
    #   C[1,0]: k=0,i=1 -> W[1]
    #   C[1,1]: k=1,i=1 -> W[3]
    
    # Strassen for A1@B1a:
    # A1 = [[U[0],U[1]],[U[4],U[5]]], B1a = [[V[0],V[1]],[V[5],V[6]]]
    # C1a = A1@B1a, C1a[0,0]->W[0], C1a[0,1]->W[2], C1a[1,0]->W[1], C1a[1,1]->W[3]
    strassen_2x2(U[0], U[1], U[4], U[5], V[0], V[1], V[5], V[6], W[0], W[2], W[1], W[3])
    
    # Strassen for A2@B2a:
    # A2 = [[U[2],U[3]],[U[6],U[7]]], B2a = [[V[10],V[11]],[V[15],V[16]]]
    # C2a[0,0]->W[0], C2a[0,1]->W[2], C2a[1,0]->W[1], C2a[1,1]->W[3]
    strassen_2x2(U[2], U[3], U[6], U[7], V[10], V[11], V[15], V[16], W[0], W[2], W[1], W[3])
    
    # C[:,2:5] = A1@B1b + A2@B2b
    #   C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8]
    #   C[1,2]=W[5], C[1,3]=W[7], C[1,4]=W[9]
    
    # Schoolbook for A1@B1b:
    # A1 = [[U[0],U[1]],[U[4],U[5]]], B1b = [[V[2],V[3],V[4]],[V[7],V[8],V[9]]]
    # C1b[0,0]->W[4], C1b[0,1]->W[6], C1b[0,2]->W[8]
    # C1b[1,0]->W[5], C1b[1,1]->W[7], C1b[1,2]->W[9]
    schoolbook_2x2x2x3(U[0], U[1], U[4], U[5], V[2], V[3], V[4], V[7], V[8], V[9], W[4], W[6], W[8], W[5], W[7], W[9])
    
    # Schoolbook for A2@B2b:
    # A2 = [[U[2],U[3]],[U[6],U[7]]], B2b = [[V[12],V[13],V[14]],[V[17],V[18],V[19]]]
    # C2b[0,0]->W[4], C2b[0,1]->W[6], C2b[0,2]->W[8]
    # C2b[1,0]->W[5], C2b[1,1]->W[7], C2b[1,2]->W[9]
    schoolbook_2x2x2x3(U[2], U[3], U[6], U[7], V[12], V[13], V[14], V[17], V[18], V[19], W[4], W[6], W[8], W[5], W[7], W[9])
    
    # Total: 7 + 7 + 12 + 12 = 38? Wait, schoolbook_2x2x2x3 does 12 mults, not 6!
    # Let me recount: 2x2 * 2x3 has 2*2*3 = 12 terms in schoolbook.
    # So total = 7 + 7 + 12 + 12 = 38. That's not 26.
    
    # I made an error earlier. 2x2 * 2x3 schoolbook is 12, not 6.
    # Let me redo: total = 7 + 7 + 12 + 12 = 38.
    # That's still better than 40!
    
    # But wait, I set rank=26 but I'm using 38 columns. Let me fix.
    
    # Actually, let me recount the schoolbook function - it does 12 additions.
    # So I need rank = 7+7+12+12 = 38.
    
    # Let me rebuild with correct rank.
    rank = 38
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    
    # Redo everything
    strassen_2x2(U[0], U[1], U[4], U[5], V[0], V[1], V[5], V[6], W[0], W[2], W[1], W[3])
    strassen_2x2(U[2], U[3], U[6], U[7], V[10], V[11], V[15], V[16], W[0], W[2], W[1], W[3])
    schoolbook_2x2x2x3(U[0], U[1], U[4], U[5], V[2], V[3], V[4], V[7], V[8], V[9], W[4], W[6], W[8], W[5], W[7], W[9])
    schoolbook_2x2x2x3(U[2], U[3], U[6], U[7], V[12], V[13], V[14], V[17], V[18], V[19], W[4], W[6], W[8], W[5], W[7], W[9])
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

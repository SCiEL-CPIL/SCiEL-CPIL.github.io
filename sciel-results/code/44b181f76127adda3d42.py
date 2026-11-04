import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll use a Strassen-based approach on the 2x4 by 4x8 product.
    # Split A into left (cols 0,1) and right (cols 2,3): A_L, A_R each 2x2
    # Split B rows into top (rows 0,1) and bottom (rows 2,3): B_T, B_B each 2x4
    # C = A_L @ B_T + A_R @ B_B
    # For each 2x2 by 2x4 block multiplication, we can further decompose.
    # Actually, let's think of it as: for each of the 4 column-pairs of B,
    # we have a 2x2 by 2x2 multiplication (since B is 4x8, each 2x2 block of B).
    # Wait, B is 4x8. Let's split B into 2x2 blocks: B has 2 row-blocks (top/bottom) 
    # and 4 col-blocks. So B = [B_{00} B_{01} B_{02} B_{03}; B_{10} B_{11} B_{12} B_{13}]
    # where each B_{ij} is 2x2.
    # A = [A_L A_R] where A_L is 2x2, A_R is 2x2.
    # C = A_L @ [B_{00} B_{01} B_{02} B_{03}] + A_R @ [B_{10} B_{11} B_{12} B_{13}]
    # C is 2x8.
    # For each column-block c (0..3): C[:, 2c:2c+2] = A_L @ B_{0c} + A_R @ B_{1c}
    # Each term A_L @ B_{0c} is 2x2 by 2x2, which can be done with Strassen (7 mults).
    # Each term A_R @ B_{1c} is 2x2 by 2x2, which can be done with Strassen (7 mults).
    # Total: 4 col-blocks * 2 (for A_L and A_R) * 7 = 56 multiplications.
    
    # Strassen for 2x2 by 2x2:
    # X = [[a, b], [c, d]] @ Y = [[e, f], [g, h]]
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-f)
    # m4 = (a+c)h
    # m5 = (b-d)(e+g)
    # m6 = (b-c)(f+h)
    # Z00 = m0 + m3 - m4 + m6
    # Z01 = m2 + m4
    # Z10 = m1 + m5
    # Z11 = m0 - m1 + m2 + m5
    
    # We need to build U, V, W such that:
    # C[i,k] = sum_r U[i,j,r] * V[j,l,r] * W[l,i,r]  (using the trace convention)
    # Wait, the interface says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And W[k*n+i, r] is the convention.
    # So C[i,k] = sum_j sum_l A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r]) * (sum_l V[j*p+l, r]) * ... 
    # Actually, the standard tensor decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] is not quite right. Let me re-read.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The decomposition is: sum_{i,j,k} A[i,j] B[j,k] e_{i,k} = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_l V[j*p+l, r] B[j,l]) * ... 
    # No, it's a 3-tensor: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T_{i,j,k} = 1 if we're looking at the (i,j,k) entry of the multiplication tensor.
    # So for each Strassen multiplication, we define one column in U, V, W.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    for c in range(4):  # 4 column-blocks of B
        # A_L @ B_{0c}: A_L is 2x2, B_{0c} is 2x2
        # A_L entries: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in U (n*m=8)
        # B_{0c} entries: B[0,2c], B[0,2c+1], B[1,2c], B[1,2c+1] -> indices 2c, 2c+1, 4+2c, 4+2c+1 in V (m*p=32)
        # Result contributes to C[0,2c], C[0,2c+1], C[1,2c], C[1,2c+1]
        
        # Strassen for A_L @ B_{0c}
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,2c], f=B[0,2c+1], g=B[1,2c], h=B[1,2c+1]
        
        # m0 = (a+d)(e+h)
        U[0, col] = 1; U[3, col] = 1  # a, d
        V[2*c, col] = 1; V[2*c+1, col] = 1  # e, f? No, e+h means B[0,2c]+B[1,2c+1]
        # Wait, e = B[0,2c], h = B[1,2c+1]. So e+h = B[0,2c] + B[1,2c+1]
        V[2*c, col] = 1; V[4+2*c+1, col] = 1  # e, h
        # Result Z00 contribution: +1 to C[0,2c], +1 to C[1,2c+1]... no
        # Z00 = m0 + m3 - m4 + m6 -> C[0, 2c]
        # Z01 = m2 + m4 -> C[0, 2c+1]
        # Z10 = m1 + m5 -> C[1, 2c]
        # Z11 = m0 - m1 + m2 + m5 -> C[1, 2c+1]
        # W[k*n+i, r]: for C[0,2c], i=0,k=2c -> W[2c*2+0, col] = W[4c, col]
        # for C[0,2c+1], i=0,k=2c+1 -> W[(2c+1)*2+0, col] = W[4c+2, col]
        # for C[1,2c], i=1,k=2c -> W[2c*2+1, col] = W[4c+1, col]
        # for C[1,2c+1], i=1,k=2c+1 -> W[(2c+1)*2+1, col] = W[4c+3, col]
        W[4*c, col] += 1  # Z00: +m0
        W[4*c+2, col] += 0  # Z01: m0 not in Z01
        W[4*c+1, col] += 0  # Z10: m0 not in Z10
        W[4*c+3, col] += 1  # Z11: +m0
        col += 1
        
        # m1 = (c+d)e
        U[2, col] = 1; U[3, col] = 1  # c, d
        V[2*c, col] = 1  # e
        W[4*c+1, col] += 1  # Z10: +m1
        W[4*c+3, col] += -1  # Z11: -m1
        col += 1
        
        # m2 = a(f-h)
        U[0, col] = 1  # a
        V[2*c+1, col] = 1; V[4+2*c, col] = -1  # f, -g (h=B[1,2c+1], g=B[1,2c])
        # Wait, f-h = B[0,2c+1] - B[1,2c+1]? No, h = B[1,2c+1], f = B[0,2c+1]
        # So f-h = B[0,2c+1] - B[1,2c+1]
        V[2*c+1, col] = 1; V[4+2*c+1, col] = -1
        W[4*c+2, col] += 1  # Z01: +m2
        W[4*c+3, col] += 1  # Z11: +m2
        col += 1
        
        # m3 = d(g-f)
        U[3, col] = 1  # d
        V[4+2*c, col] = 1; V[2*c+1, col] = -1  # g, -f
        W[4*c, col] += 1  # Z00: +m3
        col += 1
        
        # m4 = (a+c)h
        U[0, col] = 1; U[2, col] = 1  # a, c
        V[4+2*c+1, col] = 1  # h
        W[4*c, col] += -1  # Z00: -m4
        W[4*c+2, col] += 1  # Z01: +m4
        col += 1
        
        # m5 = (b-d)(e+g)
        U[1, col] = 1; U[3, col] = -1  # b, -d
        V[2*c, col] = 1; V[4+2*c, col] = 1  # e, g
        W[4*c+1, col] += 1  # Z10: +m5
        W[4*c+3, col] += 1  # Z11: +m5
        col += 1
        
        # m6 = (b-c)(f+h)
        U[1, col] = 1; U[2, col] = -1  # b, -c
        V[2*c+1, col] = 1; V[4+2*c+1, col] = 1  # f, h
        W[4*c, col] += 1  # Z00: +m6
        col += 1
        
        # A_R @ B_{1c}: A_R is 2x2, B_{1c} is 2x2
        # A_R entries: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 4,5,6,7 in U
        # B_{1c} entries: B[2,2c], B[2,2c+1], B[3,2c], B[3,2c+1] -> indices 8+2c, 8+2c+1, 12+2c, 12+2c+1 in V
        
        # m0 = (a+d)(e+h)
        U[4, col] = 1; U[7, col] = 1  # a, d (of A_R)
        V[8+2*c, col] = 1; V[12+2*c+1, col] = 1  # e, h (of B_{1c})
        W[4*c, col] += 1  # Z00: +m0
        W[4*c+3, col] += 1  # Z11: +m0
        col += 1
        
        # m1 = (c+d)e
        U[6, col] = 1; U[7, col] = 1  # c, d
        V[8+2*c, col] = 1  # e
        W[4*c+1, col] += 1  # Z10: +m1
        W[4*c+3, col] += -1  # Z11: -m1
        col += 1
        
        # m2 = a(f-h)
        U[4, col] = 1  # a
        V[8+2*c+1, col] = 1; V[12+2*c+1, col] = -1  # f, -h
        W[4*c+2, col] += 1  # Z01: +m2
        W[4*c+3, col] += 1  # Z11: +m2
        col += 1
        
        # m3 = d(g-f)
        U[7, col] = 1  # d
        V[12+2*c, col] = 1; V[8+2*c+1, col] = -1  # g, -f
        W[4*c, col] += 1  # Z00: +m3
        col += 1
        
        # m4 = (a+c)h
        U[4, col] = 1; U[6, col] = 1  # a, c
        V[12+2*c+1, col] = 1  # h
        W[4*c, col] += -1  # Z00: -m4
        W[4*c+2, col] += 1  # Z01: +m4
        col += 1
        
        # m5 = (b-d)(e+g)
        U[5, col] = 1; U[7, col] = -1  # b, -d
        V[8+2*c, col] = 1; V[12+2*c, col] = 1  # e, g
        W[4*c+1, col] += 1  # Z10: +m5
        W[4*c+3, col] += 1  # Z11: +m5
        col += 1
        
        # m6 = (b-c)(f+h)
        U[5, col] = 1; U[6, col] = -1  # b, -c
        V[8+2*c+1, col] = 1; V[12+2*c+1, col] = 1  # f, h
        W[4*c, col] += 1  # Z00: +m6
        col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into blocks
    # j splits: [0,1], [2,3], [4]
    # k splits: [0,1], [2,3], [4]
    
    # List of blocks: (j_start, j_end, k_start, k_end)
    # j_end, k_end are exclusive
    
    blocks = [
        (0, 2, 0, 2), # 2x2x2
        (0, 2, 2, 4), # 2x2x2
        (0, 2, 4, 5), # 2x2x1
        (2, 4, 0, 2), # 2x2x2
        (2, 4, 2, 4), # 2x2x2
        (2, 4, 4, 5), # 2x2x1
        (4, 5, 0, 2), # 2x1x2
        (4, 5, 2, 4), # 2x1x2
        (4, 5, 4, 5), # 2x1x1
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(j_s, j_e, k_s, k_e):
        nj = j_e - j_s
        nk = k_e - k_s
        
        if nj == 2 and nk == 2:
            # Strassen 2x2x2
            # Local indices: i in 0..1, j in 0..1, k in 0..1
            # M1: (a+d)(e+h)
            # U: A_00, A_11
            # V: B_00, B_11
            # W: C_00, C_11
            # Coeffs in W for M1: C_00 += 1, C_11 += 1
            
            # We need to map local (i,j,k) to global.
            # Global A index: i*m + (j_s + j_local)
            # Global B index: (j_s + j_local)*p + (k_s + k_local)
            # Global C index: (k_s + k_local)*n + i
            
            # Strassen Terms:
            # M1: (A_00+A_11) * (B_00+B_11) -> C_00 += 1, C_11 += 1
            # M2: (A_10+A_11) * B_00 -> C_10 += 1, C_11 += 1
            # M3: A_00 * (B_01-B_11) -> C_01 += 1
            # M4: A_11 * (B_10-B_00) -> C_10 += 1, C_11 -= 1
            # M5: (A_00+A_01) * B_11 -> C_00 -= 1, C_01 += 1
            # M6: (A_10-A_00) * (B_00+B_01) -> C_11 += 1
            # M7: (A_01-A_11) * (B_10+B_11) -> C_00 += 1, C_11 -= 1
            
            # Note: A indices are (i, j). B indices are (j, k).
            # A_00 is i=0, j=0. A_11 is i=1, j=1.
            
            # Term 1
            U1 = [0]*n; U1[0] += 1; U1[1] += 1
            # Wait, U is vector of size n*m? No, U is shape (n*m, rank).
            # So U is a column vector of length n*m.
            
            # Let's define the columns for U, V, W.
            
            # T1
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            u[1*m + j_s + 1] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            v[(j_s + 1)*p + k_s + 1] += 1
            w[(k_s + 0)*n + 0] += 1
            w[(k_s + 1)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T2: (A_10+A_11) * B_00
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            u[1*m + j_s + 1] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 1] += 1
            w[(k_s + 1)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T3: A_00 * (B_01-B_11)
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 1] += 1
            v[(j_s + 1)*p + k_s + 1] -= 1
            w[(k_s + 1)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T4: A_11 * (B_10-B_00)
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 1] += 1
            v[(j_s + 1)*p + k_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] -= 1
            w[(k_s + 0)*n + 1] += 1
            w[(k_s + 1)*n + 1] -= 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T5: (A_00+A_01) * B_11
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            u[0*m + j_s + 1] += 1
            v[(j_s + 1)*p + k_s + 1] += 1
            w[(k_s + 0)*n + 0] -= 1
            w[(k_s + 1)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T6: (A_10-A_00) * (B_00+B_01)
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            u[0*m + j_s + 0] -= 1
            v[(j_s + 0)*p + k_s + 0] += 1
            v[(j_s + 0)*p + k_s + 1] += 1
            w[(k_s + 1)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            # T7: (A_01-A_11) * (B_10+B_11)
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 1] += 1
            u[1*m + j_s + 1] -= 1
            v[(j_s + 1)*p + k_s + 0] += 1
            v[(j_s + 1)*p + k_s + 1] += 1
            w[(k_s + 0)*n + 0] += 1
            w[(k_s + 1)*n + 1] -= 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
        elif nj == 2 and nk == 1:
            # 2x2x1. Rank 4.
            # C_00 = A_00 B_0 + A_01 B_1
            # C_10 = A_10 B_0 + A_11 B_1
            # T1: A_00 B_0 -> C_00
            # T2: A_01 B_1 -> C_00
            # T3: A_10 B_0 -> C_10
            # T4: A_11 B_1 -> C_10
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 1] += 1
            v[(j_s + 1)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 1] += 1
            v[(j_s + 1)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
        elif nj == 1 and nk == 2:
            # 2x1x2. Rank 2.
            # C_00 = A_0 B_0
            # C_01 = A_0 B_1
            # C_10 = A_1 B_0
            # C_11 = A_1 B_1
            # T1: A_0 (B_0 + B_1) ? No.
            # T1: A_0 B_0 -> C_00
            # T2: A_0 B_1 -> C_01
            # T3: A_1 B_0 -> C_10
            # T4: A_1 B_1 -> C_11
            # Can we do rank 2?
            # C_0 = A_0 (B_0, B_1). C_1 = A_1 (B_0, B_1).
            # M1 = A_0 (B_0 + B_1) -> C_00 += 1, C_01 += 1
            # M2 = A_0 (B_0 - B_1) -> C_00 += 1, C_01 -= 1
            # M3 = A_1 (B_0 + B_1) -> C_10 += 1, C_11 += 1
            # M4 = A_1 (B_0 - B_1) -> C_10 += 1, C_11 -= 1
            # This is rank 4.
            # Is rank 2 possible?
            # Tensor 2x1x2.
            # M_20 = A_0 B_0
            # M_21 = A_0 B_1
            # M_10 = A_1 B_0
            # M_11 = A_1 B_1
            # These are 4 independent terms.
            # So rank is 4?
            # Wait, earlier I said 2. Let's re-evaluate.
            # $y_1 = x_1 z_1$, $y_2 = x_1 z_2$, $y_3 = x_2 z_1$, $y_4 = x_2 z_2$.
            # No, $C$ is $2 \times 2$.
            # $C_{00} = A_0 B_0$
            # $C_{01} = A_0 B_1$
            # $C_{10} = A_1 B_0$
            # $C_{11} = A_1 B_1$
            # This is just element-wise multiplication of two vectors?
            # No, $A$ is $2 \times 1$, $B$ is $1 \times 2$.
            # $C = A B^T$.
            # This is an outer product.
            # Rank of outer product tensor is 1?
            # $C_{ik} = A_i B_k$.
            # $T_{i,j,k} = A_i B_k$ where $j$ is fixed (0).
            # $T_{i,0,k} = 1$.
            # This is a rank 1 tensor?
            # $T = u \otimes v \otimes w$?
            # $T_{i,k} = u_i v_k$.
            # Yes, rank 1.
            # So $2 \times 1 \times 2$ has rank 1?
            # $C_{00} = A_0 B_0$
            # $C_{01} = A_0 B_1$
            # $C_{10} = A_1 B_0$
            # $C_{11} = A_1 B_1$
            # We can compute $M_1 = A_0 (B_0 + B_1)$? No.
            # We need 4 outputs.
            # If we use 1 multiplication $M = A_0 B_0$? No.
            # The tensor is $T_{i,j,k}$.
            # $T_{0,0,0}=1, T_{0,0,1}=1, T_{1,0,0}=1, T_{1,0,1}=1$.
            # This matrix (flattened) has rank 1?
            # $T = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ (for i,k).
            # This is rank 1.
            # So 1 multiplication?
            # $M = (A_0 + A_1) (B_0 + B_1)$?
            # $M = A_0 B_0 + A_0 B_1 + A_1 B_0 + A_1 B_1$.
            # We need to extract individual terms.
            # We cannot extract $A_0 B_0$ from the sum without more multiplications.
            # So rank is not 1.
            # Rank is 4?
            # Yes, $2 \times 1 \times 2$ is equivalent to $1 \times 2 \times 2$?
            # No.
            # It is the tensor product of a $2 \times 1$ and $1 \times 2$.
            # The multiplication tensor for outer product $A B^T$ has rank equal to the number of non-zero entries?
            # No, it's a rank 1 tensor in the algebraic geometry sense?
            # No, the rank of the tensor $T_{ijk}$ is the min number of rank 1 tensors summing to it.
            # $T_{000}=1, T_{001}=1, T_{100}=1, T_{101}=1$.
            # Can we write as sum of rank 1?
            # $u_1 \otimes v_1 \otimes w_1 + \dots$
            # If we use $M_1 = A_0 B_0$, $M_2 = A_0 B_1$, $M_3 = A_1 B_0$, $M_4 = A_1 B_1$. Rank 4.
            # Can we do better?
            # $M_1 = A_0 (B_0+B_1)$, $M_2 = A_1 (B_0+B_1)$.
            # $C_{00} = (M_1 + M_{1'})/2$?
            # No, we don't have $A_0 B_0$ separately.
            # So rank is 4.
            
            # So 2x1x2 is rank 4.
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 1] += 1
            w[(k_s + 1)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 1] += 1
            w[(k_s + 1)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)

        elif nj == 1 and nk == 1:
            # 2x1x1. Rank 2.
            # C_00 = A_0 B_0
            # C_10 = A_1 B_0
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[0*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 0] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[1*m + j_s + 0] += 1
            v[(j_s + 0)*p + k_s + 0] += 1
            w[(k_s + 0)*n + 1] += 1
            U_list.append(u); V_list.append(v); W_list.append(w)

    for b in blocks:
        add_block(*b)
        
    rank = len(U_list)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if U_list[r][i] != 0:
                U[i][r] = U_list[r][i]
        for i in range(m*p):
            if V_list[r][i] != 0:
                V[i][r] = V_list[r][i]
        for i in range(p*n):
            if W_list[r][i] != 0:
                W[i][r] = W_list[r][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

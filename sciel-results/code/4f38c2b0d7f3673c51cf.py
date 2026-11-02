import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get Strassen decomposition for 2x2x2
    def get_strassen():
        u = [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, -1, 0]
        ]
        v = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, -1, 0],
            [0, 1, -1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, -1]
        ]
        return u, v, w

    # Helper to get trivial decomposition for n x m x 1
    def get_dot(n, m):
        u = []
        v = []
        w = []
        for i in range(n):
            for j in range(m):
                ui = [0] * n
                ui[i] = 1
                vj = [0] * m
                vj[j] = 1
                w0 = [1]
                u.append(ui)
                v.append(vj)
                w.append(w0)
        return u, v, w

    # Helper to get trivial decomposition for 1 x m x p
    def get_row(m, p):
        u = []
        v = []
        w = []
        for j in range(m):
            for k in range(p):
                u0 = [1]
                vj = [0] * m
                vj[j] = 1
                wk = [0] * p
                wk[k] = 1
                u.append(u0)
                v.append(vj)
                w.append(wk)
        return u, v, w

    # Helper to get trivial decomposition for n x 1 x p
    def get_col(n, p):
        u = []
        v = []
        w = []
        for i in range(n):
            for k in range(p):
                ui = [0] * n
                ui[i] = 1
                v0 = [1]
                wk = [0] * p
                wk[k] = 1
                u.append(ui)
                v.append(v0)
                w.append(wk)
        return u, v, w

    def get_decomp(n, m, p):
        if n == 1:
            return get_row(m, p)
        if m == 1:
            return get_col(n, p)
        if p == 1:
            return get_dot(n, m)
        if n == 2 and m == 2 and p == 2:
            return get_strassen()
        
        # Split the largest dimension
        if n >= m and n >= p:
            n1 = n // 2
            n2 = n - n1
            u1, v1, w1 = get_decomp(n1, m, p)
            u2, v2, w2 = get_decomp(n2, m, p)
            # Pad u vectors
            u_new = []
            for ui in u1:
                u_new.append(ui + [0]*n2)
            for ui in u2:
                u_new.append([0]*n1 + ui)
            v_new = v1 + v2
            w_new = w1 + w2
            return u_new, v_new, w_new
        elif m >= n and m >= p:
            m1 = m // 2
            m2 = m - m1
            u1, v1, w1 = get_decomp(n, m1, p)
            u2, v2, w2 = get_decomp(n, m2, p)
            u_new = u1 + u2
            v_new = []
            for vj in v1:
                v_new.append(vj + [0]*m2)
            for vj in v2:
                v_new.append([0]*m1 + vj)
            w_new = w1 + w2
            return u_new, v_new, w_new
        else:
            p1 = p // 2
            p2 = p - p1
            u1, v1, w1 = get_decomp(n, m, p1)
            u2, v2, w2 = get_decomp(n, m, p2)
            u_new = u1 + u2
            v_new = v1 + v2
            w_new = []
            for wk in w1:
                w_new.append(wk + [0]*p2)
            for wk in w2:
                w_new.append([0]*p1 + wk)
            return u_new, v_new, w_new

    u, v, w = get_decomp(n, m, p)
    rank = len(u)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # u is in R^n, but U is n*m. 
        # The tensor is in R^n x R^m x R^p.
        # U index is i*m + j? No, U is (n*m, rank).
        # The first dimension of the tensor is n.
        # Wait, the problem says U is (n*m, rank).
        # This implies the first dimension of the tensor is n*m?
        # No, standard matrix multiplication tensor is n x m x p.
        # But the interface says U is (n*m, rank).
        # This suggests the tensor is (n*m) x (m*p) x (p*n)?
        # Let's re-read the task.
        # "Matrix multiplication 2,4,7"
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # This is the standard Kronecker product layout.
        # The tensor is $T \in \mathbb{R}^{nm} \otimes \mathbb{R}^{mp} \otimes \mathbb{R}^{pn}$.
        # Entry $T_{i,j,k}$ where $i \in [nm], j \in [mp], k \in [pn]$.
        # $i = i_A m + j_A$. $j = j_B p + k_B$. $k = k_C n + i_C$.
        # The term is $A_{i_A, j_A} B_{j_B, k_B} \to C_{i_C, k_C}$.
        # This requires $j_A = j_B$ and $i_A = i_C$.
        # So $T_{i,j,k} = 1$ if $j_A = j_B$ and $i_A = i_C$.
        # My previous analysis used $n, m, p$ as the dimensions of A, B, C.
        # The vectors $u, v, w$ I generated are in $\mathbb{R}^n, \mathbb{R}^m, \mathbb{R}^p$.
        # I need to map them to the Kronecker layout.
        
        # u is in R^n. Map to R^(n*m).
        # u[i] corresponds to row i of A.
        # In Kronecker layout, row i of A spans indices i*m to i*m + m - 1.
        # But u[i] is a scalar coefficient for the whole row?
        # No, the tensor entry is $u_{i_A} v_{j_B} w_{k_C}$.
        # The Kronecker tensor entry is $T_{i, j, k}$.
        # $T_{i, j, k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$.
        # We want $T_{i, j, k} = 1$ if $j_A = j_B, i_A = i_C$, else 0.
        # My $u, v, w$ are for the $n \times m \times p$ tensor.
        # The $n \times m \times p$ tensor has entries $T'_{i_A, j_A, k_B} = \delta_{j_A, j_B} \delta_{i_A, i_C}$?
        # No, the standard tensor is $T'_{i_A, j, k_B} = 1$ if $j=j_B$ and $i_A=i_C$?
        # Actually, the tensor for $A B$ is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # No, it is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$ where the term is $A_{i,j} B_{j,k}$.
        # So $T'_{i,j,k} = 1$ for all $i,j,k$.
        # My $u, v, w$ decompose this all-ones tensor.
        # So $u$ is in $\mathbb{R}^n$, $v$ is in $\mathbb{R}^m$, $w$ is in $\mathbb{R}^p$.
        # Now map to Kronecker.
        # $U$ is $n \times m$. $u$ is $n$.
        # $U_{i_A m + j_A, r} = u_{i_A, r} \delta_{j_A, 0}$? No.
        # The tensor is $\sum_r u_r \otimes v_r \otimes w_r$.
        # In Kronecker form, the basis vectors are $e_{i_A, j_A} \otimes e_{j_B, k_B} \otimes e_{k_C, i_C}$.
        # The term $u_{i_A} v_{j_A} w_{k_B}$?
        # No, $v$ is in $\mathbb{R}^m$, so $v_{j}$.
        # The Kronecker index for $v$ is $j_B p + k_B$? No, $v$ corresponds to the $m$ dimension.
        # The $m$ dimension is the inner product dimension.
        # In Kronecker layout, the second factor is $m \times p$.
        # So $v$ should be in $\mathbb{R}^{mp}$?
        # No, the tensor is $n \times m \times p$.
        # The Kronecker layout groups $n$ and $m$?
        # "U ... (n*m, rank)". This groups the first two dimensions?
        # No, $A$ is $n \times m$. So $U$ is $n \times m$.
        # $B$ is $m \times p$. So $V$ is $m \times p$.
        # $C$ is $p \times n$. So $W$ is $p \times n$.
        # So the tensor is in $\mathbb{R}^{nm} \otimes \mathbb{R}^{mp} \otimes \mathbb{R}^{pn}$.
        # The entry $T_{i, j, k}$ is 1 if the term $A_{i_A, j_A} B_{j_B, k_B}$ contributes to $C_{i_C, k_C}$.
        # This requires $i_A = i_C$ and $j_A = j_B$.
        # So $T_{i, j, k} = 1$ if $i_A = i_C$ and $j_A = j_B$.
        # My $u, v, w$ are for the $n \times m \times p$ tensor where $T'_{i, j, k} = 1$.
        # How to map?
        # $u$ is in $\mathbb{R}^n$. $U$ is in $\mathbb{R}^{nm}$.
        # $U_{i_A m + j_A, r} = u_{i_A, r}$?
        # If I set $U_{i_A m + j_A, r} = u_{i_A, r} \delta_{j_A, 0}$, it won't work.
        # The correct mapping is:
        # $U_{i_A m + j_A, r} = u_{i_A, r} \delta_{j_A, j}$? No.
        # The term in the sum is $u_{i_A} v_j w_{k_B}$.
        # This contributes to $C_{i_A, k_B}$.
        # In Kronecker layout, the index for $A$ is $i_A m + j$.
        # The index for $B$ is $j p + k_B$.
        # The index for $C$ is $k_B n + i_A$.
        # So $U_{i_A m + j, r} = u_{i_A, r}$.
        # $V_{j p + k_B, r} = v_{j, r} w_{k_B, r}$? No.
        # $v$ is in $\mathbb{R}^m$, $w$ is in $\mathbb{R}^p$.
        # $V_{j p + k_B, r} = v_{j, r}$.
        # $W_{k_B n + i_A, r} = w_{k_B, r}$.
        # Let's check the product:
        # $\sum_r U_{i_A m + j, r} V_{j p + k_B, r} W_{k_B n + i_A, r}$
        # $= \sum_r u_{i_A, r} v_{j, r} w_{k_B, r}$.
        # This matches the tensor $T'_{i_A, j, k_B}$.
        # And $T'_{i_A, j, k_B} = 1$ for all $i_A, j, k_B$.
        # So this works.
        
        # So:
        # $U_{i_A m + j, r} = u_{i_A, r}$ for all $j$.
        # $V_{j p + k_B, r} = v_{j, r}$ for all $k_B$.
        # $W_{k_B n + i_A, r} = w_{k_B, r}$ for all $i_A$.
        
        for i_A in range(n):
            for j in range(m):
                U[i_A * m + j][r] = u[r][i_A]
        
        for j in range(m):
            for k_B in range(p):
                V[j * p + k_B][r] = v[r][j]
        
        for k_B in range(p):
            for i_A in range(n):
                W[k_B * n + i_A][r] = w[r][k_B]

    return {"U": U, "V": V, "W": W}

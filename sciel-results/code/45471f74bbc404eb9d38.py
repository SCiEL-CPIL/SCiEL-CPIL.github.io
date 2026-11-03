import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank estimate: 124
    # We will build the factors incrementally
    
    # Helper to create zero matrices
    def zero_matrix(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # We don't know the exact rank upfront if we want to be safe, 
    # but we calculated 124. 
    # Let's allocate for 124. If it fails, we can fallback to 125.
    # But 124 is the target.
    
    # Let's define the sub-decompositions
    
    # 1. Strassen 2x2
    # U: 4x7, V: 4x7, W: 4x7 (local)
    # Indices: U[i, c], V[j, c], W[k, c] where C[i,k] = sum U[i,c]V[j,c]W[k,c]?
    # No, standard tensor decomposition: T_{ijk} = sum U_{ic} V_{jc} W_{kc}.
    # For matrix mult A*B=C: C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} is 1 if it's part of the sum.
    # U_{ic} multiplies A_{ij}? No.
    # The decomposition is: A_{ij} B_{jk} = sum_c U_{i,c} V_{j,c} W_{k,c} ?? No.
    # The tensor is $T_{ijk} = \delta_{?}$?
    # The matrix multiplication tensor is $M_{ijk} = 1$ if indices match?
    # Actually, the bilinear form is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This can be written as $\sum_c (\sum_i A_{ij} U_{ij,c}) (\sum_k B_{jk} V_{jk,c}) W_{ik,c}$?
    # No, the standard form is:
    # $C_{ik} = \sum_c U_{i,c} V_{k,c} (\sum_j A_{ij} W_{j,c})$? No.
    #
    # Let's stick to the seed program's convention:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ??
    # No, the evaluator checks:
    # sum_c U[i*m+j, c] * V[j*p+k, c] == C[i,k] ? No.
    #
    # The tensor decomposition for matrix multiplication $A B = C$ is:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a bilinear map.
    # We want to represent this as $\sum_c U_c \otimes V_c \otimes W_c$.
    # Where $U_c$ is a vector of length $n*m$ (flattened A), $V_c$ length $m*p$ (flattened B), $W_c$ length $p*n$ (flattened C).
    # The condition is: For all A, B, 
    # $\langle A, U_c \rangle \langle B, V_c \rangle$ sums to $\langle C, W_c \rangle$?
    # No.
    # The identity is:
    # $\sum_c (\sum_{i,j} A_{ij} U_{i,j,c}) (\sum_{j,k} B_{jk} V_{j,k,c}) = \sum_{i,k} C_{ik} W_{i,k,c}$
    # This must hold for all A, B such that C=AB.
    # This implies that the tensor $T_{ijk}$ (where $C_{ik} = \sum_j A_{ij} B_{jk}$) is decomposed.
    # The entries of the tensor are $T_{i,j,k} = 1$ if the term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # So $T_{i,j,k} = 1$ for all valid i,j,k.
    # The decomposition is:
    # $1 = \sum_c U_{i,j,c} V_{j,k,c} W_{i,k,c}$? No.
    # The bilinear form is linear in A and B.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # We approximate $A_{ij} B_{jk}$ by $\sum_c U_{i,j,c} V_{j,k,c} W_{i,k,c}$?
    # No, the output is C.
    # The correct identity is:
    # $\sum_j A_{ij} B_{jk} = \sum_c (\sum_{j'} A_{i,j'} U_{i,j',c}) (\sum_{j''} B_{j'',k} V_{j'',k,c}) W_{i,k,c}$?
    # This doesn't look right dimensionally or structurally.
    
    # Let's look at the seed program again.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term $A_{ij} B_{jk}$.
    # The sum over columns c gives:
    # $\sum_c U_{ij,c} V_{jk,c} W_{ik,c}$?
    # If we fix i,k, the term in C is $\sum_j A_{ij} B_{jk}$.
    # In the seed, for a specific (i,j,k), there is one column c.
    # U_{ij,c}=1, V_{jk,c}=1, W_{ik,c}=1.
    # So $\sum_c U_{ij,c} V_{jk,c} W_{ik,c} = 1$ for that specific (i,j,k) combination?
    # No, the evaluator likely computes:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # And checks if $\sum_c (\sum_j A_{ij} U_{ij,c}) (\sum_j B_{jk} V_{jk,c}) W_{ik,c}$ matches?
    # No, that would be rank-1 approximation of the whole matrix.
    
    # The standard definition of tensor rank for matrix multiplication:
    # The tensor $M$ has indices $i,j,k$.
    # $M_{ijk} = 1$ if the multiplication $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # Actually, the tensor is $T_{ijk} = \delta_{?}$?
    # The identity is $T_{ijk} = 1$ for all $i,j,k$? No.
    # The bilinear form is $f(A,B) = AB$.
    # $f(A,B)_{ik} = \sum_j A_{ij} B_{jk}$.
    # We want $f(A,B) = \sum_c u_c(A) v_c(B) w_c$?
    # Where $u_c(A)$ is a scalar, $v_c(B)$ is a scalar, and $w_c$ is a matrix?
    # No, $w_c$ must be a matrix such that $\sum_c u_c(A) v_c(B) w_c = AB$.
    # This is equivalent to:
    # $\sum_j A_{ij} B_{jk} = \sum_c u_c(A) v_c(B) W_{ik,c}$.
    # For this to hold for all A, B:
    # $u_c(A)$ must be linear in A, $v_c(B)$ linear in B.
    # $u_c(A) = \sum_{i',j'} A_{i'j'} U_{i'j',c}$.
    # $v_c(B) = \sum_{j',k'} B_{j'k'} V_{j'k',c}$.
    # Then LHS: $\sum_c (\sum_{i',j'} A_{i'j'} U_{i'j',c}) (\sum_{j',k'} B_{j'k'} V_{j'k',c}) W_{ik,c}$.
    # We want this to equal $\sum_j A_{ij} B_{jk}$.
    # Comparing coefficients of $A_{i'j'} B_{j'k'}$:
    # LHS coeff of $A_{i'j'} B_{j'k'}$ is $\sum_c U_{i'j',c} V_{j'k',c} W_{ik,c}$.
    # RHS coeff of $A_{i'j'} B_{j'k'}$ is $1$ if $i'=i, j'=j, k'=k$?
    # Wait, the output index is $i,k$.
    # The term $A_{i'j'} B_{j'k'}$ contributes to $C_{i'k'}$.
    # So we require:
    # $\sum_c U_{i'j',c} V_{j'k',c} W_{i'k',c} = 1$ if $j'=j$? No.
    # The sum is over $j$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is 1.
    # The coefficient of $A_{i'j'} B_{j'k'}$ in $C_{ik}$ is 0 if $i' \ne i$ or $k' \ne k$.
    # So we require:
    # $\sum_c U_{i'j',c} V_{j'k',c} W_{ik,c} = \delta_{i,i'} \delta_{k,k'}$ ?
    # No, the LHS depends on $i,k$ via $W_{ik,c}$.
    # So for a fixed output $(i,k)$, we need:
    # $\sum_c U_{i'j',c} V_{j'k',c} W_{ik,c} = 1$ if $i'=i, k'=k$ and $j'=j$?
    # No, the sum is over $j$.
    # The term $A_{ij} B_{jk}$ appears in the expansion of $C_{ik}$.
    # So for a specific $i,k$, the coefficient of $A_{ij} B_{jk}$ is 1.
    # So $\sum_c U_{ij,c} V_{jk,c} W_{ik,c} = 1$?
    # This must hold for all $i,j,k$?
    # If we set $i=0, k=0$, we need $\sum_c U_{0j,c} V_{j0,c} W_{00,c} = 1$ for all j?
    # And for $i=0, k=1$, we need $\sum_c U_{0j,c} V_{j1,c} W_{01,c} = 0$?
    # This is the condition.
    
    # So the seed program sets:
    # For each (i,j,k):
    # $U_{ij,c}=1, V_{jk,c}=1, W_{ik,c}=1$.
    # Check:
    # For output (i,k), term $A_{ij} B_{jk}$:
    # Sum over c. Only one c has non-zero U and V for this (i,j,k).
    # For that c, $W_{ik,c}=1$.
    # So sum is 1.
    # For other (i',j',k') with $i' \ne i$ or $k' \ne k$:
    # The term $A_{i'j'} B_{j'k'}$ uses $U_{i'j',c'}$ and $V_{j'k',c'}$.
    # These are different columns (since c is unique to i,j,k).
    # So for column c, $U_{i'j',c}=0$ if $i' \ne i$ or $j' \ne j$.
    # So the product is 0.
    # Thus the condition holds.
    
    # So the algorithm is:
    # 1. Define local decompositions for sub-blocks.
    # 2. For each local term (i_local, j_local, k_local) with weight w:
    #    It contributes to global U, V, W.
    #    Global U index: $i_{global} * m + j_{global}$.
    #    Global V index: $j_{global} * p + k_{global}$.
    #    Global W index: $k_{global} * n + i_{global}$.
    
    # We will implement a helper to add a local decomposition to the global ones.
    
    def add_decomposition(global_U, global_V, global_W, rank_offset, 
                         i_offset, j_offset, k_offset,
                         local_U, local_V, local_W,
                         m, p, n):
        # local_U: (m1, r), local_V: (n1, r), local_W: (p1, r)
        # Wait, local dimensions:
        # A block: m1 x n1
        # B block: n1 x p1
        # C block: p1 x m1? No, C is m1 x p1.
        # W index is k*n + i.
        # So local W index is k_local * m1 + i_local.
        
        r = len(local_U[0])
        for c in range(r):
            # U
            for i_loc in range(len(local_U)):
                val_u = local_U[i_loc][c]
                if val_u != 0:
                    i_glob = i_offset + i_loc
                    # j_loc is implicit in the column? No.
                    # local_U is indexed by i_loc?
                    # No, in the seed, U is indexed by i*m+j.
                    # So local_U should be indexed by i_loc * n1 + j_loc.
                    # Let's assume local_U is flattened (m1*n1, r).
                    pass
        
        # Let's re-structure.
        # local_U: list of lists, size (m1*n1, r).
        # local_V: list of lists, size (n1*p1, r).
        # local_W: list of lists, size (p1*m1, r).
        
        for c in range(r):
            # U
            for idx_u in range(len(local_U)):
                val_u = local_U[idx_u][c]
                if val_u != 0:
                    i_loc = idx_u // n1
                    j_loc = idx_u % n1
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    idx_u_glob = i_glob * m + j_glob
                    global_U[idx_u_glob][rank_offset + c] += val_u
            
            # V
            for idx_v in range(len(local_V)):
                val_v = local_V[idx_v][c]
                if val_v != 0:
                    j_loc = idx_v // p1
                    k_loc = idx_v % p1
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    idx_v_glob = j_glob * p + k_glob
                    global_V[idx_v_glob][rank_offset + c] += val_v
            
            # W
            for idx_w in range(len(local_W)):
                val_w = local_W[idx_w][c]
                if val_w != 0:
                    k_loc = idx_w // m1
                    i_loc = idx_w % m1
                    k_glob = k_offset + k_loc
                    i_glob = i_offset + i_loc
                    idx_w_glob = k_glob * n + i_glob
                    global_W[idx_w_glob][rank_offset + c] += val_w

    # We need m1, n1, p1 for the block.
    # A: m1 x n1. B: n1 x p1. C: p1 x m1? No, C: m1 x p1.
    # W index: k*n + i.
    # So local W index: k_loc * m1 + i_loc.
    
    # Corrected add_decomposition signature
    def add_decomp(gU, gV, gW, rank_off, i_off, j_off, k_off, lU, lV, lW, m, p, n, m1, n1, p1):
        r = len(lU[0])
        for c in range(r):
            for idx_u in range(len(lU)):
                val = lU[idx_u][c]
                if val:
                    i_loc = idx_u // n1
                    j_loc = idx_u % n1
                    i_g = i_off + i_loc
                    j_g = j_off + j_loc
                    gU[i_g * m + j_g][rank_off + c] += val
            for idx_v in range(len(lV)):
                val = lV[idx_v][c]
                if val:
                    j_loc = idx_v // p1
                    k_loc = idx_v % p1
                    j_g = j_off + j_loc
                    k_g = k_off + k_loc
                    gV[j_g * p + k_g][rank_off + c] += val
            for idx_w in range(len(lW)):
                val = lW[idx_w][c]
                if val:
                    k_loc = idx_w // m1
                    i_loc = idx_w % m1
                    k_g = k_off + k_loc
                    i_g = i_off + i_loc
                    gW[k_g * n + i_g][rank_off + c] += val

    # Generate Schoolbook for m1 x n1 * n1 x p1
    def schoolbook(m1, n1, p1):
        r = m1 * n1 * p1
        U = [[0]*r for _ in range(m1*n1)]
        V = [[0]*r for _ in range(n1*p1)]
        W = [[0]*r for _ in range(p1*m1)]
        c = 0
        for i in range(m1):
            for j in range(n1):
                for k in range(p1):
                    U[i*n1+j][c] = 1
                    V[j*p1+k][c] = 1
                    W[k*m1+i][c] = 1
                    c += 1
        return U, V, W

    # Generate Strassen for 2x2
    def strassen_2x2():
        # m1=2, n1=2, p1=2
        # Rank 7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Indices:
        # U: 0:A11, 1:A12, 2:A21, 3:A22
        # V: 0:B11, 1:B12, 2:B21, 3:B22
        # W: 0:C11, 1:C12, 2:C21, 3:C22
        
        # M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1 # Wait, W is for C.
        # C11 gets M1. C22 gets M1.
        # W[0,0]=1 (C11), W[3,0]=1 (C22)
        
        # M2
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1 # C21
        W[3][1]=-1 # C22
        
        # M3
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1 # C12
        W[3][2]=1 # C22
        
        # M4
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[0][3]=1 # C11
        W[2][3]=1 # C21
        
        # M5
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1 # C11
        W[1][4]=1 # C12
        
        # M6
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1 # C22
        
        # M7
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1 # C11
        W[1][6]=-1 # C12
        
        return U, V, W

    # Initialize global matrices
    # Max rank 124
    max_rank = 124
    U = [[0]*max_rank for _ in range(25)]
    V = [[0]*max_rank for _ in range(25)]
    W = [[0]*max_rank for _ in range(25)]
    
    current_rank = 0
    
    # Block 1: i=0,1; j=0,1; k=0,1 (2x2)
    # Block 2: i=2,3,4; j=2,3,4; k=2,3,4 (3x3)
    
    # 1. A11(2x2) * B11(2x2) -> C11(2x2)
    lU, lV, lW = strassen_2x2()
    add_decomp(U, V, W, current_rank, 0, 0, 0, lU, lV, lW, 5, 5, 5, 2, 2, 2)
    current_rank += 7
    
    # 2. A12(2x3) * B21(3x2) -> C11(2x2)
    # i=0,1; j=2,3,4; k=0,1
    lU, lV, lW = schoolbook(2, 3, 2)
    add_decomp(U, V, W, current_rank, 0, 2, 0, lU, lV, lW, 5, 5, 5, 2, 3, 2)
    current_rank += 12
    
    # 3. A11(2x2) * B12(2x3) -> C12(2x3)
    # i=0,1; j=0,1; k=2,3,4
    lU, lV, lW = schoolbook(2, 2, 3)
    add_decomp(U, V, W, current_rank, 0, 0, 2, lU, lV, lW, 5, 5, 5, 2, 2, 3)
    current_rank += 12
    
    # 4. A12(2x3) * B22(3x3) -> C12(2x3)
    # i=0,1; j=2,3,4; k=2,3,4
    lU, lV, lW = schoolbook(2, 3, 3)
    add_decomp(U, V, W, current_rank, 0, 2, 2, lU, lV, lW, 5, 5, 5, 2, 3, 3)
    current_rank += 18
    
    # 5. A21(3x2) * B11(2x2) -> C21(3x2)
    # i=2,3,4; j=0,1; k=0,1
    lU, lV, lW = schoolbook(3, 2, 2)
    add_decomp(U, V, W, current_rank, 2, 0, 0, lU, lV, lW, 5, 5, 5, 3, 2, 2)
    current_rank += 12
    
    # 6. A22(3x3) * B21(3x2) -> C21(3x2)
    # i=2,3,4; j=2,3,4; k=0,1
    lU, lV, lW = schoolbook(3, 3, 2)
    add_decomp(U, V, W, current_rank, 2, 2, 0, lU, lV, lW, 5, 5, 5, 3, 3, 2)
    current_rank += 18
    
    # 7. A21(3x2) * B12(2x3) -> C22(3x3)
    # i=2,3,4; j=0,1; k=2,3,4
    lU, lV, lW = schoolbook(3, 2, 3)
    add_decomp(U, V, W, current_rank, 2, 0, 2, lU, lV, lW, 5, 5, 5, 3, 2, 3)
    current_rank += 18
    
    # 8. A22(3x3) * B22(3x3) -> C22(3x3)
    # i=2,3,4; j=2,3,4; k=2,3,4
    lU, lV, lW = schoolbook(3, 3, 3)
    add_decomp(U, V, W, current_rank, 2, 2, 2, lU, lV, lW, 5, 5, 5, 3, 3, 3)
    current_rank += 27
    
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

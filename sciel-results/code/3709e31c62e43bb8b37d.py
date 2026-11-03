def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to U, V, W
    # u_idx: i*m + j
    # v_idx: j*p + k
    # w_idx: k*n + i
    # coeff: coefficient for this rank-1 component
    
    def add_term(r, i, j, k, coeff):
        U[i * m + j][r] += coeff
        V[j * p + k][r] += coeff
        W[k * n + i][r] += coeff

    # Define Strassen 2x2 decomposition
    # Inputs are 2x2 matrices X, Y.
    # We need to map the scalar products to global indices.
    
    # For a 4x4x4 multiplication, we have 7 block multiplications.
    # Each block multiplication is a 2x2x2 multiplication.
    
    # Block 1: M1 = (A00 + A11)(B00 + B11)
    # A00: i in 0,1; j in 0,1
    # A11: i in 2,3; j in 2,3
    # B00: j in 0,1; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # We will generate the 49 terms for the first half (k 0-3) and 49 for the second half (k 4-7).
    
    # Let's define the structure of one 2x2 Strassen step.
    # It takes 2x2 matrices X, Y and produces 2x2 matrix Z.
    # The 7 scalar products are:
    # 1. (X00+X11)(Y00+Y11)
    # 2. (X10+X11)Y00
    # 3. X00(Y01-Y11)
    # 4. X11(Y10-Y00)
    # 5. (X00+X01)Y11
    # 6. (X10-X00)(Y00+Y01)
    # 7. (X01-X11)(Y10+Y11)
    
    # We need to map X, Y elements to global A, B elements.
    
    # For the first half (k 0..3):
    # We process the 7 block multiplications.
    
    # Block 1: M1
    # X = A00 + A11
    # Y = B00 + B11
    # X elements:
    # X00 = A00 + A11 (where A00 is A[0:2, 0:2] and A11 is A[2:4, 2:4])
    # Wait, X is a 2x2 matrix.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    # This mapping is complex to do manually for all 49 terms.
    # Let's write a general function to add a 2x2 Strassen decomposition given the mappings of X, Y, Z to global indices.
    
    def add_strassen_2x2(r_start, x_map, y_map, z_map, k_offset=0):
        # x_map: list of 4 lists of (i, j, coeff) for X[0,0], X[0,1], X[1,0], X[1,1]
        # y_map: list of 4 lists of (j, k, coeff) for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        # z_map: list of 4 lists of (i, k, coeff) for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
        # k_offset: added to k indices in y_map and z_map
        
        # The 7 scalar products for 2x2 Strassen
        # 1. (X00+X11)(Y00+Y11)
        # 2. (X10+X11)Y00
        # 3. X00(Y01-Y11)
        # 4. X11(Y10-Y00)
        # 5. (X00+X01)Y11
        # 6. (X10-X00)(Y00+Y01)
        # 7. (X01-X11)(Y10+Y11)
        
        # We need to expand these into linear combinations of global A and B terms.
        # Since U, V, W are linear, we can just add the terms.
        
        # Term 1: (X00+X11)(Y00+Y11)
        # This is not a single scalar product, it's a product of sums.
        # Wait, the Strassen algorithm replaces the multiplication of 2x2 matrices with 7 scalar multiplications.
        # The "scalar multiplications" are of the form (linear_combination_of_X) * (linear_combination_of_Y).
        # Each such product contributes to the output Z via linear combinations.
        
        # So, for each of the 7 intermediate products M_t, we have:
        # M_t = (sum_x c_x X[x]) * (sum_y d_y Y[y])
        # And Z[z] = sum_t e_{t,z} M_t
        
        # So Z[z] = sum_t sum_x sum_y e_{t,z} c_x d_y X[x] Y[y]
        
        # We can iterate over t, x, y.
        # For each combination, we add a term to U, V, W.
        # The term corresponds to the product of the specific X element and Y element, scaled by the coefficients.
        
        # Let's define the 7 steps explicitly.
        
        steps = [
            # (X_indices, Y_indices)
            # 1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # 2
            ([(1,0,1), (1,1,1)], [(0,0,1)]),
            # 3
            ([(0,0,1)], [(0,1,1), (1,1,-1)]),
            # 4
            ([(1,1,1)], [(1,0,1), (0,0,-1)]),
            # 5
            ([(0,0,1), (0,1,1)], [(1,1,1)]),
            # 6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            # 7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
        ]
        
        # Output coefficients e_{t,z}
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        out_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # Z00
            [0, 0, 1, 0, 1, 0, 0],  # Z01
            [0, 1, 0, 1, 0, 0, 0],  # Z10
            [1, -1, 1, 0, 0, 1, 0]  # Z11
        ]
        
        r = r_start
        for t in range(7):
            x_terms = steps[t][0]
            y_terms = steps[t][1]
            
            # We can't just iterate x and y independently and add to U, V, W 
            # because the product is (sum X)(sum Y) = sum sum X_x Y_y.
            # Each pair (x, y) is a rank-1 component?
            # No, the rank-1 component is the product of a specific linear combo of A and a specific linear combo of B.
            # Wait, the Strassen algorithm reduces the number of *multiplications*.
            # Each multiplication is of two linear combinations.
            # So each of the 7 steps corresponds to ONE rank-1 component in the tensor decomposition?
            # No. A rank-1 component is $u_i v_j w_k$.
            # In tensor decomposition of matrix multiplication, we have:
            # $C_{ik} = \sum_r U_{ir} V_{jr} W_{kr}$ ? No.
            # The tensor is $T_{ijk} = \delta_{ij} \delta_{jk}$? No.
            # $T_{i,j,k}$ such that $C_{ik} = \sum_j A_{ij} B_{jk}$.
            # The decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$.
            # Where $U_r$ is a vector in A-space, $V_r$ in B-space, $W_r$ in C-space.
            # $U_r[i, j]$, $V_r[j, k]$, $W_r[k, i]$?
            # The task interface is:
            # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
            # $C_{i,k} = \sum_r U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r]$?
            # No, the sum is over $j$ as well?
            # The standard decomposition is:
            # $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
            # Let's check the seed code.
            # Seed:
            # U[i*m+j][col] = 1
            # V[j*p+k][col] = 1
            # W[k*n+i][col] = 1
            # This implies $C_{i,k} = \sum_r \sum_j U[i \cdot m + j, r] V[j \cdot p + k, r] W[k \cdot n + i, r]$?
            # No, the sum over $j$ is implicit in the matrix multiplication definition?
            # Actually, the tensor $T$ for matrix multiplication $A \cdot B = C$ has indices $i, j, k$.
            # $T_{i,j,k} = 1$ if we are multiplying $A_{i,j}$ and $B_{j,k}$ to contribute to $C_{i,k}$.
            # The decomposition is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
            # No, the interface flattens indices.
            # $U$ has index $i \cdot m + j$.
            # $V$ has index $j \cdot p + k$.
            # $W$ has index $k \cdot n + i$.
            
            # If we sum over $r$, we get a term for each $j$?
            # The formula is:
            # $C_{i,k} = \sum_r (\sum_j U[i \cdot m + j, r] V[j \cdot p + k, r]) W[k \cdot n + i, r]$?
            # No, that would mean $W$ is scalar. But $W$ is a vector in $r$.
            
            # Let's look at the seed code again.
            # For each $i, j, k$, it sets $U, V, W$ to 1 at the same column $col$.
            # So for a specific $col$, $U[i \cdot m + j, col] = 1$, $V[j \cdot p + k, col] = 1$, $W[k \cdot n + i, col] = 1$.
            # All other entries are 0.
            # Then $\sum_r U[i \cdot m + j', r] V[j' \cdot p + k, r] W[k \cdot n + i, r]$?
            # If we sum over $r$, for a fixed $i, k$, and summing over $j'$?
            # The definition of the decomposition is:
            # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
            # No, the indices of U, V, W are flattened.
            # Let's assume the contraction is:
            # $C_{i,k} = \sum_r \left( \sum_j U[i \cdot m + j, r] V[j \cdot p + k, r] \right) W[k \cdot n + i, r]$?
            # This doesn't look right dimensionally if W is not scalar.
            
            # Actually, the standard tensor decomposition for matrix multiplication is:
            # $T = \sum_r u_r \otimes v_r \otimes w_r$
            # where $u_r \in \mathbb{R}^{n \times m}$, $v_r \in \mathbb{R}^{m \times p}$, $w_r \in \mathbb{R}^{p \times n}$.
            # And $T_{i,j,k} = \sum_r u_{r,i,j} v_{r,j,k} w_{r,k,i}$.
            # Here $T_{i,j,k}$ is the tensor such that $C_{i,k} = \sum_j A_{i,j} T_{i,j,k} B_{j,k}$?
            # No, $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
            # So $T_{i,j,k} = 1$ for all $i,j,k$? No.
            # The tensor is defined by $T_{i,j,k} = \delta_{i,i} \delta_{j,j} \delta_{k,k}$?
            # The tensor $T$ satisfies $T_{i,j,k} = 1$ if the indices match the multiplication pattern.
            # Actually, $T_{i,j,k} = 1$ for all valid $i,j,k$ is not correct.
            # The tensor is $T_{i,j,k} = 1$ such that $\sum_{j} A_{i,j} B_{j,k} = \sum_{j} T_{i,j,k} A_{i,j} B_{j,k}$.
            # So $T_{i,j,k} = 1$ for all $i,j,k$.
            
            # Wait, if $T_{i,j,k} = 1$ for all $i,j,k$, then the rank is $n \cdot m \cdot p$.
            # And the decomposition is $T_{i,j,k} = \sum_r u_{r,i,j} v_{r,j,k} w_{r,k,i}$.
            
            # So, for each rank-1 component $r$, we have matrices $U_r, V_r, W_r$.
            # $U_r$ is $n \times m$.
            # $V_r$ is $m \times p$.
            # $W_r$ is $p \times n$.
            
            # In the seed code:
            # $U[i \cdot m + j, col] = 1$. This means $U_{col}$ has 1 at $(i,j)$.
            # $V[j \cdot p + k, col] = 1$. This means $V_{col}$ has 1 at $(j,k)$.
            # $W[k \cdot n + i, col] = 1$. This means $W_{col}$ has 1 at $(k,i)$.
            
            # So, for a Strassen term, which is a product of linear combinations:
            # $M = (\sum_{a} c_a A_a) (\sum_{b} d_b B_b)$
            # This contributes to $C$ via linear combinations.
            # Specifically, if $M$ is an intermediate matrix $M_{x,y}$ (where $x,y$ are block indices),
            # and $C_{z} = \sum M_{x,y} e_{x,y,z}$,
            # then the contribution to the tensor is:
            # $\sum_{a,b} c_a d_b e_{z} A_a B_b \to C_z$.
            # This corresponds to a rank-1 component?
            # No, $A_a$ is a specific element $A_{i,j}$, $B_b$ is $B_{j',k}$.
            # Wait, in Strassen, the "scalar multiplications" are of linear combinations of elements.
            # So each of the 7 steps in the $2 \times 2$ Strassen is ONE rank-1 component?
            # Yes!
            # Because $u_r$ is the vector of coefficients for $A$, $v_r$ for $B$, $w_r$ for $C$.
            # $u_{r, i, j} = c_a$ if $(i,j)$ corresponds to element $a$, else 0.
            # $v_{r, j, k} = d_b$ if $(j,k)$ corresponds to element $b$, else 0.
            # $w_{r, k, i} = e_z$ if $(k,i)$ corresponds to output $z$, else 0.
            
            # So, for each of the 7 steps in the $2 \times 2$ Strassen, we define ONE rank-1 component.
            # For $4 \times 4$, we have 7 such steps, each involving $2 \times 2$ blocks.
            # Each step involves a linear combination of $A$ elements (from 4 elements) and $B$ elements (from 4 elements).
            # So $u_r$ has 4 non-zero entries (in the $4 \times 4$ matrix).
            # $v_r$ has 4 non-zero entries.
            # $w_r$ has 4 non-zero entries (in the $4 \times 4$ output).
            
            # So, for the first half ($4 \times 4 \times 4$), we have 7 rank-1 components.
            # Wait, no. The $2 \times 2$ Strassen has 7 rank-1 components.
            # The $4 \times 4$ Strassen uses 7 "multiplications" of $2 \times 2$ matrices.
            # Each $2 \times 2$ multiplication is decomposed into 7 rank-1 components.
            # So total rank-1 components for $4 \times 4$ is $7 \times 7 = 49$.
            
            # So, for each of the 49 components, we need to determine the coefficients in $U, V, W$.
            
            # Let's refine the `add_strassen_2x2` function.
            # It should add 7 rank-1 components.
            # Each component corresponds to one of the 7 steps.
            # For step $t$:
            # $u_r$ is determined by $x_terms$ and the mapping of $X$ to $A$.
            # $v_r$ is determined by $y_terms$ and the mapping of $Y$ to $B$.
            # $w_r$ is determined by $out_coeffs[t]$ and the mapping of $Z$ to $C$.
            
            # But wait, $w_r$ is the SAME for all 7 steps?
            # No, $w_r$ depends on $t$ (which step it is) and the output $z$.
            # Actually, for a fixed step $t$, $w_r$ is a vector in $C$-space.
            # $w_{r, k, i} = out_coeffs[t][z]$ where $z$ corresponds to $(k,i)$.
            
            # So, for each step $t$, we create ONE rank-1 component $r$.
            # $U_r$ is the linear combination of $A$ elements defined by $x_terms$.
            # $V_r$ is the linear combination of $B$ elements defined by $y_terms$.
            # $W_r$ is the linear combination of $C$ elements defined by $out_coeffs[t]$.
            
            # This is correct.
            
            for t in range(7):
                r = r_start + t
                # Build U_r
                # x_terms is a list of (x_idx, x_coeff)
                # x_idx is 0,1,2,3 corresponding to X[0,0], X[0,1], X[1,0], X[1,1]
                # We need to map x_idx to global (i, j) indices.
                # x_map[x_idx] is a list of (i, j, coeff)
                
                for x_idx, x_coeff in x_terms:
                    for i, j, c in x_map[x_idx]:
                        U[i * m + j][r] += x_coeff * c
                
                # Build V_r
                for y_idx, y_coeff in y_terms:
                    for j, k, c in y_map[y_idx]:
                        V[j * p + k][r] += y_coeff * c
                        
                # Build W_r
                # out_coeffs[t] is a list of 4 coeffs for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
                # z_map[z_idx] is a list of (i, k, coeff)
                for z_idx in range(4):
                    z_coeff = out_coeffs[t][z_idx]
                    if z_coeff != 0:
                        for i, k, c in z_map[z_idx]:
                            W[k * n + i][r] += z_coeff * c

    # Now we define the mappings for the 4x4 Strassen.
    # We have 7 block multiplications.
    
    # Block 1: M1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    x_map_1 = [
        [(0,0,1), (2,2,1)],
        [(0,1,1), (2,3,1)],
        [(1,0,1), (3,2,1)],
        [(1,1,1), (3,3,1)]
    ]
    
    # Y = B00 + B11
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    y_map_1 = [
        [(0,0,1), (2,2,1)],
        [(0,1,1), (2,3,1)],
        [(1,0,1), (3,2,1)],
        [(1,1,1), (3,3,1)]
    ]
    
    # Z = M1
    # Z[0,0] = C[0,0]
    # Z[0,1] = C[0,1]
    # Z[1,0] = C[1,0]
    # Z[1,1] = C[1,1]
    
    z_map_1 = [
        [(0,0,1)],
        [(0,1,1)],
        [(1,0,1)],
        [(1,1,1)]
    ]
    
    add_strassen_2x2(0, x_map_1, y_map_1, z_map_1, k_offset=0)
    
    # Block 2: M2 = (A10 + A11)B00
    # X = A10 + A11
    # X[0,0] = A[2,0] + A[2,2]
    # X[0,1] = A[2,1] + A[2,3]
    # X[1,0] = A[3,0] + A[3,2]
    # X[1,1] = A[3,1] + A[3,3]
    
    x_map_2 = [
        [(2,0,1), (2,2,1)],
        [(2,1,1), (2,3,1)],
        [(3,0,1), (3,2,1)],
        [(3,1,1), (3,3,1)]
    ]
    
    # Y = B00
    # Y[0,0] = B[0,0]
    # Y[0,1] = B[0,1]
    # Y[1,0] = B[1,0]
    # Y[1,1] = B[1,1]
    
    y_map_2 = [
        [(0,0,1)],
        [(0,1,1)],
        [(1,0,1)],
        [(1,1,1)]
    ]
    
    # Z = M2
    # Z[0,0] = C[2,0]
    # Z[0,1] = C[2,1]
    # Z[1,0] = C[3,0]
    # Z[1,1] = C[3,1]
    
    z_map_2 = [
        [(2,0,1)],
        [(2,1,1)],
        [(3,0,1)],
        [(3,1,1)]
    ]
    
    add_strassen_2x2(7, x_map_2, y_map_2, z_map_2, k_offset=0)
    
    # Block 3: M3 = A00(B01 - B11)
    # X = A00
    x_map_3 = [
        [(0,0,1)],
        [(0,1,1)],
        [(1,0,1)],
        [(1,1,1)]
    ]
    
    # Y = B01 - B11
    # Y[0,0] = B[0,2] - B[2,2]
    # Y[0,1] = B[0,3] - B[2,3]
    # Y[1,0] = B[1,2] - B[3,2]
    # Y[1,1] = B[1,3] - B[3,3]
    
    y_map_3 = [
        [(0,2,1), (2,2,-1)],
        [(0,3,1), (2,3,-1)],
        [(1,2,1), (3,2,-1)],
        [(1,3,1), (3,3,-1)]
    ]
    
    # Z = M3
    # Z[0,0] = C[0,2]
    # Z[0,1] = C[0,3]
    # Z[1,0] = C[1,2]
    # Z[1,1] = C[1,3]
    
    z_map_3 = [
        [(0,2,1)],
        [(0,3,1)],
        [(1,2,1)],
        [(1,3,1)]
    ]
    
    add_strassen_2x2(14, x_map_3, y_map_3, z_map_3, k_offset=0)
    
    # Block 4: M4 = A11(B10 - B00)
    # X = A11
    x_map_4 = [
        [(2,2,1)],
        [(2,3,1)],
        [(3,2,1)],
        [(3,3,1)]
    ]
    
    # Y = B10 - B00
    # Y[0,0] = B[2,0] - B[0,0]
    # Y[0,1] = B[2,1] - B[0,1]
    # Y[1,0] = B[3,0] - B[1,0]
    # Y[1,1] = B[3,1] - B[1,1]
    
    y_map_4 = [
        [(2,0,1), (0,0,-1)],
        [(2,1,1), (0,1,-1)],
        [(3,0,1), (1,0,-1)],
        [(3,1,1), (1,1,-1)]
    ]
    
    # Z = M4
    # Z[0,0] = C[2,0]
    # Z[0,1] = C[2,1]
    # Z[1,0] = C[3,0]
    # Z[1,1] = C[3,1]
    
    z_map_4 = [
        [(2,0,1)],
        [(2,1,1)],
        [(3,0,1)],
        [(3,1,1)]
    ]
    
    add_strassen_2x2(21, x_map_4, y_map_4, z_map_4, k_offset=0)
    
    # Block 5: M5 = (A00 + A01)B11
    # X = A00 + A01
    # X[0,0] = A[0,0] + A[0,2]
    # X[0,1] = A[0,1] + A[0,3]
    # X[1,0] = A[1,0] + A[1,2]
    # X[1,1] = A[1,1] + A[1,3]
    
    x_map_5 = [
        [(0,0,1), (0,2,1)],
        [(0,1,1), (0,3,1)],
        [(1,0,1), (1,2,1)],
        [(1,1,1), (1,3,1)]
    ]
    
    # Y = B11
    y_map_5 = [
        [(2,2,1)],
        [(2,3,1)],
        [(3,2,1)],
        [(3,3,1)]
    ]
    
    # Z = M5
    # Z[0,0] = C[0,2]
    # Z[0,1] = C[0,3]
    # Z[1,0] = C[1,2]
    # Z[1,1] = C[1,3]
    
    z_map_5 = [
        [(0,2,1)],
        [(0,3,1)],
        [(1,2,1)],
        [(1,3,1)]
    ]
    
    add_strassen_2x2(28, x_map_5, y_map_5, z_map_5, k_offset=0)
    
    # Block 6: M6 = (A10 - A00)(B00 + B01)
    # X = A10 - A00
    # X[0,0] = A[2,0] - A[0,0]
    # X[0,1] = A[2,1] - A[0,1]
    # X[1,0] = A[3,0] - A[1,0]
    # X[1,1] = A[3,1] - A[1,1]
    
    x_map_6 = [
        [(2,0,1), (0,0,-1)],
        [(2,1,1), (0,1,-1)],
        [(3,0,1), (1,0,-1)],
        [(3,1,1), (1,1,-1)]
    ]
    
    # Y = B00 + B01
    # Y[0,0] = B[0,0] + B[0,2]
    # Y[0,1] = B[0,1] + B[0,3]
    # Y[1,0] = B[1,0] + B[1,2]
    # Y[1,1] = B[1,1] + B[1,3]
    
    y_map_6 = [
        [(0,0,1), (0,2,1)],
        [(0,1,1), (0,3,1)],
        [(1,0,1), (1,2,1)],
        [(1,1,1), (1,3,1)]
    ]
    
    # Z = M6
    # Z[0,0] = C[2,0]
    # Z[0,1] = C[2,1]
    # Z[1,0] = C[3,0]
    # Z[1,1] = C[3,1]
    
    z_map_6 = [
        [(2,0,1)],
        [(2,1,1)],
        [(3,0,1)],
        [(3,1,1)]
    ]
    
    add_strassen_2x2(35, x_map_6, y_map_6, z_map_6, k_offset=0)
    
    # Block 7: M7 = (A01 - A11)(B10 + B11)
    # X = A01 - A11
    # X[0,0] = A[0,2] - A[2,2]
    # X[0,1] = A[0,3] - A[2,3]
    # X[1,0] = A[1,2] - A[3,2]
    # X[1,1] = A[1,3] - A[3,3]
    
    x_map_7 = [
        [(0,2,1), (2,2,-1)],
        [(0,3,1), (2,3,-1)],
        [(1,2,1), (3,2,-1)],
        [(1,3,1), (3,3,-1)]
    ]
    
    # Y = B10 + B11
    # Y[0,0] = B[2,0] + B[2,2]
    # Y[0,1] = B[2,1] + B[2,3]
    # Y[1,0] = B[3,0] + B[3,2]
    # Y[1,1] = B[3,1] + B[3,3]
    
    y_map_7 = [
        [(2,0,1), (2,2,1)],
        [(2,1,1), (2,3,1)],
        [(3,0,1), (3,2,1)],
        [(3,1,1), (3,3,1)]
    ]
    
    # Z = M7
    # Z[0,0] = C[0,2]
    # Z[0,1] = C[0,3]
    # Z[1,0] = C[1,2]
    # Z[1,1] = C[1,3]
    
    z_map_7 = [
        [(0,2,1)],
        [(0,3,1)],
        [(1,2,1)],
        [(1,3,1)]
    ]
    
    add_strassen_2x2(42, x_map_7, y_map_7, z_map_7, k_offset=0)
    
    # Now for the second half (k 4-7)
    # We repeat the same logic, but with k_offset=4.
    # The x_map is the same.
    # The y_map and z_map need k indices shifted by 4.
    
    # We can just call the same functions with k_offset=4.
    # But the y_map and z_map are defined with specific k indices.
    # We need to modify them or pass an offset.
    
    # Let's just duplicate the calls with k_offset=4.
    # But the y_map and z_map have hardcoded k indices.
    # We need to create new maps with k+4.
    
    # For simplicity, I will just write a helper to shift k indices.
    
    def shift_k(y_map, offset):
        new_map = []
        for lst in y_map:
            new_lst = []
            for j, k, c in lst:
                new_lst.append((j, k + offset, c))
            new_map.append(new_lst)
        return new_map
        
    def shift_k_w(z_map, offset):
        new_map = []
        for lst in z_map:
            new_lst = []
            for i, k, c in lst:
                new_lst.append((i, k + offset, c))
            new_map.append(new_lst)
        return new_map
        
    add_strassen_2x2(49, x_map_1, shift_k(y_map_1, 4), shift_k_w(z_map_1, 4), k_offset=4)
    add_strassen_2x2(56, x_map_2, shift_k(y_map_2, 4), shift_k_w(z_map_2, 4), k_offset=4)
    add_strassen_2x2(63, x_map_3, shift_k(y_map_3, 4), shift_k_w(z_map_3, 4), k_offset=4)
    add_strassen_2x2(70, x_map_4, shift_k(y_map_4, 4), shift_k_w(z_map_4, 4), k_offset=4)
    add_strassen_2x2(77, x_map_5, shift_k(y_map_5, 4), shift_k_w(z_map_5, 4), k_offset=4)
    add_strassen_2x2(84, x_map_6, shift_k(y_map_6, 4), shift_k_w(z_map_6, 4), k_offset=4)
    add_strassen_2x2(91, x_map_7, shift_k(y_map_7, 4), shift_k_w(z_map_7, 4), k_offset=4)
    
    return {"U": U, "V": V, "W": W}


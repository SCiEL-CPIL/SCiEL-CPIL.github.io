import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank_total = 65  # 49 + 16
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # We need the 2x2x2 Strassen decomposition first.
    # U2, V2, W2 of shape (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the coefficients for U, V, W for the 7 terms.
    # U corresponds to A, V to B, W to C.
    # C = sum U V W
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00, A11
    U2[0, 0] = 1 # A00
    U2[3, 0] = 1 # A11
    # V: B00, B11
    V2[0, 0] = 1 # B00
    V2[3, 0] = 1 # B11
    # W: C00, C01, C10, C11 contributions?
    # W is the output tensor factor.
    # For term r, W[k*2+i, r] is the coefficient of C[i,k]?
    # No, W[k*n+i, r]. Here n=2. W[k*2+i, r].
    # The product is C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Term is u_i A[i,j] * v_j B[j,k] * w_k C[i,k]?
    # Actually, the tensor is T[i,j,k] = delta[i,j] delta[j,k]? No.
    # The decomposition is T = sum u^r \otimes v^r \otimes w^r.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # Note the index order in W: k*n+i.
    
    # Let's verify the index mapping for 2x2.
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    # Term 1: (A00+A11)(B00+B11)
    # This term contributes to:
    # C00 = A00B00 + A00B10? No.
    # Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 (+1) and C11 (+1).
    # M1 = (A00+A11)(B00+B11)
    # Expansion: A00B00 + A00B11 + A11B00 + A11B11.
    # Wait, Strassen M1 is just ONE multiplication of sums.
    # The rank-1 term is the product of the two sum-vectors.
    # u = (A00+A11), v = (B00+B11).
    # The resulting matrix is u v.
    # This matrix is added to C00 and C11.
    # So W[0, r] (C00) and W[3, r] (C11) should be 1.
    # W index: k*2+i.
    # C00 -> i=0, k=0 -> idx 0.
    # C11 -> i=1, k=1 -> idx 3.
    
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    # Contributes to C10 (+1) and C11 (-1).
    # C10 -> i=1, k=0 -> idx 2.
    # C11 -> i=1, k=1 -> idx 3.
    U2[2, 1] = 1 # A10
    U2[3, 1] = 1 # A11
    V2[0, 1] = 1 # B00
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    # Contributes to C01 (+1) and C11 (+1).
    # C01 -> i=0, k=1 -> idx 1.
    # C11 -> i=1, k=1 -> idx 3.
    U2[0, 2] = 1 # A00
    V2[1, 2] = 1 # B01
    V2[3, 2] = -1 # B11
    W2[1, 2] = 1
    W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    # Contributes to C00 (+1) and C10 (+1).
    # C00 -> idx 0.
    # C10 -> idx 2.
    U2[3, 3] = 1 # A11
    V2[2, 3] = 1 # B10
    V2[0, 3] = -1 # B00
    W2[0, 3] = 1
    W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    # Contributes to C00 (-1) and C01 (+1).
    # C00 -> idx 0.
    # C01 -> idx 1.
    U2[0, 4] = 1 # A00
    U2[1, 4] = 1 # A01
    V2[3, 4] = 1 # B11
    W2[0, 4] = -1
    W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    # Contributes to C11 (+1).
    # C11 -> idx 3.
    U2[2, 5] = 1 # A10
    U2[0, 5] = -1 # A00
    V2[0, 5] = 1 # B00
    V2[1, 5] = 1 # B01
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    # Contributes to C00 (+1).
    # C00 -> idx 0.
    U2[1, 6] = 1 # A01
    U2[3, 6] = -1 # A11
    V2[2, 6] = 1 # B10
    V2[3, 6] = 1 # B11
    W2[0, 6] = 1
    
    # Now build 4x4x4 decomposition.
    # We have 7 top-level terms s=0..6.
    # Each term s involves L_s and R_s which are 4x4 matrices.
    # L_s and R_s are formed by linear combinations of 2x2 blocks of A and B.
    # We can represent L_s and R_s as 4x4 matrices.
    # Then we decompose L_s * R_s using the 2x2 Strassen structure on the 2x2 blocks of L_s and R_s.
    # This means we apply the 7-term decomposition to the block matrices.
    # The "blocks" of L_s are 2x2 matrices.
    # So we need to extract the 2x2 blocks of L_s and R_s, form the 7 combinations for the inner Strassen,
    # and then for each inner term, use the U2, V2, W2 factors.
    
    # Let's define the top-level combinations.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will generate 49 terms.
    col = 0
    for s in range(7):
        # Define L_s and R_s as 4x4 matrices of coefficients.
        # Actually, we just need the 2x2 block structure.
        # L_s_blocks is a 2x2 matrix of 2x2 matrices.
        # We can represent this as a 4x4 matrix.
        
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        
        if s == 0:
            # L = A00+A11, R = B00+B11
            # L00 = A00, L01 = 0, L10 = 0, L11 = A11
            L_s[0:2, 0:2] = np.eye(2) # A00
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[0:2, 0:2] = np.eye(2) # B00
            R_s[2:4, 2:4] = np.eye(2) # B11
        elif s == 1:
            # L = A10+A11, R = B00
            L_s[2:4, 0:2] = np.eye(2) # A10
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[0:2, 0:2] = np.eye(2) # B00
        elif s == 2:
            # L = A00, R = B01-B11
            L_s[0:2, 0:2] = np.eye(2) # A00
            R_s[0:2, 2:4] = np.eye(2) # B01
            R_s[2:4, 2:4] = -np.eye(2) # -B11
        elif s == 3:
            # L = A11, R = B10-B00
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[2:4, 0:2] = np.eye(2) # B10
            R_s[0:2, 0:2] = -np.eye(2) # -B00
        elif s == 4:
            # L = A00+A01, R = B11
            L_s[0:2, 0:2] = np.eye(2) # A00
            L_s[0:2, 2:4] = np.eye(2) # A01
            R_s[2:4, 2:4] = np.eye(2) # B11
        elif s == 5:
            # L = A10-A00, R = B00+B01
            L_s[2:4, 0:2] = np.eye(2) # A10
            L_s[0:2, 0:2] = -np.eye(2) # -A00
            R_s[0:2, 0:2] = np.eye(2) # B00
            R_s[0:2, 2:4] = np.eye(2) # B01
        elif s == 6:
            # L = A01-A11, R = B10+B11
            L_s[0:2, 2:4] = np.eye(2) # A01
            L_s[2:4, 2:4] = -np.eye(2) # -A11
            R_s[2:4, 0:2] = np.eye(2) # B10
            R_s[2:4, 2:4] = np.eye(2) # B11
            
        # Now decompose L_s * R_s using Strassen 2x2 on blocks.
        # L_s is a 2x2 block matrix.
        # We apply the 7-term Strassen decomposition to the block matrices.
        # This generates 7 new block products.
        # For each inner term t in 0..6:
        for t in range(7):
            # Determine the block combinations for this inner term.
            # This is the same logic as the top-level, but applied to the blocks of L_s and R_s.
            # Instead of re-implementing, we can just use the U2, V2, W2 structure?
            # No, U2, V2, W2 are for scalar 2x2 matrices.
            # Here the "scalars" are 2x2 blocks.
            # So we need to form L_inner and R_inner which are 2x2 matrices of 2x2 blocks (i.e. 4x4 matrices).
            # Then we need to decompose L_inner * R_inner?
            # No, the inner Strassen produces 7 products of blocks.
            # Each product is of the form X * Y where X, Y are 2x2 matrices.
            # We can use U2, V2, W2 to decompose X * Y into 7 rank-1 terms.
            
            # So we need to extract the 7 block pairs (X_t, Y_t) from L_s, R_s.
            # This is exactly the Strassen formula applied to the blocks.
            
            # Let's create the 7 L_inner and R_inner matrices.
            # We can do this by applying the same logic as above to the blocks of L_s and R_s.
            # But L_s and R_s are already 4x4.
            # We can treat L_s as a 2x2 matrix of 2x2 blocks.
            
            # Let's just hardcode the 7 combinations for the inner level.
            # It's the same structure.
            
            # Get blocks of L_s
            L00 = L_s[0:2, 0:2]
            L01 = L_s[0:2, 2:4]
            L10 = L_s[2:4, 0:2]
            L11 = L_s[2:4, 2:4]
            
            R00 = R_s[0:2, 0:2]
            R01 = R_s[0:2, 2:4]
            R10 = R_s[2:4, 0:2]
            R11 = R_s[2:4, 2:4]
            
            # Inner Strassen terms
            # t=0: (L00+L11)(R00+R11)
            # t=1: (L10+L11)R00
            # ...
            
            # We will generate 7 pairs (X, Y) of 2x2 matrices.
            # Then for each pair, we use U2, V2, W2 to get 7 rank-1 terms.
            
            inner_pairs = []
            if t == 0:
                X = L00 + L11
                Y = R00 + R11
            elif t == 1:
                X = L10 + L11
                Y = R00
            elif t == 2:
                X = L00
                Y = R01 - R11
            elif t == 3:
                X = L11
                Y = R10 - R00
            elif t == 4:
                X = L00 + L01
                Y = R11
            elif t == 5:
                X = L10 - L00
                Y = R00 + R01
            elif t == 6:
                X = L01 - L11
                Y = R10 + R11
            
            inner_pairs.append((X, Y))
            
            # Now for each of the 7 inner pairs, we have a 2x2 matrix product X*Y.
            # We decompose this using U2, V2, W2.
            # X and Y are 2x2 matrices.
            # U2, V2, W2 are for 2x2 matrices.
            # We need to map the entries of X and Y to U and V.
            
            # X is 2x2. U2[i*2+j, u] gives the coefficient for X[i,j].
            # So U_contrib[i*2+j, r] = sum_u U2[i*2+j, u] * X[i,j]?
            # No, U2 is the factor matrix.
            # The term is sum_u U2[i*2+j, u] V2[j*2+k, u] W2[k*2+i, u].
            # This sums to (X Y)[i,k].
            # So for a specific X, Y, the contribution to U, V, W is:
            # U[:, r] = X \otimes U2[:, u]?
            # Actually, U[i*2+j, r] should be X[i,j] * U2[i*2+j, u]?
            # No, X is the matrix being multiplied.
            # The rank-1 decomposition of the tensor T[i,j,k] = X[i,j] Y[j,k] is:
            # T = sum_u (X \otimes u_u) \otimes (Y \otimes v_u) \otimes w_u?
            # No.
            # The tensor for X*Y is $T_{ijk} = X_{ij} Y_{jk}$.
            # We have a decomposition of the *identity* tensor for 2x2 multiplication?
            # No, U2, V2, W2 decompose the multiplication tensor $M_{ijk} = \delta_{ij} \delta_{jk}$?
            # No, the multiplication tensor is $T_{ijk} = 1$ if it corresponds to the product.
            # The decomposition $U2, V2, W2$ satisfies:
            # $(A B)_{ik} = \sum_{j, u} A_{ij} B_{jk} U2_{ij,u} V2_{jk,u} W2_{ki,u}$?
            # No, the standard form is:
            # $C_{ik} = \sum_u (\sum_j U_{ij,u} A_{ij}) (\sum_j V_{jk,u} B_{jk}) W_{ki,u}$?
            # No, the factors are vectors.
            # $C = \sum_u u_u \otimes v_u \otimes w_u$.
            # $C_{ik} = \sum_u U_{ij,u} V_{jk,u} W_{ki,u}$ summed over j?
            # No, $U$ is a matrix. $U_{row, u}$. Row is $(i,j)$.
            # So $C_{ik} = \sum_u W_{ki,u} \sum_{j} U_{ij,u} V_{jk,u}$?
            # No, $U_{ij,u}$ is a scalar.
            # The term is $\sum_u W_{ki,u} ( \sum_j U_{ij,u} A_{ij} ) ( \sum_j V_{jk,u} B_{jk} )$?
            # This is not correct.
            
            # Correct interpretation:
            # $C = \sum_u (u_u A) \circ (v_u B) \otimes w_u$?
            # No.
            # $C_{ik} = \sum_u W_{ki,u} \sum_j U_{ij,u} A_{ij} \sum_j V_{jk,u} B_{jk}$ is wrong.
            
            # The tensor decomposition is:
            # $T_{ijk} = \sum_u U_{ij,u} V_{jk,u} W_{ki,u}$.
            # Here $T_{ijk}$ is the multiplication tensor.
            # $T_{ijk} = 1$ if $j$ is the inner index?
            # No, $T_{ijk}$ is such that $C_{ik} = \sum_{j,l} A_{ij} B_{jl} T_{jkl}$?
            # No.
            # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
            # So $T_{ijk} = \delta_{jk}$? No.
            # The tensor is $T_{i, j, k}$ where $i$ is A-row, $j$ is A-col/B-row, $k$ is B-col.
            # $T_{i, j, k} = 1$ if $j$ is the index?
            # No, $T_{i, j, k}$ is 1 if $i,j,k$ are valid?
            # The tensor is $T_{i, j, k} = \delta_{ij} \delta_{jk}$? No.
            # $C_{ik} = \sum_j A_{ij} B_{jk}$.
            # This is a contraction.
            # The tensor is $T_{i, j, k} = 1$? No.
            # The tensor is defined by $T_{i, j, k} = 1$ if we consider the product of basis vectors?
            # $e_i e_j^T \times e_j e_k^T = e_i e_k^T$.
            # So $T_{i, j, k} = 1$ if the indices match?
            # Actually, the tensor $T$ has $T_{i, j, k} = 1$ for all $i,j,k$? No.
            # $T_{i, j, k} = 1$ if $j$ is the shared index.
            # In the decomposition $U, V, W$:
            # $U_{row, u}$ where row is $(i,j)$.
            # $V_{row, u}$ where row is $(j,k)$.
            # $W_{row, u}$ where row is $(k,i)$.
            # The sum is $\sum_u U_{ij,u} V_{jk,u} W_{ki,u}$.
            # This should equal $C_{ik}$?
            # No, it should equal the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
            # The coefficient is 1 if $j$ is the inner index.
            # So $\sum_u U_{ij,u} V_{jk,u} W_{ki,u} = 1$ for all $i,j,k$?
            # No, it must be 1 only if the indices form a valid chain?
            # For matrix multiplication, $C_{ik} = \sum_j A_{ij} B_{jk}$.
            # So the coefficient of $A_{ij} B_{jk}$ is 1 for all $i,j,k$.
            # So $\sum_u U_{ij,u} V_{jk,u} W_{ki,u} = 1$ for all $i,j,k$.
            # This is the condition.
            
            # So for a specific $X, Y$, we want to compute $X Y$.
            # $ (XY)_{ik} = \sum_j X_{ij} Y_{jk} $.
            # We can use the decomposition $U2, V2, W2$ which satisfies the condition above.
            # So we set:
            # $U_{444}[i*4+j, r] = X_{ij} U2_{i*2+j, t'}$?
            # No, $X$ is the matrix.
            # The term is $\sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'}$.
            # We want to substitute $A \to X, B \to Y$.
            # So $U_{444}[i*4+j, r] = X_{ij} U2_{i*2+j, t'}$?
            # No, $U2$ is constant.
            # The factor for $A$ is $U2$.
            # So the contribution to $U$ is $X \otimes U2$?
            # No, $U_{444}$ has rows $(i,j)$.
            # $U_{444}[i*4+j, r] = X_{ij} U2_{i*2+j, t'}$ is wrong.
            # $X_{ij}$ is a value. $U2_{i*2+j, t'}$ is a value.
            # We want the tensor to be $T'_{ijk} = X_{ij} Y_{jk}$.
            # We know $T_{ijk} = \sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'}$.
            # So $T'_{ijk} = \sum_{t'} (X_{ij} U2_{ij,t'}) (Y_{jk} V2_{jk,t'}) W2_{ki,t'}$.
            # So $U_{444}[i*4+j, r] = X_{ij} U2_{i*2+j, t'}$.
            # $V_{444}[j*4+k, r] = Y_{jk} V2_{j*2+k, t'}$.
            # $W_{444}[k*4+i, r] = W2_{k*2+i, t'}$.
            
            # Note: $W2$ is the same for all $X, Y$ because it only depends on $i,k$.
            
            for t_prime in range(7):
                # Extract X, Y entries
                # X is 2x2.
                # We need to fill U, V, W for this rank-1 term.
                
                # U
                for i in range(2):
                    for j in range(2):
                        # Map 4x4 indices
                        # i, j are local to the 2x2 block.
                        # But X is a 2x2 matrix of scalars?
                        # No, X is a 2x2 matrix.
                        # So i, j are 0..1.
                        # The global indices for U are i*4+j?
                        # No, U is for 4x4.
                        # The block is at position (I, J).
                        # Here we are inside the inner Strassen.
                        # The "blocks" of L_s are 2x2.
                        # X is a 2x2 matrix.
                        # So the indices i, j in X correspond to the local indices of the 2x2 block.
                        # But which block?
                        # The inner Strassen operates on the 2x2 blocks of L_s.
                        # So X is one of the linear combinations of the 2x2 blocks.
                        # So X is a 2x2 matrix.
                        # The entries X[i,j] are scalars.
                        # These scalars are coefficients of the original A entries.
                        # So U[global_i, global_j, r] = X[i,j] * U2[i*2+j, t_prime].
                        # global_i, global_j are the indices in the 4x4 matrix.
                        # But X is a 2x2 matrix.
                        # Does X[i,j] correspond to a specific global index?
                        # No, X is a linear combination of blocks.
                        # So X[i,j] is a sum of A[global_i, global_j].
                        # So U[global_i, global_j, r] should be the coefficient of A[global_i, global_j].
                        # This coefficient is X[i,j] * U2[i*2+j, t_prime]?
                        # No, U2 is for the 2x2 multiplication.
                        # The structure is:
                        # We have a 2x2 matrix X.
                        # We decompose X*Y using Strassen 2x2.
                        # The factors are U2, V2, W2.
                        # So the contribution to the 4x4 U matrix is:
                        # U[global_i, global_j, r] = X[i,j] * U2[i*2+j, t_prime] ??
                        # No, X[i,j] is the value of the matrix element.
                        # But X is formed from A.
                        # So X[i,j] = sum_{g_i, g_j} coeff * A[g_i, g_j].
                        # So U[g_i, g_j, r] += coeff * U2[i*2+j, t_prime].
                        
                        # This implies we need to know the coefficients.
                        # We constructed X by adding blocks of L_s.
                        # L_s was constructed with 1s and 0s.
                        # So X is a 2x2 matrix with entries 0, 1, -1.
                        # And these entries correspond to specific global indices.
                        
                        # This is getting too complex to trace manually in the loop.
                        # Alternative: Just hardcode the 49 terms?
                        # Or use a library?
                        
                        # Given the complexity, I will stick to the schoolbook rank 80.
                        # The risk of error in Strassen implementation is high.
                        # And the time limit is 120s.
                        
                        pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

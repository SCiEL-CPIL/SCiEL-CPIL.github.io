    def construct():
        n, m, p = 4, 5, 6
        rank = 84
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_comp(u, v, w, val):
            nonlocal col
            U[u][col] += val
            V[v][col] += val
            W[w][col] += val
            col += 1
            
        def add_schoolbook_2x2(A_r, A_c, B_r, B_c, C_r, C_c, coeff):
            # A_r, A_c: top-left row/col of A block
            # B_r, B_c: top-left row/col of B block
            # C_r, C_c: top-left row/col of C block
            for r in range(2):
                for p_loc in range(2):
                    for q_loc in range(2):
                        u_idx = (A_r + p_loc) * m + (A_c + r)
                        v_idx = (B_r + r) * p + (B_c + q_loc)
                        w_idx = (C_c + q_loc) * n + (C_r + p_loc)
                        add_comp(u_idx, v_idx, w_idx, coeff)
                        
        # Block 1: 4x4x4 Strassen (Rank 28)
        # A: 0-3, 0-3. B: 0-3, 0-3. C: 0-3, 0-3.
        
        # M1 = (A00+A11)(B00+B11)
        # A00: 0,0. A11: 2,2.
        # B00: 0,0. B11: 2,2.
        # C00: 0,0.
        
        # We need to expand (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # Each term is a 2x2x2 mult.
        
        # Term 1.1: A00 B00
        add_schoolbook_2x2(0, 0, 0, 0, 0, 0, 1)
        # Term 1.2: A00 B11
        add_schoolbook_2x2(0, 0, 2, 2, 0, 0, 1)
        # Term 1.3: A11 B00
        add_schoolbook_2x2(2, 2, 0, 0, 0, 0, 1)
        # Term 1.4: A11 B11
        add_schoolbook_2x2(2, 2, 2, 2, 0, 0, 1)
        
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # A10: 2,0. A11: 2,2. B00: 0,0. C10: 2,0.
        add_schoolbook_2x2(2, 0, 0, 0, 2, 0, 1)
        add_schoolbook_2x2(2, 2, 0, 0, 2, 0, 1)
        
        # M3 = A00(B10-B11) = A00B10 - A00B11
        # A00: 0,0. B10: 2,0. B11: 2,2. C01: 0,2.
        add_schoolbook_2x2(0, 0, 2, 0, 0, 2, 1)
        add_schoolbook_2x2(0, 0, 2, 2, 0, 2, -1)
        
        # M4 = A11(B01-B00) = A11B01 - A11B00
        # A11: 2,2. B01: 0,2. B00: 0,0. C11: 2,2.
        add_schoolbook_2x2(2, 2, 0, 2, 2, 2, 1)
        add_schoolbook_2x2(2, 2, 0, 0, 2, 2, -1)
        
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # A00: 0,0. A01: 0,2. B11: 2,2. C00: 0,0 (subtracted in C00), C01: 0,2 (added in C01)
        # Wait, C00 uses -M5. C01 uses +M5.
        # We just add the components to U,V,W. The signs are handled in the W accumulation?
        # NO. U, V, W are the factors. The sum of U*V*W gives C.
        # If C = M1 - M5, then the components of M5 must have a negative sign in U, V, or W?
        # No, the decomposition is linear.
        # $C = \sum u_c \otimes v_c \otimes w_c$.
        # If $C = M_1 - M_5$, we can write $M_5$ with a negative coefficient in one of the matrices.
        # Usually, we put the sign in W or U.
        # My `add_schoolbook_2x2` adds `coeff` to U, V, W.
        # If I pass -1, it subtracts from all three?
        # $u \otimes v \otimes w$. If I multiply u by -1, the term becomes $(-u) \otimes v \otimes w = - (u \otimes v \otimes w)$.
        # So yes, passing -1 to `add_comp` (which adds to U, V, W) effectively negates the tensor component.
        
        add_schoolbook_2x2(0, 0, 2, 2, 0, 0, 1)
        add_schoolbook_2x2(0, 2, 2, 2, 0, 0, 1)
        
        # M6 = (A10-A00)(B00+B10) = A10B00 + A10B10 - A00B00 - A00B10
        # C11 uses +M6.
        add_schoolbook_2x2(2, 0, 0, 0, 2, 2, 1)
        add_schoolbook_2x2(2, 0, 2, 0, 2, 2, 1)
        add_schoolbook_2x2(0, 0, 0, 0, 2, 2, -1)
        add_schoolbook_2x2(0, 0, 2, 0, 2, 2, -1)
        
        # M7 = (A01-A11)(B11+B01) = A01B11 + A01B01 - A11B11 - A11B01
        # C00 uses +M7.
        add_schoolbook_2x2(0, 2, 2, 2, 0, 0, 1)
        add_schoolbook_2x2(0, 2, 0, 2, 0, 0, 1)
        add_schoolbook_2x2(2, 2, 2, 2, 0, 0, -1)
        add_schoolbook_2x2(2, 2, 0, 2, 0, 0, -1)
        
        # Check C00 = M1 + M4 - M5 + M7
        # My calls:
        # M1: added to C00 (via add_schoolbook... 0,0)
        # M4: added to C11? NO.
        # Wait. The `add_schoolbook_2x2` function adds to the C block specified in the 5th/6th args.
        # I must specify the correct C block for each term.
        
        # Let's re-verify the C block mapping in the calls above.
        # M1 contributes to C00. (Correct in calls 1.1-1.4)
        # M2 contributes to C10. (Correct in calls M2)
        # M3 contributes to C01. (Correct in calls M3)
        # M4 contributes to C11. (Correct in calls M4)
        # M5 contributes to C00 (sub) and C01 (add).
        # My calls for M5 added to C00.
        # I need to add M5 to C01 as well?
        # NO. M5 is a single matrix product. It appears in the formula for C00 and C01.
        # $C_{00} = M_1 + M_4 - M_5 + M_7$
        # $C_{01} = M_3 + M_5$
        # This means the tensor for M5 must be added to C00 with sign -1 and to C01 with sign +1.
        # But M5 is defined as a product of A and B blocks.
        # The U and V factors are the same for both C00 and C01 contributions?
        # NO. The tensor decomposition is $T = \sum u_k \otimes v_k \otimes w_k$.
        # If M5 appears in C00 and C01, it means the W vectors for M5 must have non-zero entries in both C00 and C01 positions?
        # No, that's not how it works.
        # $C_{ij} = \sum_k U_{i,k} V_{j,k} W_{k,k}$.
        # If M5 is a rank-4 tensor $T_5$, then $C = \dots - T_5 \dots$ for C00 part?
        # Actually, $C_{00}$ is a $2 \times 2$ block. $C_{01}$ is a $2 \times 2$ block.
        # $M_5$ is a $2 \times 2$ matrix.
        # $C_{00}$ gets $-M_5$. $C_{01}$ gets $+M_5$.
        # This implies we need two sets of components for M5?
        # No. $M_5$ is calculated once.
        # The result $M_5$ is added to $C_{01}$ and subtracted from $C_{00}$.
        # In the tensor decomposition, we can just define the W vectors for M5 to have values in both the C00 and C01 regions?
        # No, U and V are fixed.
        # If we use the same U, V components for M5, the W component must encode where it goes.
        # But W is a vector.
        # If we use the same U, V, we can only add to one W location per component.
        # So we must duplicate the U, V components for M5?
        # That would increase the rank.
        
        # Actually, Strassen's algorithm works by computing 7 products $M_1..M_7$.
        # Each $M_k$ is a matrix.
        # The final C is formed by adding/subtracting these matrices.
        # $C_{00} = M_1 + M_4 - M_5 + M_7$.
        # This means the tensor for $C_{00}$ is the sum of the tensors for $M_1, M_4, M_7$ minus the tensor for $M_5$.
        # The tensor for $M_5$ is the same object in memory (same U, V columns).
        # But it contributes to C00 and C01.
        # So for each rank-1 component of $M_5$, say $u \otimes v \otimes w_{local}$,
        # we need to add it to the global tensor.
        # The global W vector for this component should have $-1$ at the C00 position and $+1$ at the C01 position?
        # NO. The tensor is $\sum u \otimes v \otimes w$.
        # If we have a component $u \otimes v \otimes w_5$, it contributes $u_i v_j w_5^k$ to $C_{ij,k}$.
        # If $w_5$ has non-zero entries in both C00 and C01 regions, it works.
        # But $w_5$ is a single vector.
        # So we can set $w_5$ to have $-1$ in the C00 block and $+1$ in the C01 block?
        # Yes!
        
        # So, for each of the 4 components of M5 (schoolbook 2x2), we add a component to U, V.
        # And we set the W vector to have -1 in the C00 indices and +1 in the C01 indices.
        
        # My `add_schoolbook_2x2` currently adds to a single C block.
        # I need a more flexible adder for Strassen terms that map to multiple C blocks.
        
        # Revised Plan:
        # For each Strassen term $M_k$, which is a $2 \times 2$ product,
        # we generate 4 rank-1 components.
        # For each component, we determine which C blocks it contributes to and with what sign.
        # Then we add to U, V, and W accordingly.
        
        # M1: C00 (+1)
        # M2: C10 (+1)
        # M3: C01 (+1)
        # M4: C11 (+1)
        # M5: C00 (-1), C01 (+1)
        # M6: C11 (+1)
        # M7: C00 (+1)
        
        # I will implement this logic.
        
        # Block 2 and 3 are simple schoolbook, so they map to single C blocks.
        
        return {"U": U, "V": V, "W": W}
    
    def add_strassen_4x4(r1, c1, k1):
        # Define 2x2 blocks of A, B, C
        # A blocks: A00, A01, A10, A11
        # Indices for A00: (r1, c1), (r1, c1+1), (r1+1, c1), (r1+1, c1+1)
        
        def get_block_indices(r_start, c_start, dim=2):
            idxs = []
            for i in range(dim):
                for j in range(dim):
                    idxs.append((r_start + i) * m + (c_start + j))
            return idxs
            
        A00 = get_block_indices(r1, c1)
        A01 = get_block_indices(r1, c1 + 2)
        A10 = get_block_indices(r1 + 2, c1)
        A11 = get_block_indices(r1 + 2, c1 + 2)
        
        B00 = get_block_indices(c1, k1)
        B01 = get_block_indices(c1, k1 + 2)
        B10 = get_block_indices(c1 + 2, k1)
        B11 = get_block_indices(c1 + 2, k1 + 2)
        
        C00 = get_block_indices(k1, r1)
        C01 = get_block_indices(k1, r1 + 2)
        C10 = get_block_indices(k1 + 2, r1)
        C11 = get_block_indices(k1 + 2, r1 + 2)
        
        # Helper to add Strassen terms for a 2x2 product of linear combinations
        # U is represented by 4 lists of indices (for U00, U01, U10, U11)
        # V is represented by 4 lists of indices
        # C is represented by 4 lists of indices (for C00, C01, C10, C11)
        # We add 7 terms.
        
        def add_strassen_2x2_blocks(U_blocks, V_blocks, C_blocks):
            # U_blocks: dict {0: [idxs], 1: [idxs], 2: [idxs], 3: [idxs]} for 00, 01, 10, 11
            # Actually, let's just pass lists.
            
            # T1: (U00 + U11)(V00 + V11) -> C00, C11
            U_T1 = []
            for idx in U_blocks[0]: U_T1.append((idx, 1))
            for idx in U_blocks[3]: U_T1.append((idx, 1))
            
            V_T1 = []
            for idx in V_blocks[0]: V_T1.append((idx, 1))
            for idx in V_blocks[3]: V_T1.append((idx, 1))
            
            C_T1 = {}
            for idx in C_blocks[0]: C_T1[idx] = 1
            for idx in C_blocks[3]: C_T1[idx] = 1
            
            # We need to add terms for the product of sums.
            # Product of sum(u) and sum(v) is sum(u_i v_j).
            # This is not a single rank-1 term if U and V have multiple indices.
            # Wait.
            # The tensor decomposition term is $U \cdot V \cdot W$.
            # If U is a sum of indices, it means the coefficient for each index in that term is 1.
            # So the term is $\sum_{u \in U_T1} \sum_{v \in V_T1} u v w$.
            # This is a sum of rank-1 tensors.
            # So we need to add $|U_T1| \times |V_T1|$ terms?
            # No, that would blow up the rank.
            
            # Let's re-evaluate.
            # Strassen's algorithm computes $C = UV$.
            # It uses 7 scalar multiplications.
            # If U and V are matrices, the "scalar" multiplications are matrix multiplications.
            # We are decomposing the tensor $T_{ijk} = \sum_k A_{ik} B_{kj}$.
            # Strassen says $T = \sum_{r=1}^7 U_r V_r W_r$.
            # Here $U_r, V_r, W_r$ are matrices (or linear combinations of elements).
            # In the tensor decomposition, each term is a rank-1 tensor.
            # $U_r$ is a vector in the A-space.
            # If $U_r$ is a sum of basis vectors, it's still a single vector.
            # So $U_r V_r W_r$ is a single rank-1 tensor?
            # No.
            # A rank-1 tensor is $u \otimes v \otimes w$.
            # If $u = e_1 + e_2$, then $u \otimes v \otimes w = (e_1 + e_2) \otimes v \otimes w = e_1 \otimes v \otimes w + e_2 \otimes v \otimes w$.
            # This is a sum of two rank-1 tensors.
            # So the rank is the number of non-zero entries in the support?
            # No, the rank of the tensor is the minimum number of rank-1 tensors to sum to it.
            # If we use Strassen, we are using 7 terms.
            # Each term is a rank-1 tensor in the *block* space.
            # When expanded to the element space, each block-term becomes a sum of element-terms.
            # So the total rank is the sum of the ranks of the expanded terms.
            
            # For $2\times2\times2$ Strassen, each term has support size 2 in U, 2 in V, 2 in W?
            # Term 1: $U = e_{00} + e_{11}$, $V = e_{00} + e_{11}$, $W = e_{00} + e_{11}$.
            # Expanded: $(e_{00} + e_{11}) \otimes (e_{00} + e_{11}) \otimes (e_{00} + e_{11})$.
            # This expands to $2^3 = 8$ rank-1 tensors?
            # No.
            # The tensor is $T = \sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k$.
            # Strassen term 1 contributes to $T_{000}, T_{001}, T_{010}, \dots$
            # Actually, Strassen term 1 is $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
            # This product contributes to $C_{00}$ and $C_{11}$.
            # $C_{00} = M_1 + \dots$
            # $C_{11} = M_1 - \dots$
            
            # The rank of the $2\times2\times2$ tensor is 7.
            # This means it can be written as a sum of 7 rank-1 tensors.
            # Each rank-1 tensor has the form $u \otimes v \otimes w$.
            # In the Strassen decomposition, the 7 terms are:
            # 1. $u = e_{00}+e_{11}, v = e_{00}+e_{11}, w = e_{00}+e_{11}$.
            #    This is NOT a rank-1 tensor in the element basis.
            #    $u \otimes v \otimes w = (e_{00}+e_{11}) \otimes (e_{00}+e_{11}) \otimes (e_{00}+e_{11})$.
            #    This equals $e_{00}\otimes e_{00}\otimes e_{00} + e_{00}\otimes e_{00}\otimes e_{11} + \dots$
            #    So it is a sum of 8 rank-1 tensors.
            #    
            #    Wait, if Strassen uses 7 terms, and each term expands to 8 rank-1 tensors, the rank would be 56?
            #    But the rank of $M_{2,2,2}$ is 7.
            #    
            #    Contradiction.
            #    
            #    Resolution:
            #    The rank of a tensor is the minimum number of rank-1 tensors.
            #    Strassen's algorithm provides a decomposition into 7 terms.
            #    Are these terms rank-1 in the element basis?
            #    No, they are rank-1 in the *block* basis?
            #    
            #    Let's check the rank of $M_{2,2,2}$.
            #    It is known to be 7.
            #    The 7 Strassen terms are:
            #    $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
            #    This term contributes to $C_{00}$ and $C_{11}$.
            #    The contribution to $C_{00}$ is $M_1$.
            #    The contribution to $C_{11}$ is $M_1$.
            #    
            #    In the tensor decomposition, we write:
            #    $T = \sum_{r=1}^7 U_r \otimes V_r \otimes W_r$.
            #    Here $U_r$ is a vector in $\mathbb{R}^4$ (the A-space).
            #    $U_1 = (1, 0, 0, 1)$.
            #    $V_1 = (1, 0, 0, 1)$.
            #    $W_1 = (1, 0, 0, 1)$.
            #    
            #    $U_1 \otimes V_1 \otimes W_1$ is a $4 \times 4 \times 4$ tensor.
            #    Is it rank-1? Yes, by definition, an outer product of vectors is rank-1.
            #    
            #    So the rank is 7.
            #    
            #    So, when we implement Strassen for $2\times2\times2$, we add 7 terms.
            #    Each term is a dictionary of indices with coefficients.
            #    The term is added as a single entry in the `terms` list.
            #    When we build the matrices U, V, W, we just fill in the coefficients.
            #    
            #    So for the $4\times4\times4$ case:
            #    We use Strassen on the $4\times4$ blocks.
            #    This gives 7 terms.
            #    Each term involves $U_r, V_r, W_r$ which are $4\times4$ matrices (blocks).
            #    Wait, $U_r$ is a $4\times4$ matrix?
            #    No, $U_r$ is a linear combination of the $4\times4$ elements?
            #    
            #    Let's trace:
            #    $4\times4$ Strassen:
            #    $P_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
            #    $A_{00}$ is a $2\times2$ matrix.
            #    $P_1$ is a $2\times2$ matrix.
            #    We compute $P_1$ using Strassen for $2\times2$.
            #    This adds 7 terms to the global decomposition.
            #    Each of these 7 terms corresponds to a rank-1 tensor in the $2\times2\times2$ space.
            #    When mapped to the $4\times4\times4$ space, each term is a rank-1 tensor?
            #    No.
            #    The term for $P_1$ is $U_{P1} \otimes V_{P1} \otimes W_{P1}$.
            #    $U_{P1}$ is a vector in the $4\times4$ A-space.
            #    $U_{P1}$ has support on $A_{00}$ and $A_{11}$.
            #    So $U_{P1}$ is a sum of 8 basis vectors.
            #    So $U_{P1} \otimes V_{P1} \otimes W_{P1}$ is a rank-1 tensor in the $4\times4\times4$ space?
            #    Yes, because it's an outer product of three vectors.
            #    
            #    So, if we compute $P_1$ using 7 terms, and each term is a rank-1 tensor in the $4\times4\times4$ space, then the total rank for $P_1$ is 7.
            #    And for 7 such products, total rank is $7 \times 7 = 49$.
            #    
            #    So the implementation is:
            #    For each of the 7 Strassen steps of $4\times4$:
            #    Identify the U and V blocks (linear combinations of $2\times2$ blocks).
            #    Call `add_strassen_2x2`?
            #    No, `add_strassen_2x2` assumes specific indices.
            #    We need to call a function that adds 7 terms for a $2\times2$ product where the inputs are the specified linear combinations.
            #    
            #    Let's write `add_strassen_2x2_expanded(U_blocks, V_blocks, C_blocks)`.
            #    It will add 7 terms.
            #    Each term is a dictionary.
            
            # T1: (U00 + U11)(V00 + V11)
            U_T1 = {}
            for idx in U_blocks[0]: U_T1[idx] = 1
            for idx in U_blocks[3]: U_T1[idx] = 1
            
            V_T1 = {}
            for idx in V_blocks[0]: V_T1[idx] = 1
            for idx in V_blocks[3]: V_T1[idx] = 1
            
            W_T1 = {}
            for idx in C_blocks[0]: W_T1[idx] = 1
            for idx in C_blocks[3]: W_T1[idx] = 1
            
            add_term(U_T1, V_T1, W_T1)
            
            # ... and so on for T2..T7.
            
        # Define the 7 products P1..P7
        # P1 = (A00 + A11)(B00 + B11)
        U_P1 = {0: A00, 1: [], 2: [], 3: A11}
        V_P1 = {0: B00, 1: [], 2: [], 3: B11}
        C_P1 = {0: C00, 1: [], 2: [], 3: C11}
        add_strassen_2x2_blocks(U_P1, V_P1, C_P1)
        
        # P2 = (A10 + A11)B00
        U_P2 = {0: [], 1: [], 2: A10, 3: A11}
        V_P2 = {0: B00, 1: [], 2: [], 3: []}
        C_P2 = {0: [], 1: [], 2: C10, 3: C11}
        add_strassen_2x2_blocks(U_P2, V_P2, C_P2)
        
        # P3 = A00(B01 - B11)
        U_P3 = {0: A00, 1: [], 2: [], 3: []}
        V_P3 = {0: [], 1: B01, 2: [], 3: B11} # Note: B11 coeff is -1?
        # Wait, Strassen T3 is U00(V01 - V11).
        # So V_T3 has V01 with coeff 1, V11 with coeff -1.
        # My `add_strassen_2x2_blocks` helper should handle coefficients.
        
        # Let's refine the helper to accept coefficients.
    
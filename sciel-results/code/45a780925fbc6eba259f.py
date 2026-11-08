    def construct():
        n, m, p = 4, 4, 7
        rank = 0
        U = []
        V = []
        W = []
        
        # Helper to add a term
        def add_term(u_dict, v_dict, w_dict):
            # u_dict: {(i,j): val}
            # v_dict: {(j,k): val}
            # w_dict: {(k,i): val}
            pass # Will handle in main loop by appending to lists
            
        # Part 1: Strassen for 4x4x4 (cols 0-3 of B)
        # We generate 49 terms.
        
        # Define 2x2 Strassen terms
        # Each term is (u_vec, v_vec, w_vec) for 2x2
        # u_vec is length 4 (i,j in 0..1)
        # v_vec is length 4
        # w_vec is length 4
        
        # We will store the 7 terms for 2x2
        strassen_2x2 = []
        
        # P1
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[3]=1
        strassen_2x2.append((u,v,w))
        
        # P2
        u = [0]*4; u[2]=1; u[3]=1 # A10, A11
        v = [0]*4; v[0]=1 # B00
        w = [0]*4; w[2]=1; w[3]=1 # C10, C11
        strassen_2x2.append((u,v,w))
        
        # P3
        u = [0]*4; u[0]=1 # A00
        v = [0]*4; v[1]=1; v[3]=-1 # B01, -B11
        w = [0]*4; w[1]=1; w[3]=1 # C01, C11
        strassen_2x2.append((u,v,w))
        
        # P4
        u = [0]*4; u[3]=1 # A11
        v = [0]*4; v[2]=1; v[0]=-1 # B10, -B00
        w = [0]*4; w[2]=1; w[0]=1 # C10, C00
        strassen_2x2.append((u,v,w))
        
        # P5
        u = [0]*4; u[0]=1; u[1]=1 # A00, A01
        v = [0]*4; v[3]=1 # B11
        w = [0]*4; w[0]=-1; w[1]=1 # -C00, C01
        strassen_2x2.append((u,v,w))
        
        # P6
        u = [0]*4; u[2]=1; u[0]=-1 # A10, -A00
        v = [0]*4; v[0]=1; v[1]=1 # B00, B01
        w = [0]*4; w[3]=1 # C11
        strassen_2x2.append((u,v,w))
        
        # P7
        u = [0]*4; u[1]=1; u[3]=-1 # A01, -A11
        v = [0]*4; v[2]=1; v[3]=1 # B10, B11
        w = [0]*4; w[0]=1; w[1]=-1 # C00, -C01
        strassen_2x2.append((u,v,w))
        
        # Now map to 4x4
        # Top level 7 terms.
        # For each top term t (0..6):
        #   It involves specific 2x2 blocks of A and B.
        #   We need to map the 2x2 local indices to 4x4 global indices.
        
        # This mapping is complex to hardcode manually.
        # Alternative: Just use the known rank 49 decomposition?
        # Or, simpler: Use Strassen for 2x2x2 and apply it to 4 disjoint 2x2x2 blocks?
        # No, Strassen mixes the blocks.
        
        # Given the time, maybe I should just stick to a simpler improvement?
        # What if I just use Strassen for the first 2x2x2 blocks and schoolbook for the rest?
        # 4x4x4 can be seen as 4 blocks of 2x2x2? No, it's one 4x4x4.
        
        # Let's try a different "new approach":
        # Use the fact that rank(4x4x4) <= 49.
        # I can just hardcode the 49 terms for 4x4x4 if I find them online or generate them.
        # But generating them correctly is risky.
        
        # Let's try a simpler hybrid:
        # Split B into two 4x4 blocks? No, B is 4x7.
        # Split B into 4x4 and 4x3.
        # For 4x4x4, use rank 49.
        # For 4x4x3, use rank 48.
        # Total 97.
        
        # I will implement the Strassen 4x4x4 by recursively calling the 2x2 logic.
        # I need a function `expand_strassen(local_u, local_v, local_w, block_offsets)`
        
        # Let's define the top-level 7 products' block mappings.
        # A blocks: A00, A01, A10, A11 (2x2 each)
        # B blocks: B00, B01, B10, B11
        
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B01-B11)
        # P4: A11(B10-B00)
        # P5: (A00+A01)B11
        # P6: (A10-A00)(B00+B01)
        # P7: (A01-A11)(B10+B11)
        
        # For each P, we have a linear combination of A blocks and B blocks.
        # We can represent this as a list of (coeff, block_idx) for U and V.
        # Then for each such combination, we expand using the 2x2 Strassen terms.
        
        # This is doable.
    
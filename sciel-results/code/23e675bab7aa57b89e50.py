        def construct():
            n, m, p = 3, 5, 7
            # Split m into m1=3, m2=2
            m1, m2 = 3, 2
            
            # Decomposition for A1 (3x3) * B1 (3x7)
            # Split B1 into 3x3, 3x3, 3x1
            decomp1_1 = schoolbook(3, 3, 3)
            decomp1_2 = schoolbook(3, 3, 3)
            decomp1_3 = schoolbook(3, 3, 1)
            
            # Decomposition for A2 (3x2) * B2 (2x7)
            # Split B2 into 2x3, 2x3, 2x1
            decomp2_1 = schoolbook(3, 2, 3)
            decomp2_2 = schoolbook(3, 2, 3)
            decomp2_3 = schoolbook(3, 2, 1)
            
            # Combine
            # Need to map the indices correctly.
            # The global indices for A are 0..14.
            # A1 uses indices 0..8 (3x3).
            # A2 uses indices 9..14 (3x2).
            # B1 uses indices 0..20 (3x7).
            # B2 uses indices 21..34 (2x7).
            # C uses indices 0..20 (7x3).
            
            # The schoolbook function should return U, V, W with local indices.
            # We need to embed them into the global space.
        
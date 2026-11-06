            # Initialize colors for n=50
            full_colors = []
            idx_seed = 0
            
            # Pre-allocate
            full_colors = [0] * m_target
            
            # Fill seed part
            for i in range(32):
                for j in range(i + 1, 32):
                    # Index in full_colors
                    idx_full = i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)
                    full_colors[idx_full] = seed_colors_32[idx_seed]
                    idx_seed += 1
            
            # Fill new parts
            # Edges between old (0-31) and new (32-49)
            for i in range(32):
                for j in range(32, n_target):
                    idx_full = i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)
                    # Copy pattern from a random old vertex
                    # Let's pick vertex 0 for simplicity, or random
                    src = random.randint(0, 31)
                    # We need color of (src, i) if src != i
                    # But we need to map (src, i) to seed index
                    # If src < i:
                    #   idx_seed_src = src * 31 - src*(src-1)//2 + (i - src - 1)
                    # If src > i:
                    #   idx_seed_src = i * 31 - i*(i-1)//2 + (src - i - 1)
                    if src < i:
                        s_idx = src * 31 - src * (src - 1) // 2 + (i - src - 1)
                    else:
                        s_idx = i * 31 - i * (i - 1) // 2 + (src - i - 1)
                    full_colors[idx_full] = seed_colors_32[s_idx]

            # Edges between new (32-49)
            for i in range(32, n_target):
                for j in range(i + 1, n_target):
                    idx_full = i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)
                    full_colors[idx_full] = random.randint(0, 3)

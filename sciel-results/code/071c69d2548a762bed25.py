            for local_a, c_x in x_sub.items():
                la_r = local_a // 2
                la_c = local_a % 2
                
                for block_idx, c_outer in x_def.items():
                    b_r = block_idx // 2
                    b_c = block_idx % 2
                    
                    i = 2 * b_r + la_r
                    j = 2 * b_c + la_c
                    u_idx = i * m + j
                    u[u_idx] += c_outer * c_x

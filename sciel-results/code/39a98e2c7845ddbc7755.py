<<<<<<< SEARCH
            # Expand U
            for b in range(4):
                if u_block[b] != 0:
                    for l in range(4):
                        if u_sub[l] != 0:
                            idx = b * 4 + l
                            U[idx] += u_block[b] * u_sub[l]
                            
            # Expand V
            for b in range(4):
                if v_block[b] != 0:
                    for l in range(4):
                        if v_sub[l] != 0:
                            idx = b * 4 + l
                            V[idx] += v_block[b] * v_sub[l]
                            
            # Expand W
            for b in range(4):
                if w_block[b] != 0:
                    for l in range(4):
                        if w_sub[l] != 0:
                            idx = b * 4 + l
                            W[idx] += w_block[b] * w_sub[l]
=======
            # Expand U
            # b = 2*B_i + B_j, l = 2*S_i + S_j
            # i = 2*B_i + S_i, j = 2*B_j + S_j
            # idx = i * 4 + j
            for b in range(4):
                if u_block[b] != 0:
                    B_i = b // 2
                    B_j = b % 2
                    for l in range(4):
                        if u_sub[l] != 0:
                            S_i = l // 2
                            S_j = l % 2
                            i = 2 * B_i + S_i
                            j = 2 * B_j + S_j
                            idx = i * 4 + j
                            U[idx] += u_block[b] * u_sub[l]
                            
            # Expand V
            # b = 2*B_j + B_k, l = 2*S_j + S_k
            # j = 2*B_j + S_j, k = 2*B_k + S_k
            # idx = j * 4 + k
            for b in range(4):
                if v_block[b] != 0:
                    B_j = b // 2
                    B_k = b % 2
                    for l in range(4):
                        if v_sub[l] != 0:
                            S_j = l // 2
                            S_k = l % 2
                            j = 2 * B_j + S_j
                            k = 2 * B_k + S_k
                            idx = j * 4 + k
                            V[idx] += v_block[b] * v_sub[l]
                            
            # Expand W
            # b = 2*B_i + B_k, l = 2*S_i + S_k
            # i = 2*B_i + S_i, k = 2*B_k + S_k
            # idx = k * 4 + i
            for b in range(4):
                if w_block[b] != 0:
                    B_i = b // 2
                    B_k = b % 2
                    for l in range(4):
                        if w_sub[l] != 0:
                            S_i = l // 2
                            S_k = l % 2
                            i = 2 * B_i + S_i
                            k = 2 * B_k + S_k
                            idx = k * 4 + i
                            W[idx] += w_block[b] * w_sub[l]
>>>>>>> REPLACE

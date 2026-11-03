                    # Map local to global
                    def map_u(local_i, local_j):
                        return (I_list[local_i], J_list[local_j])
                    def map_v(local_j, local_k):
                        return (J_list[local_j], K_list[local_k])
                    def map_w(local_k, local_i):
                        return (K_list[local_k], I_list[local_i])
                    
                    # P1
                    u1 = {map_u(0,0): 1, map_u(1,1): 1}
                    v1 = {map_v(0,0): 1, map_v(1,1): 1}
                    w1 = {map_w(0,0): 1, map_w(1,1): 1}
                    terms.append((u1, v1, w1))
                    
                    # P2
                    u2 = {map_u(1,0): 1, map_u(1,1): 1}
                    v2 = {map_v(0,0): 1}
                    w2 = {map_w(0,1): 1, map_w(1,1): -1} # C10 is (k=1, i=0)? No.
                    # C10 means row 1, col 0. i=1, k=0.
                    # My W index is (k, i).
                    # C10 -> k=0, i=1. So map_w(0, 1).
                    # C11 -> k=1, i=1. So map_w(1, 1).
                    # Strassen: C10 = P2 + P4. C11 = P1 - P2 + ...
                    # So W for P2 should be:
                    # map_w(0,1) += 1 (for C10)
                    # map_w(1,1) += -1 (for C11)
                    w2 = {map_w(0,1): 1, map_w(1,1): -1}
                    terms.append((u2, v2, w2))
                    
                    # P3
                    u3 = {map_u(0,0): 1}
                    v3 = {map_v(0,1): 1, map_v(1,1): -1}
                    # C01 = P3 + P5. C11 = ... + P3.
                    # C01 -> k=1, i=0. map_w(1,0).
                    # C11 -> k=1, i=1. map_w(1,1).
                    w3 = {map_w(1,0): 1, map_w(1,1): 1}
                    terms.append((u3, v3, w3))
                    
                    # P4
                    u4 = {map_u(1,1): 1}
                    v4 = {map_v(1,0): 1, map_v(0,0): -1}
                    # C00 = ... + P4. C10 = P2 + P4.
                    # C00 -> k=0, i=0. map_w(0,0).
                    # C10 -> k=0, i=1. map_w(0,1).
                    w4 = {map_w(0,0): 1, map_w(0,1): 1}
                    terms.append((u4, v4, w4))
                    
                    # P5
                    u5 = {map_u(0,0): 1, map_u(0,1): 1}
                    v5 = {map_v(1,1): 1}
                    # C00 = ... - P5. C01 = P3 + P5.
                    # C00 -> k=0, i=0. map_w(0,0).
                    # C01 -> k=1, i=0. map_w(1,0).
                    w5 = {map_w(0,0): -1, map_w(1,0): 1}
                    terms.append((u5, v5, w5))
                    
                    # P6
                    u6 = {map_u(1,0): 1, map_u(0,0): -1}
                    v6 = {map_v(0,0): 1, map_v(0,1): 1}
                    # C11 = ... + P6.
                    # C11 -> k=1, i=1. map_w(1,1).
                    w6 = {map_w(1,1): 1}
                    terms.append((u6, v6, w6))
                    
                    # P7
                    u7 = {map_u(0,1): 1, map_u(1,1): -1}
                    v7 = {map_v(1,0): 1, map_v(1,1): 1}
                    # C00 = ... + P7.
                    # C00 -> k=0, i=0. map_w(0,0).
                    w7 = {map_w(0,0): 1}
                    terms.append((u7, v7, w7))

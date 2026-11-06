        def greedy_mis(candidates):
            size = 0
            mask = 0
            while candidates:
                # Pick node with min degree in remaining subgraph
                # to maximize the size of the independent set found
                min_deg = m + 1
                pivot_bit = 0
                pivot_v = -1
                temp_cand = candidates
                while temp_cand:
                    v_bit = temp_cand & -temp_cand
                    v = v_bit.bit_length() - 1
                    deg = (candidates & adj_conf[v]).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        pivot_bit = v_bit
                        pivot_v = v
                    temp_cand &= temp_cand - 1
                
                mask |= pivot_bit
                size += 1
                candidates &= ~(adj_conf[pivot_v] | pivot_bit)
            return size, mask

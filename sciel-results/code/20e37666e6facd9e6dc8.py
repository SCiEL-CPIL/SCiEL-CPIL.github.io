            for c_idx in affected:
                cyc = cycles[c_idx]
                c0 = colors[cyc[0]]
                c1 = colors[cyc[1]]
                c2 = colors[cyc[2]]
                c3 = colors[cyc[3]]
                
                # Determine new colors
                if cyc[0] == e:
                    n0, n1, n2, n3 = new_c, c1, c2, c3
                elif cyc[1] == e:
                    n0, n1, n2, n3 = c0, new_c, c2, c3
                elif cyc[2] == e:
                    n0, n1, n2, n3 = c0, c1, new_c, c3
                else:
                    n0, n1, n2, n3 = c0, c1, c2, new_c
                    
                # Check distinctness
                is_distinct = (n0 != n1 and n0 != n2 and n0 != n3 and n1 != n2 and n1 != n3 and n2 != n3)
                
                if cycle_valid[c_idx]:
                    if is_distinct:
                        delta_valid -= 1
                else:
                    if not is_distinct:
                        delta_valid += 1

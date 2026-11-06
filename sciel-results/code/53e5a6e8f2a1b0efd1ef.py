                    # Calculate distinct colors for k4_i with e_idx having new_color
                    c_set = set()
                    for e in k4_edges[k4_i]:
                        if e == e_idx:
                            c_set.add(new_color)
                        else:
                            c_set.add(colors[e])
                    now_bad = len(c_set) < 5

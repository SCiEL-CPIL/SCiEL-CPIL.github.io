                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    idx = get_edge_idx(u, v)
                    available_mask &= ~(1 << idx)

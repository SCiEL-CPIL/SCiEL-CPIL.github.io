                for t_idx in block_t_indices[b]:
                    coverage[t_idx] -= 1
                    if coverage[t_idx] == 0:
                        uncovered_count += 1

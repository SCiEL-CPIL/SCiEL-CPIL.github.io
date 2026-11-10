                    if not improved:
                        # Revert the swap
                        selected[idx] = (m, r_old)
                        if len(selected) > len(selected) - 1:  # if we added one
                            selected.pop()

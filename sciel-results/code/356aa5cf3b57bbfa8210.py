        if min_size == 0:
            # Must skip
            unassigned.remove(min_idx)
            solve(assigned, domains, unassigned, current_count)
            unassigned.add(min_idx)
            return

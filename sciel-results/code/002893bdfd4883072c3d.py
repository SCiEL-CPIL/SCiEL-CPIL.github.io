    def try_swap(old_c, new_c):
        nonlocal rem_count
        neigh_new_c = neigh_sets[new_c]
        for u in neigh[old_c]:
            if counts[u] == 1:
                if u not in neigh_new_c:
                    return False
        # Perform swap
        idx = code.index(old_c)
        code[idx] = new_c
        code_set.remove(old_c)
        code_set.add(new_c)
        
        # Update counts and remaining
        for u in neigh[old_c]:
            counts[u] -= 1
            # counts[u] cannot become 0 due to validity check
        for u in neigh[new_c]:
            old_count = counts[u]
            counts[u] += 1
            if old_count == 0:
                remaining[u] = False
                rem_count -= 1
        return True
   
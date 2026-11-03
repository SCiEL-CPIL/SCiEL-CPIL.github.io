def _greedy_sidon_deterministic(n):
    current_set = []
    used_sums = set()
    for i in range(n):
        valid = True
        for s in current_set:
            if (s + i) in used_sums:
                valid = False
                break
        if not valid:
            continue
        if (i + i) in used_sums:
            continue
        
        current_set.append(i)
        for s in current_set[:-1]:
            used_sums.add(s + i)
        used_sums.add(i + i)
        
    return current_set

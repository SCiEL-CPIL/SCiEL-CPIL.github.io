    def solve_once():
        in_set = [False] * (N + 1)
        for i in range(1, N + 1):
            in_set[i] = True
            
        conflict_count = [0] * (N + 1)
        for a, b, c in forbidden:
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
            
        total_conflicts = len(forbidden)
        
        while total_conflicts > 0:
            max_count = 0
            candidates = []
            for i in range(1, N + 1):
                if conflict_count[i] > max_count:
                    max_count = conflict_count[i]
                    candidates = [i]
                elif conflict_count[i] == max_count and max_count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            x = random.choice(candidates)
            in_set[x] = False
            conflict_count[x] = 0
            
            for y, z in conflicts[x]:
                if in_set[y] and in_set[z]:
                    total_conflicts -= 1
                    conflict_count[y] -= 1
                    conflict_count[z] -= 1

        unselected = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(unselected)
        
        changed = True
        while changed:
            changed = False
            for u in unselected:
                if not in_set[u]:
                    if can_add(u, in_set):
                        in_set[u] = True
                        changed = True
        
        return sum(in_set[1:]), in_set

    best_size = 0
    best_in_set = None
    
    while time.time() < deadline:
        size, in_set = solve_once()
        if size > best_size:
            best_size = size
            best_in_set = in_set
            
    if best_in_set is None:
        return [1]
        
    return [i for i in range(1, N + 1) if best_in_set[i]]

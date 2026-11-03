def find_max_sidon(n, max_size):
    best_set = []
    
    def backtrack(start, current, sums):
        nonlocal best_set
        if len(current) > len(best_set):
            best_set = current[:]
        if len(current) == max_size:
            return
        
        # Heuristic: try to place next element
        for i in range(start, n):
            # Check validity
            valid = True
            for x in current:
                if x + i in sums:
                    valid = False
                    break
            if not valid:
                continue
            if i + i in sums:
                continue
            
            # Add i
            current.append(i)
            new_sums = set()
            for x in current[:-1]:
                new_sums.add(x + i)
            new_sums.add(i + i)
            sums.update(new_sums)
            
            backtrack(i + 1, current, sums)
            
            sums.difference_update(new_sums)
            current.pop()
            
    backtrack(0, [], set())
    return best_set

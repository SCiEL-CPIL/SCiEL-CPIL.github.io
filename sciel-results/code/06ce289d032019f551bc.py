    import time
    import random

    def construct():
        start_time = time.time()
        deadline = start_time + 110
        
        fixed = [1, 2, 5, 11]
        best_set = list(fixed)
        best_size = len(fixed)
        
        # Pre-calculate initial differences
        init_diffs = set()
        for i in range(len(fixed)):
            for j in range(i + 1, len(fixed)):
                init_diffs.add(abs(fixed[i] - fixed[j]))
                
        # Pool of candidates
        all_candidates = [x for x in range(1, 285) if x not in fixed]
        
        # We perform randomized greedy restarts
        # Each restart tries to build a maximal Sidon set starting from 'fixed'
        
        while time.time() < deadline:
            current_set = set(fixed)
            diffs = set(init_diffs)
            
            # Shuffle candidates to explore different orderings
            candidates = all_candidates[:]
            random.shuffle(candidates)
            
            # Greedily add valid candidates
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Check if x can be added
                can_add = True
                for y in current_set:
                    d = abs(x - y)
                    if d in diffs:
                        can_add = False
                        break
                
                if can_add:
                    current_set.add(x)
                    # Update the set of used differences
                    for y in current_set:
                        if y != x:
                            diffs.add(abs(x - y))
                            
            # Update best solution if we found a larger set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
            # If we reached the known maximum possible size, we can stop
            if best_size == 20:
                break

        return best_set
    
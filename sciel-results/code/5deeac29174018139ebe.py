        # Phase B: Random Perturbation
        if not S:
            break
        x_to_remove = random.choice(list(S))
        S_before = set(S)
        S.remove(x_to_remove)
        
        # Find candidates to add
        candidates = [y for y in range(1, N + 1) if y not in S and can_add(y, S)]
        
        if candidates:
            y_to_add = random.choice(candidates)
            S.add(y_to_add)
            
            # Greedy Add rest
            for z in range(1, N + 1):
                if z not in S and can_add(z, S):
                    S.add(z)
        else:
            # No candidate, just revert
            S = S_before
            continue

        new_size = len(S)
        old_size = len(S_before)
        
        if new_size > old_size:
            if new_size > best_size:
                best_size = new_size
                best_S = set(S)
            neutral_count = 0
        elif new_size == old_size:
            neutral_count += 1
            if neutral_count > 50:
                S = S_before
                neutral_count = 0
            else:
                if random.random() < 0.2:
                    pass # Keep new
                else:
                    S = S_before
        else:
            S = S_before
            neutral_count = 0

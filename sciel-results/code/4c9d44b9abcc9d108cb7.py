    # ... inside construct() ...
    
    for U in top_candidates:
        if time.time() > deadline - 5:
            break
        
        current_U = list(U)
        current_score = evaluate(current_U)
        
        while time.time() < deadline - 5:
            improved = False
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                    
                move_type = random.choice(['add', 'remove', 'replace'])
                new_U = list(current_U)
                
                if move_type == 'add':
                    if len(new_U) >= max_n_local:
                        continue
                    val = random.randint(1, 200)
                    if val not in new_U:
                        new_U.append(val)
                        new_U = sorted(new_U)
                elif move_type == 'remove':
                    if len(new_U) <= 2:
                        continue
                    idx = random.choice([i for i, x in enumerate(new_U) if x != 0])
                    new_U.pop(idx)
                elif move_type == 'replace':
                    if len(new_U) <= 2:
                        continue
                    idx = random.choice([i for i, x in enumerate(new_U) if x != 0])
                    new_val = random.randint(1, 200)
                    if new_val not in new_U:
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
                        if len(new_U) != len(current_U):
                            continue
                
                new_score = evaluate(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_U)
            
            if not improved:
                break # Move to next candidate

    return best_set

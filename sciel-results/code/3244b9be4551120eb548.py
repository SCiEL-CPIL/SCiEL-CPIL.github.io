        is_valid = True
        if len(set(new_diffs)) != len(new_diffs):
            is_valid = False
        else:
            new_diff_set = set(new_diffs)
            for d in range(1, 31):
                if d not in new_diff_set:
                    is_valid = False
                    break
        
        if is_valid:
            new_score = new_max
            # Update best
            if new_max < best_val:
                best_val = new_max
                best_a = new_a
            # Update current
            if new_score < current_score:
                current_a = new_a
                current_diffs = new_diffs
                current_max = new_max
                current_score = new_score
            else:
                # SA accept
                delta = new_score - current_score
                if T > 0:
                    if delta > 100 * T: p = 0
                    else: p = 2.71828 ** (-delta / T)
                    if random.random() < p:
                        current_a = new_a
                        current_diffs = new_diffs
                        current_max = new_max
                        current_score = new_score
        else:
            # Calculate penalty score for SA
            if len(set(new_diffs)) != len(new_diffs):
                dup_count = len(new_diffs) - len(set(new_diffs))
                new_score = new_max + 1000 * dup_count
            else:
                new_diff_set = set(new_diffs)
                missing = 0
                for d in range(1, 31):
                    if d not in new_diff_set:
                        missing += 1
                new_score = new_max + 100 * missing
            
            # SA accept
            if new_score < current_score:
                current_a = new_a
                current_diffs = new_diffs
                current_max = new_max
                current_score = new_score
            else:
                delta = new_score - current_score
                if T > 0:
                    if delta > 100 * T: p = 0
                    else: p = 2.71828 ** (-delta / T)
                    if random.random() < p:
                        current_a = new_a
                        current_diffs = new_diffs
                        current_max = new_max
                        current_score = new_score

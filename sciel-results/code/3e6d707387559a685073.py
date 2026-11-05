        # Check validity: all new_diffs must not be in used_diffs (except those that were in old_diffs)
        # Remove old_diffs from used_diffs
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Check if new_diffs conflict
        valid = True
        for d in new_diffs:
            if d in used_diffs:
                valid = False
                break
        
        if valid:
            # Add new_diffs to used_diffs
            for d in new_diffs:
                used_diffs.add(d)
            
            new_scope = get_scope(current_rows)
            delta_scope = new_scope - current_scope
            
            # Acceptance
            accept = False
            if delta_scope < 0:
                accept = True
            else:
                if T > 0:
                    if delta_scope / T < 700:
                        accept = random.random() < (2.718281828 ** (-delta_scope / T))
                    else:
                        accept = False
        
        if accept:
            # ...
        else:
            # Revert changes
            # Remove new_diffs
            for d in new_diffs:
                used_diffs.discard(d)
            
            # Restore old val
            row[j_idx] = old_val
            
            # Restore old_diffs
            for d in old_diffs:
                used_diffs.add(d)
   
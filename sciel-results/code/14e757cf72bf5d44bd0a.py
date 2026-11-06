def construct():
    max_val = 256
    best_seq = []
    
    def dfs(seq, sums, last):
        nonlocal best_seq
        
        # Update best
        if len(seq) > len(best_seq):
            best_seq = list(seq)
            
        # Pruning
        if last == max_val:
            return
            
        # Max possible length from here:
        # We can add at most (256 - last) more numbers.
        if len(seq) + (256 - last) <= len(best_seq):
            return

        for x in range(last + 1, max_val + 1):
            # Check if x can be added
            # New sums are x, x+a_n, x+a_n+a_{n-1}, ...
            
            total = x
            new_sums = [total]
            
            if total in sums:
                continue
                
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                if total in sums:
                    break
            else:
                # No collision found in the loop
                # Add x
                for s in new_sums:
                    sums.add(s)
                    
                dfs(seq + [x], sums, x)
                
                # Restore
                for s in new_sums:
                    sums.remove(s)
                # Continue to next x
            # If break occurred, continue to next x

    dfs([], set(), 0)
    
    return best_seq

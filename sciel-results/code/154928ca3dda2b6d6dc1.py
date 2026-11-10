import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    best_seq = []
    
    # We use a list of available numbers. 
    # To avoid copying the list, we can pass a set of used numbers and iterate 1..256.
    # But checking `if x not in used_set` is O(1).
    # Iterating 256 times per node is fine.
    
    def dfs(current_seq, used_sums, used_numbers, suffix_sums):
        nonlocal best_seq, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning: If we can't beat the best, stop
        # Max possible length is len(current_seq) + (256 - len(used_numbers))
        if len(current_seq) + (256 - len(used_numbers)) <= len(best_seq):
            return

        # Update best if current is longer
        if len(current_seq) > len(best_seq):
            best_seq = list(current_seq)
            
        # If no numbers left, return
        if len(used_numbers) == 256:
            return

        # Try adding each available number
        # Order: We can try ascending, descending, or random.
        # Ascending is deterministic. Let's stick to it but backtracking handles the rest.
        # To potentially find longer sequences faster, maybe larger numbers are better?
        # Let's try ascending first as it's standard.
        
        for x in range(1, 257):
            if time.time() > deadline:
                return
            
            if x in used_numbers:
                continue
            
            # Calculate new sums
            # New sums are x and x + s for s in suffix_sums
            # Check for conflict
            conflict = False
            if x in used_sums:
                conflict = True
            else:
                for s in suffix_sums:
                    if (x + s) in used_sums:
                        conflict = True
                        break
            
            if conflict:
                continue
            
            # Valid move
            next_suffix_sums = [x] + [x + s for s in suffix_sums]
            next_used_sums = set(used_sums)
            next_used_sums.add(x)
            for s in suffix_sums:
                next_used_sums.add(x + s)
            
            next_used_numbers = set(used_numbers)
            next_used_numbers.add(x)
            
            dfs(current_seq + [x], next_used_sums, next_used_numbers, next_suffix_sums)
            
            # Backtrack: local variables handle it

    dfs([], set(), set(), [])
    
    return best_seq

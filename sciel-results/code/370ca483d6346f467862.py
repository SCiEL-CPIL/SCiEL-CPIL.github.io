def solve_scope(scope, n_rows, k, time_limit=5.0):
    rng = random.Random(int(time.time() * 1000) % 10000)
    start_time = time.time()
    rows = []
    
    def backtrack(row_idx, current_row, used_bits):
        if time.time() - start_time > time_limit:
            return False
        if row_idx == n_rows:
            return True
        if len(current_row) == k + 1:
            rows.append(current_row)
            if backtrack(row_idx + 1, [0], used_bits):
                return True
            rows.pop()
            return False
            
        last = current_row[-1]
        remaining = (k + 1) - len(current_row)
        min_x = last + 1
        max_x = scope - (remaining - 1)
        
        if min_x > max_x:
            return False
            
        # Generate valid candidates
        # To speed up, we can break early if we find too many? No, we need to try them.
        # But we can randomize the order.
        candidates = []
        for x in range(min_x, max_x + 1):
            valid = True
            for a in current_row:
                if (used_bits >> (x - a)) & 1:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return False
            
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        for x in candidates:
            new_used = used_bits
            for a in current_row:
                new_used |= (1 << (x - a))
            if backtrack(row_idx, current_row + [x], new_used):
                return True
                
        return False

    return backtrack(0, [0], 0)

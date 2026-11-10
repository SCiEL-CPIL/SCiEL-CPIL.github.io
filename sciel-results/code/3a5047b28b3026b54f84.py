def construct(n_rows, k):
    def solve(M):
        # used is an integer bitset
        # rows is built in the return value
        def backtrack(row_idx, current_row, used):
            if len(current_row) == k + 1:
                if row_idx == n_rows - 1:
                    return [current_row]
                else:
                    # Recurse for next row
                    sub_result = backtrack(row_idx + 1, [0], used)
                    if sub_result is not None:
                        return [current_row] + sub_result
                    return None
            
            current_len = len(current_row)
            rem = k + 1 - current_len
            needed = rem * current_len + rem * (rem - 1) // 2
            
            if (M - used.bit_count()) < needed:
                return None
            
            start_val = current_row[-1] + 1
            # We can also bound end_val more tightly?
            # The largest element in the row is x.
            # The differences must be <= M.
            # So x - 0 <= M => x <= M.
            
            for x in range(start_val, M + 1):
                valid = True
                new_used = used
                for y in current_row:
                    d = x - y
                    if used & (1 << d):
                        valid = False
                        break
                    new_used |= (1 << d)
                
                if not valid:
                    continue
                
                result = backtrack(row_idx, current_row + [x], new_used)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(0, [0], 0)

    # Search for minimal M
    # The counting bound is 112.
    # We can start from 112.
    for M in range(112, 150): # Safe upper bound
        res = solve(M)
        if res is not None:
            return res
    
    # Fallback
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

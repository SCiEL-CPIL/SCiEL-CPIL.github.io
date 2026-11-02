def construct(n_rows, k):
    import sys
    import time
    from itertools import combinations

    sys.setrecursionlimit(100000)
    start_time = time.time()
    time_limit = 115.0

    # Step 1: Greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    # Step 2: Improved Search with Bitsets and Symmetry Breaking
    def solve_with_max(M, n_rows, k, time_limit):
        if M < k:
            return None
        
        found = [False]
        solution = [None]
        
        # Bitset for global used differences
        # We will search for rows in lexicographic order to break symmetry
        
        def dfs(row_idx, prev_row, used_mask):
            if found[0] or time.time() - start_time > time_limit:
                return
            
            if row_idx == n_rows:
                found[0] = True
                return

            # Generate next row
            # We need a row r such that:
            # 1. r is a Sidon set in [0, M]
            # 2. r is lex greater than prev_row
            # 3. r's diffs are disjoint from used_mask
            
            def gen_next_row():
                # Recursive generator for a single row
                def gen_row(prefix, current_mask):
                    if len(prefix) == k + 1:
                        # Check lex order
                        if prev_row is None or prefix > prev_row:
                            yield prefix
                        return
                    
                    last = prefix[-1]
                    
                    # Determine lower bound for next element to satisfy lex order
                    lower_bound = last + 1
                    if prev_row is not None:
                        if len(prefix) < len(prev_row):
                            if prefix < prev_row[:len(prefix)]:
                                return # Prune
                            elif prefix == prev_row[:len(prefix)]:
                                lower_bound = max(lower_bound, prev_row[len(prefix)] + 1)
                        # If prefix > prev_row[:len(prefix)], lower_bound stays last+1
                    
                    for x in range(lower_bound, M + 1):
                        # Check differences
                        valid = True
                        new_mask = 0
                        for a in prefix:
                            d = x - a
                            bit = 1 << d
                            if (used_mask | current_mask) & bit:
                                valid = False
                                break
                            new_mask |= bit
                        if valid:
                            yield from gen_row(prefix + [x], current_mask | new_mask)
                
                yield from gen_row([0], 0)

            # Iterate through candidate rows
            # To improve performance, we can try the first few candidates
            for row in gen_next_row():
                # Calculate mask for this row
                row_mask = 0
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        row_mask |= (1 << d)
                
                # Recurse
                dfs(row_idx + 1, row, used_mask | row_mask)
                
                if found[0]:
                    return

        dfs(0, None, 0)
        
        if found[0]:
            # We need to reconstruct the rows. 
            # The current dfs just returns True. 
            # We need to store the path.
            # Let's modify dfs to store the solution.
            pass
            
        return None # Placeholder, need to fix reconstruction

    # The above reconstruction is tricky. 
    # Let's use a list to store the path.
    
    # Revised solve_with_max
    def solve_with_max_v2(M, n_rows, k, time_limit):
        if M < k:
            return None
        
        found = [False]
        solution = []
        
        def dfs(row_idx, prev_row, used_mask):
            if found[0] or time.time() - start_time > time_limit:
                return
            
            if row_idx == n_rows:
                found[0] = True
                return

            def gen_next_row():
                def gen_row(prefix, current_mask):
                    if len(prefix) == k + 1:
                        if prev_row is None or prefix > prev_row:
                            yield prefix
                        return
                    
                    last = prefix[-1]
                    lower_bound = last + 1
                    if prev_row is not None:
                        if len(prefix) < len(prev_row):
                            if prefix < prev_row[:len(prefix)]:
                                return
                            elif prefix == prev_row[:len(prefix)]:
                                lower_bound = max(lower_bound, prev_row[len(prefix)] + 1)
                    
                    for x in range(lower_bound, M + 1):
                        valid = True
                        new_mask = 0
                        for a in prefix:
                            d = x - a
                            bit = 1 << d
                            if (used_mask | current_mask) & bit:
                                valid = False
                                break
                            new_mask |= bit
                        if valid:
                            yield from gen_row(prefix + [x], current_mask | new_mask)
                
                yield from gen_row([0], 0)

            for row in gen_next_row():
                row_mask = 0
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        row_mask |= (1 << d)
                
                solution.append(row)
                dfs(row_idx + 1, row, used_mask | row_mask)
                
                if found[0]:
                    return
                solution.pop()

        dfs(0, None, 0)
        
        if found[0]:
            return [list(r) for r in solution]
        return None

    # Try to find better solutions
    for target_max in range(best_max - 1, max(best_max - 20, k), -1):
        if time.time() - start_time > time_limit:
            break
        result = solve_with_max_v2(target_max, n_rows, k, time_limit)
        if result is not None:
            best_rows = result
            best_max = target_max
            break

    return best_rows

import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    # Lower bound estimation
    # Counting bound: n * C(k+1, 2) <= M
    lower_bound = n_rows * (k + 1) * k // 2
    
    def solve_for_M(M):
        """
        Finds a valid set of n_rows rows with max element <= M.
        Returns the list of rows or None.
        Uses DFS with lexicographic ordering to break symmetry.
        """
        current_rows = []
        used_diffs = set()
        
        def is_row_valid(row):
            # Check internal diffs
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
            if diffs & used_diffs:
                return False
            return True

        def dfs(row_idx, pos, current_row, prev_row):
            # row_idx: index of the row currently being built
            # pos: index in current_row currently being filled
            # current_row: list of elements filled so far in current_row
            # prev_row: the previous row (lexicographically smaller)
            
            if time.time() - start_time > time_limit - 5:
                return False # Timeout
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Finished building a row
                if not is_row_valid(current_row):
                    return False
                
                # Check lexicographic order
                if row_idx > 0 and current_row <= prev_row:
                    return False
                
                # Add to current_rows and used_diffs
                current_rows.append(current_row)
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        row_diffs.add(current_row[j] - current_row[i])
                used_diffs.update(row_diffs)
                
                # Recurse to next row
                if dfs(row_idx + 1, 0, [], current_row):
                    return True
                
                # Backtrack
                used_diffs -= row_diffs
                current_rows.pop()
                return False
            
            # Determine range for current_row[pos]
            lower = 0
            upper = M
            
            if pos > 0:
                lower = current_row[-1] + 1
            
            # Lexicographic constraint with prev_row
            if row_idx > 0:
                if current_row == prev_row[:pos]:
                    if pos < len(prev_row):
                        lower = max(lower, prev_row[pos] + 1)
            
            # Iterate x from lower to upper
            for x in range(lower, upper + 1):
                new_row = current_row + [x]
                
                # Check for internal duplicates and conflict with used_diffs
                row_diffs = set()
                valid = True
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if valid and not (row_diffs & used_diffs):
                    if dfs(row_idx, pos + 1, new_row, prev_row):
                        return True
            
            return False
        
        # Start DFS
        if dfs(0, 0, [], []):
            return current_rows[:]
        
        return None

    # Iterate M
    best_rows = None
    for M in range(lower_bound, lower_bound + 30):
        if time.time() - start_time > time_limit - 2:
            break
            
        rows = solve_for_M(M)
        if rows is not None:
            best_rows = rows
            break
            
    if best_rows is None:
        # Fallback to a greedy solution
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows

        if len(current_row) == k + 1:
            if row_idx > 0 and current_row[-1] < rows[row_idx - 1][-1]:
                return False
            
            rows.append(current_row[:])
            
            # Calculate diffs for this row to remove them on backtrack
            row_diffs = set()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    row_diffs.add(current_row[j] - current_row[i])
            
            if backtrack(row_idx + 1, []):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.difference_update(row_diffs)
            return False

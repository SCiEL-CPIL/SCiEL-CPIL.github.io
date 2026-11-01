    def try_move(row_idx, point_idx, new_val):
        """
        Checks if moving current_rows[row_idx][point_idx] to new_val is valid.
        Returns (is_valid, old_diffs, new_diffs)
        """
        row = current_rows[row_idx]
        old_val = row[point_idx]
        
        # Order check
        if point_idx > 0 and new_val <= row[point_idx - 1]:
            return False, None, None
        if point_idx < len(row) - 1 and new_val >= row[point_idx + 1]:
            return False, None, None
            
        # Calculate old diffs involving this point
        old_diffs = set()
        for j, v in enumerate(row):
            if j != point_idx:
                old_diffs.add(abs(old_val - v))
                
        # Calculate new diffs involving this point
        new_diffs = set()
        for j, v in enumerate(row):
            if j != point_idx:
                new_diffs.add(abs(new_val - v))
                
        # Check for internal duplicates in new_diffs
        if len(new_diffs) != len(row) - 1:
            return False, None, None
            
        # Check against global used set
        for d in new_diffs:
            if d in current_used:
                if d not in old_diffs:
                    return False, None, None
                    
        return True, old_diffs, new_diffs

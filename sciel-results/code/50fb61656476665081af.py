        # Shrink window from the left until it is valid
        while conflict_count > 0 and L <= R:
            # Remove L from the window
            if L in self_inv:
                conflict_count -= 1
            else:
                inv_L = inv[L]
                # If the inverse of L is still in the window (i.e., > L), 
                # then removing L resolves the conflict created by that pair.
                if present[inv_L]:
                    conflict_count -= 1
            
            present[L] = 0
            L += 1

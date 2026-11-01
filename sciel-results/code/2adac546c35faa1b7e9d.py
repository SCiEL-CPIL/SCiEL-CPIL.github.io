        def backtrack(idx, used_bits, selected):
            if len(selected) == n_rows:
                return selected
                
            needed = n_rows - len(selected)
            
            # Count valid candidates from idx onwards
            valid_count = 0
            valid_indices = []
            for i in range(idx, len(candidates)):
                if not (candidates[i][1] & used_bits):
                    valid_count += 1
                    valid_indices.append(i)
                    if valid_count >= needed:
                        break # Early stop counting if we have enough? 
                        # No, we need to try them. But if total < needed, fail.
                        
            # If we didn't break, we counted all.
            # Actually, just count all.
            # Re-evaluating: 
            # If valid_count < needed: return None
            
            # Let's just iterate and check.
            for i in range(idx, len(candidates)):
                marks, bits = candidates[i]
                if bits & used_bits:
                    continue
                res = backtrack(i + 1, used_bits | bits, selected + [marks])
                if res:
                    return res
            return None

<<<<<<< SEARCH
        if len(current_code) < best_code_size:
            best_code_size = len(current_code)
            best_code_set = set(current_code)
            
            # Optional: try to improve further with different heuristics?
            # For now, just keep the best found.
=======
        if len(current_code) <= best_code_size:
            best_code_size = len(current_code)
            best_code_set = set(current_code)
>>>>>>> REPLACE

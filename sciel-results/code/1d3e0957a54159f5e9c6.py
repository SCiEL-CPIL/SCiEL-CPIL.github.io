<<<<<<< SEARCH
        if improved:
            if current_size < best_size:
                best_size = current_size
                best_codewords = [c.tolist() for c in codewords]
        
        # Cool down temperature
=======
        if improved:
            if current_size < best_size:
                best_size = current_size
                if isinstance(codewords[0], np.ndarray):
                    best_codewords = [c.tolist() for c in codewords]
                else:
                    best_codewords = [list(c) for c in codewords]
        
        # Cool down temperature
>>>>>>> REPLACE

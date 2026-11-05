        # Try kicks
        # Do many small kicks to explore the neighborhood thoroughly
        for _ in range(100): # Increased from 5 distinct kick sizes to 100 single kicks
            if time.time() > deadline - 2:
                break
            code = kick_and_repair(code, 1, deadline - 2)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                if best_size <= 156: # Theoretical lower bound, stop if reached
                    break

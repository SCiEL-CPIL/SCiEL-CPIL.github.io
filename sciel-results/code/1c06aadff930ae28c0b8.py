    def search(target_k, start_idx=0, current_set=None, current_sums=None):
        if current_set is None:
            current_set = []
        if current_sums is None:
            current_sums = {Fraction(0)}
            
        if len(current_set) == target_k:
            return current_set
        
        if time.time() > deadline:
            return None
            
        if N - start_idx + 1 < target_k - len(current_set):
            return None
            
        # Try larger indices first (smaller reciprocals)
        for i in range(N, start_idx - 1, -1):
            rx = reciprocals[i - 1]
            new_sums = {s + rx for s in current_sums}
            
            if current_sums.isdisjoint(new_sums):
                # Next we can pick any j < i
                result = search(
                    target_k, 
                    i - 1, 
                    current_set + [i], 
                    current_sums | new_sums
                )
                if result is not None:
                    return result
                    
            if time.time() > deadline:
                return None
                
        return None

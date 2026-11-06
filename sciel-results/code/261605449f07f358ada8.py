import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s buffer
    
    selected = []
    existing_sums = set()
    
    # Helper to check if adding x is valid
    def can_add(x):
        # New sums involving x:
        # 1. x+x+x
        # 2. x+x+y for y in selected
        # 3. x+y+z for y <= z in selected
        
        new_sums = set()
        
        # 1. 3x
        s = 3 * x
        if s in existing_sums:
            return False
        new_sums.add(s)
        
        # 2. 2x + y
        for y in selected:
            s = 2 * x + y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
            
        # 3. x + y + z
        # y <= z from selected
        # We can iterate with combinations_with_replacement
        for y, z in itertools.combinations_with_replacement(selected, 2):
            s = x + y + z
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
            
        return True

    for x in range(1, 4097):
        if time.time() - start_time > time_limit:
            break
        
        if can_add(x):
            # Commit x
            selected.append(x)
            
            # Update existing_sums
            existing_sums.add(3 * x)
            for y in selected[:-1]: # y is from previous selected
                existing_sums.add(2 * x + y)
            for y, z in itertools.combinations_with_replacement(selected[:-1], 2):
                existing_sums.add(x + y + z)
                
            # Optimization: If we have enough elements, we might stop early if we reach a known high bound?
            # No, just let it run.
            
    return selected

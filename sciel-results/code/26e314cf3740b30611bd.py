def is_valid_addition(current_sums, x):
    new_sums = set(current_sums)
    if x in new_sums:
        return False, set()
    new_sums.add(x)
    
    # Generate new sums s+x
    # We need to check divisibility between:
    # 1. x and current_sums
    # 2. x and (s+x) -> implies x|s+x => x|s. So if x|s for any s in current_sums, invalid.
    # 3. (s+x) and current_sums
    # 4. (s+x) and (t+x)
    
    # Optimization:
    # Check x vs current_sums
    for s in current_sums:
        if s % x == 0 or x % s == 0:
            return False, set()
            
    # Now we know x is compatible with current_sums.
    # New sums are X = {x} U {s+x for s in current_sums}
    # We need to check compatibility of {s+x} with current_sums and with each other.
    
    shifted = [s + x for s in current_sums]
    
    # Check shifted vs current_sums
    for ns in shifted:
        for s in current_sums:
            if ns % s == 0 or s % ns == 0:
                return False, set()
                
    # Check shifted vs shifted
    # This is O(N^2)
    for i in range(len(shifted)):
        for j in range(i+1, len(shifted)):
            a = shifted[i]
            b = shifted[j]
            if a % b == 0 or b % a == 0:
                return False, set()
                
    # Also check x vs shifted?
    # x vs (s+x): x | (s+x) <=> x | s. Already checked.
    # (s+x) | x? Impossible since s+x > x.
    
    new_sums.update(shifted)
    return True, new_sums

import time

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def construct():
    deadline = time.time() + 75
    # Start with a valid small set, e.g., [1]
    # Or start with the best AP found?
    # Greedy from scratch is risky if early choices block later ones.
    # However, for this problem, smaller elements generally allow more elements to fit in the range.
    # But "validity" requires separation.
    
    # Let's try to find the largest k by greedy expansion.
    # We will try to add elements one by one.
    # To improve chances, we can try multiple strategies or backtracking?
    # Backtracking is too slow.
    
    # Let's stick to the AP logic but try to extend it?
    # Or just run greedy smallest-valid.
    
    current_set = []
    # Try to add elements
    for _ in range(60):
        if time.time() > deadline:
            break
        found = False
        # Start candidate from max(current_set) + 1 if current_set else 1
        start_val = max(current_set) + 1 if current_set else 1
        for x in range(start_val, 257):
            new_set = current_set + [x]
            if _is_valid_bitset(new_set):
                current_set = new_set
                found = True
                break
        if not found:
            break
            
    if not current_set:
        return [1]
        
    return current_set

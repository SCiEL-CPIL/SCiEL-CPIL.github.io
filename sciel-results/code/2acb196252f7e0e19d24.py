import time

def construct():
    MOD = 1009
    N = 1008
    t0 = time.time()
    deadline = t0 + 5.0  # Very fast, no need for long deadline

    # 1. Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # 2. Build list of inverse pairs (low, high)
    # Exclude self-inverses (1 and 1008) from pairs list as they are handled by boundaries
    pairs = []
    for x in range(1, MOD):
        y = inv[x]
        if x < y:
            # x and y form a pair
            # Check if x or y is a self-inverse (should not be if x != y)
            pairs.append((x, y))
    
    # Sort pairs by 'low' in descending order to facilitate the sweep
    pairs.sort(key=lambda p: p[0], reverse=True)

    best_start = 2
    best_len = 0
    min_high = float('inf')
    
    # Pointer to the next pair to consider
    # We iterate L from 1007 down to 2
    # We need to add pairs where low >= L
    # Since pairs are sorted by low descending, we can just keep adding
    # as L decreases, more pairs satisfy low >= L
    
    pair_idx = 0
    num_pairs = len(pairs)
    
    # We only care about L in [2, 1007]
    # R is capped at 1007
    
    for L in range(1007, 1, -1):
        # Add all pairs with low >= L
        # Since we are iterating L downwards, and pairs are sorted by low descending,
        # pairs with low == L, L+1, ... have been considered or are about to be.
        # Wait, if pairs are sorted descending:
        # Pairs with high 'low' come first.
        # When L is large (e.g., 1007), we want pairs with low >= 1007.
        # These are at the beginning of the sorted list.
        # As L decreases, we want to include pairs with lower 'low'.
        # So we just advance pair_idx to include more pairs.
        
        while pair_idx < num_pairs and pairs[pair_idx][0] >= L:
            high_val = pairs[pair_idx][1]
            if high_val < min_high:
                min_high = high_val
            pair_idx += 1
            
        # Now min_high is the min high among all pairs with low >= L
        # The first invalid R is min_high
        # Max valid R is min_high - 1
        # But R cannot exceed 1007
        max_valid_R = min_high - 1
        if max_valid_R > 1007:
            max_valid_R = 1007
            
        if max_valid_R >= L:
            current_len = max_valid_R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L

    # Fallback if no valid interval found (should not happen)
    if best_len <= 0:
        best_start = 2
        best_len = 1

    # Final validation check (optional but safe)
    def is_valid(s, ln):
        if s < 1 or s + ln - 1 > N:
            return False
        end = s + ln - 1
        for x in range(s, end + 1):
            ix = inv[x]
            if s <= ix <= end and ix != x:
                return False
            if ix == x: # self inverse
                return False
        return True

    if not is_valid(best_start, best_len):
        # This should not happen with the logic above, but just in case
        # Try a simple linear search for a valid single element
        for x in range(2, 1008):
            if is_valid(x, 1):
                best_start = x
                best_len = 1
                break

    if time.time() > deadline:
        # Return best found so far
        pass

    return [best_start, best_len]

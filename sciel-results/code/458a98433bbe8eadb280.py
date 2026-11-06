import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses via Fermat's little theorem
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Find self-inverse elements
    self_inv = set()
    for i in range(1, N + 1):
        if inv[i] == i:
            self_inv.add(i)

    # For each position, compute the nearest "obstacle" to the left
    # An obstacle at position p means that if p is in the interval, 
    # its inverse inv[p] cannot also be in the interval.
    # We need: for all x in [start, end], inv[x] not in [start, end]
    # This means: for all x in [start, end], either inv[x] < start or inv[x] > end
    
    # Alternative approach: for each right endpoint, find the maximum valid left
    # The constraint is that no pair (x, inv[x]) is fully contained in the interval.
    # For a fixed right, the leftmost valid start is max over all x in [start, right] 
    # of (if inv[x] >= start then inv[x]+1 else start)
    # This is circular, so use two-pointer.
    
    best_start = 1
    best_len = 0

    left = 1

    for right in range(1, N + 1):
        if time.time() > deadline:
            break

        x = right
        y = inv[x]

        if x == y:
            # Self-inverse residues cannot appear in any valid interval
            left = right + 1
            # After skipping, we need to ensure window is valid
            # But since x=y is excluded, and we moved left past it,
            # the window [left, right] doesn't contain x, so it's fine
            # Actually we should continue without updating best since right is excluded
            # But the window [left, right] with left=right+1 is empty
            continue

        # If y lies in the current window [left, right-1], advance left past y
        if left <= y < right:
            left = y + 1

        # Now verify: is [left, right] actually valid?
        # Check that no element in [left, right] has its inverse in [left, right]
        # Due to the two-pointer invariant, this should hold, but let's verify for safety
        # Only do full check if we have spare time (expensive O(n) check)
        
        current_len = right - left + 1
        if current_len > 0 and current_len > best_len:
            # Quick verification: check that inv[right] is not in [left, right-1]
            # (other elements were already verified when they were added)
            if not (left <= inv[right] <= right):
                best_len = current_len
                best_start = left
            else:
                # This shouldn't happen with correct two-pointer, but handle it
                # Find the actual valid left by scanning
                valid_left = left
                for x in range(left, right + 1):
                    iy = inv[x]
                    if valid_left <= iy <= right:
                        valid_left = iy + 1
                current_len = right - valid_left + 1
                if current_len > best_len:
                    best_len = current_len
                    best_start = valid_left
                left = valid_left

    # Ensure we return a valid non-empty interval
    if best_len <= 0:
        # Find any single element that is not self-inverse
        for i in range(1, N + 1):
            if i not in self_inv:
                best_start = i
                best_len = 1
                break

    return [best_start, best_len]

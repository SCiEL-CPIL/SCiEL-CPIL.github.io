import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin

    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility: divs[i] = set of j where smooth[j] divides smooth[i]
    divs = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth_numbers[i] % smooth_numbers[j] == 0:
                    divs[i].add(j)
    
    # compat[i][j] = True if i and j can coexist in an antichain
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in divs[i]:
            compat[i][j] = False
            compat[j][i] = False

    # Precompute all valid antichains of size 1-5 and map sum -> list of terms
    antichain_sums = {k: {} for k in range(1, 6)}
    
    # Size 1
    for i in range(n_smooth):
        s = smooth_numbers[i]
        if s <= 128:
            antichain_sums[1][s] = [smooth_numbers[i]]
    
    # Size 2
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if compat[i][j]:
                s = smooth_numbers[i] + smooth_numbers[j]
                if s <= 128:
                    if s not in antichain_sums[2]:
                        antichain_sums[2][s] = [smooth_numbers[i], smooth_numbers[j]]
    
    # Size 3
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if not compat[i][j]:
                continue
            for k in range(j+1, n_smooth):
                if compat[i][k] and compat[j][k]:
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                    if s <= 128:
                        if s not in antichain_sums[3]:
                            antichain_sums[3][s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]]
    
    # Identify targets not covered by sizes 1-3
    targets_needing_4 = set()
    for target in range(1, 129):
        if target not in antichain_sums[1] and target not in antichain_sums[2] and target not in antichain_sums[3]:
            targets_needing_4.add(target)
    
    # Size 4
    if targets_needing_4:
        for i in range(n_smooth):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, n_smooth):
                if not compat[i][j]:
                    continue
                for k in range(j+1, n_smooth):
                    if not (compat[i][k] and compat[j][k]):
                        continue
                    for m in range(k+1, n_smooth):
                        if compat[i][m] and compat[j][m] and compat[k][m]:
                            s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m]
                            if s <= 128 and s in targets_needing_4:
                                if s not in antichain_sums[4]:
                                    antichain_sums[4][s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[m]]
    
    # Size 5 for any remaining targets
    targets_needing_5 = set()
    for target in range(1, 129):
        if target not in antichain_sums[1] and target not in antichain_sums[2] and target not in antichain_sums[3] and target not in antichain_sums[4]:
            targets_needing_5.add(target)
    
    if targets_needing_5 and time.time() < deadline - 3.0:
        for i in range(n_smooth):
            if time.time() > deadline - 3.0:
                break
            for j in range(i+1, n_smooth):
                if not compat[i][j]:
                    continue
                for k in range(j+1, n_smooth):
                    if not (compat[i][k] and compat[j][k]):
                        continue
                    for m in range(k+1, n_smooth):
                        if not (compat[i][m] and compat[j][m] and compat[k][m]):
                            continue
                        for p in range(m+1, n_smooth):
                            if compat[i][p] and compat[j][p] and compat[k][p] and compat[m][p]:
                                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m] + smooth_numbers[p]
                                if s <= 128 and s in targets_needing_5:
                                    if s not in antichain_sums[5]:
                                        antichain_sums[5][s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[m], smooth_numbers[p]]
    
    # Build answers
    answers = []
    for target in range(1, 129):
        found = None
        for k in range(1, 6):
            if target in antichain_sums[k]:
                found = antichain_sums[k][target]
                break
        
        if found is None:
            # Fallback: proper backtracking
            found = _backtrack_search(target, smooth_numbers, compat, n_smooth)
        
        if found is None:
            # Last resort: allow up to 30 terms via greedy
            found = _greedy_antichain(target, smooth_numbers, compat, n_smooth)
        
        answers.append(found)
    
    return answers

def _backtrack_search(target, smooth_numbers, compat, n):
    """Proper backtracking to find an antichain summing to target."""
    # Try sizes 1 through 6
    for size in range(1, 7):
        result = _bt(target, size, 0, [], smooth_numbers, compat, n)
        if result is not None:
            return result
    return None

def _bt(remaining, size, start, current_idx, smooth_numbers, compat, n):
    if size == 0:
        return list(current_idx) if remaining == 0 else None
    if remaining <= 0:
        return None
    
    # Pruning: minimum possible sum with 'size' more elements starting from 'start'
    # Each element >= smooth_numbers[start] (since sorted)
    if size > 0 and start + size > n:
        return None
    min_possible = sum(smooth_numbers[start:start+size]) if start + size <= n else float('inf')
    if min_possible > remaining:
        return None
    
    for i in range(start, n):
        val = smooth_numbers[i]
        if val > remaining:
            break
        # Check compatibility with all current elements
        compatible = True
        for j in current_idx:
            if not compat[i][j]:
                compatible = False
                break
        if not compatible:
            continue
        
        new_current = current_idx + [i]
        result = _bt(remaining - val, size - 1, i + 1, new_current, smooth_numbers, compat, n)
        if result is not None:
            return [smooth_numbers[x] for x in result]
    return None

def _greedy_antichain(target, smooth_numbers, compat, n):
    """Greedy fallback: pick largest compatible smooth numbers that sum to target."""
    # This is a last resort; try to build an antichain greedily
    used = set()
    remaining = target
    result = []
    
    # Try all smooth numbers in descending order
    for i in range(n-1, -1, -1):
        val = smooth_numbers[i]
        if val > remaining:
            continue
        # Check if adding this maintains antichain
        compatible = True
        for j in used:
            if not compat[i][j]:
                compatible = False
                break
        if compatible:
            result.append(val)
            used.add(i)
            remaining -= val
            if remaining == 0:
                return result
    
    return None

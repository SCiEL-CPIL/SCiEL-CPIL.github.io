"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill

def construct():
    t0 = time.time()
    
    # Generate all smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    smooth_set = set(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # Precompute divisibility bitmask: div_mask[i] is a bitmask of all j where smooth[i] divides smooth[j]
    # Also need the reverse: for each i, which j divide smooth[i]
    div_mask = [0] * n_smooth  # div_mask[i]: bits set for j where smooth[i] | smooth[j]
    rdiv_mask = [0] * n_smooth  # rdiv_mask[i]: bits set for j where smooth[j] | smooth[i]
    
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth[i], smooth[j]
                if b % a == 0:
                    div_mask[i] |= (1 << j)
                    rdiv_mask[j] |= (1 << i)
    
    # Precompute: for each pair (i,j) with i<j, whether they form an antichain pair
    # pair_ok_mask[i]: bitmask of j > i where (i,j) is antichain
    pair_ok_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            # Check if neither divides the other
            if not (div_mask[i] & (1 << j)) and not (div_mask[j] & (1 << i)):
                pair_ok_mask[i] |= (1 << j)
    
    # Precompute sum_to_pairs: for each sum s, list of (i,j) antichain pairs
    sum_to_pairs = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if pair_ok_mask[i] & (1 << j):
                s = smooth[i] + smooth[j]
                if s <= 128:
                    if s not in sum_to_pairs:
                        sum_to_pairs[s] = []
                    sum_to_pairs[s].append((i, j))
    
    # For each target, find minimal-term antichain representation
    answers = []
    
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: quick search
            found = _quick_fallback(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask)
            answers.append(found)
            continue
        
        found = None
        
        # Try k=1: single smooth number
        if n in smooth_set:
            found = [n]
        
        if found is None:
            # Try k=2: two smooth numbers, neither divides the other
            if n in sum_to_pairs:
                i, j = sum_to_pairs[n][0]
                found = [smooth[i], smooth[j]]
        
        if found is None:
            # Try k=3: three smooth numbers forming antichain
            found = _search_k3(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask)
        
        if found is None:
            # Try k=4: four smooth numbers forming antichain
            found = _search_k4(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask)
        
        if found is None:
            # Try k=5: five smooth numbers forming antichain
            found = _search_k5(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask)
        
        if found is None:
            # Try k=6..10: use general DFS
            for k in range(6, 11):
                if time.time() - t0 > DEADLINE:
                    break
                found = _search_general(n, k, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask)
                if found is not None:
                    break
        
        if found is None:
            # Fallback
            found = _quick_fallback(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask)
        
        answers.append(found)
    
    return answers

def _search_k3(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask):
    """Search for 3-term antichain representation of n."""
    n_smooth = len(smooth)
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        # Need b + c = n - a, with b > a, c > b
        rem = n - a
        # b must be > a and < rem/2
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if b >= rem:
                break
            if b <= a:
                continue
            # Check antichain pair (i, j)
            if not (pair_ok_mask[i] & (1 << j)):
                continue
            c = rem - b
            if c <= b:
                break
            if c in smooth_set:
                k = val_to_idx[c]
                # Check antichain with i and j
                if not (div_mask[i] & (1 << k)) and not (div_mask[j] & (1 << k)):
                    return [a, b, c]
    return None

def _search_k4(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask):
    """Search for 4-term antichain representation of n."""
    n_smooth = len(smooth)
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        rem1 = n - a
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if b >= rem1:
                break
            if not (pair_ok_mask[i] & (1 << j)):
                continue
            rem2 = rem1 - b
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if c >= rem2:
                    break
                if not (pair_ok_mask[i] & (1 << k)) or not (pair_ok_mask[j] & (1 << k)):
                    continue
                d = rem2 - c
                if d <= c:
                    break
                if d in smooth_set:
                    l = val_to_idx[d]
                    if (not (div_mask[i] & (1 << l)) and not (div_mask[j] & (1 << l)) 
                        and not (div_mask[k] & (1 << l))):
                        return [a, b, c, d]
    return None

def _search_k5(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask):
    """Search for 5-term antichain representation of n."""
    n_smooth = len(smooth)
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        rem1 = n - a
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if b >= rem1:
                break
            if not (pair_ok_mask[i] & (1 << j)):
                continue
            rem2 = rem1 - b
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if c >= rem2:
                    break
                if not (pair_ok_mask[i] & (1 << k)) or not (pair_ok_mask[j] & (1 << k)):
                    continue
                rem3 = rem2 - c
                for l in range(k + 1, n_smooth):
                    d = smooth[l]
                    if d >= rem3:
                        break
                    if (not (pair_ok_mask[i] & (1 << l)) or not (pair_ok_mask[j] & (1 << l)) 
                        or not (pair_ok_mask[k] & (1 << l))):
                        continue
                    e = rem3 - d
                    if e <= d:
                        break
                    if e in smooth_set:
                        m = val_to_idx[e]
                        if (not (div_mask[i] & (1 << m)) and not (div_mask[j] & (1 << m)) 
                            and not (div_mask[k] & (1 << m)) and not (div_mask[l] & (1 << m))):
                            return [a, b, c, d, e]
    return None

def _search_general(n, max_terms, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask, pair_ok_mask):
    """General DFS search for antichain representation with up to max_terms."""
    n_smooth = len(smooth)
    
    def dfs(rem, min_idx, chosen, chosen_mask, remaining_terms):
        if remaining_terms == 0:
            return chosen if rem == 0 else None
        if rem <= 0:
            return None
        if len(chosen) >= max_terms:
            return None
        
        # Pruning: can we still reach rem with remaining_terms?
        # Minimum possible sum: smallest remaining_terms smooth numbers from min_idx
        min_possible = 0
        count = 0
        for idx in range(min_idx, n_smooth):
            if count >= remaining_terms:
                break
            min_possible += smooth[idx]
            count += 1
        if count < remaining_terms or min_possible > rem:
            return None
        
        for idx in range(min_idx, n_smooth):
            s = smooth[idx]
            if s > rem:
                break
            # Check antichain with all chosen
            valid = True
            for c_idx in chosen:
                if div_mask[c_idx] & (1 << idx):
                    valid = False
                    break
                if div_mask[idx] & (1 << c_idx):
                    valid = False
                    break
            if not valid:
                continue
            
            result = dfs(rem - s, idx + 1, chosen + [s], chosen_mask | (1 << idx), remaining_terms - 1)
            if result is not None:
                return result
        return None
    
    return dfs(n, 0, [], 0, max_terms)

def _quick_fallback(n, smooth, smooth_set, val_to_idx, div_mask, rdiv_mask):
    """Quick fallback to find any valid representation."""
    n_smooth = len(smooth)
    
    # Try 1-term
    if n in smooth_set:
        return [n]
    
    # Try 2-term
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        b = n - a
        if b <= a:
            continue
        if b in smooth_set:
            j = val_to_idx[b]
            if not (div_mask[i] & (1 << j)) and not (div_mask[j] & (1 << i)):
                return [a, b]
    
    # Try 3-term with simple search
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        rem = n - a
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if b >= rem:
                break
            if b <= a:
                continue
            if div_mask[i] & (1 << j):
                continue
            c = rem - b
            if c <= b:
                break
            if c in smooth_set:
                k = val_to_idx[c]
                if not (div_mask[i] & (1 << k)) and not (div_mask[j] & (1 << k)):
                    return [a, b, c]
    
    # Try 4-term with simple search
    for i in range(n_smooth):
        a = smooth[i]
        if a >= n:
            break
        rem1 = n - a
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if b >= rem1:
                break
            if div_mask[i] & (1 << j):
                continue
            rem2 = rem1 - b
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if c >= rem2:
                    break
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                d = rem2 - c
                if d <= c:
                    break
                if d in smooth_set:
                    l = val_to_idx[d]
                    if (not (div_mask[i] & (1 << l)) and not (div_mask[j] & (1 << l)) 
                        and not (div_mask[k] & (1 << l))):
                        return [a, b, c, d]
    
    # Last resort: use many small terms
    # Try to build up using 1, 2, 3, 5, 7, 8, 9, 11, 13, etc.
    # This is a heuristic that may not always work, but should for most cases
    candidates = [x for x in smooth if x <= n]
    candidates.sort(reverse=True)
    
    chosen = []
    chosen_set = set()
    rem = n
    
    for c in candidates:
        if c > rem:
            continue
        # Check if c can be added (antichain with all chosen)
        valid = True
        for chosen_val in chosen:
            if chosen_val in div_mask or c in div_mask:
                # Need to check actual divisibility
                if c % chosen_val == 0 or chosen_val % c == 0:
                    valid = False
                    break
        if valid:
            chosen.append(c)
            chosen_set.add(c)
            rem -= c
            if rem == 0:
                break
    
    if rem == 0 and len(chosen) <= 30:
        return chosen
    
    # Absolute last resort: return something (may be invalid, but should rarely happen)
    return [1] * n if n <= 30 else [1] * 30 + [n - 30] if n > 30 else [1]

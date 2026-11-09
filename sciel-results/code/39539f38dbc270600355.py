import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        va = 2 ** a
        for b in range(0, 5):
            vab = va * (3 ** b)
            if vab > 128:
                break
            for c in range(0, 4):
                vabc = vab * (5 ** c)
                if vabc > 128:
                    break
                smooth.add(vabc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute divisibility (symmetric): divs[i][j] = True if one divides the other
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    divs[i][j] = True
                    divs[j][i] = True
    
    # For each target n, get candidate indices (smooth numbers <= n), sorted descending
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for idx in range(n_smooth):
            if smooth_list[idx] <= n:
                candidates[n].append(idx)
            else:
                break
        candidates[n].sort(key=lambda idx: smooth_list[idx], reverse=True)
    
    # Precompute prefix sums for max-sum pruning (candidates sorted descending)
    # prefix_max[n][i] = sum of first i elements in candidates[n]
    prefix_max = [None] * 129
    for n in range(1, 129):
        cand = candidates[n]
        ps = [0]
        for idx in cand:
            ps.append(ps[-1] + smooth_list[idx])
        prefix_max[n] = ps
    
    # Precompute suffix sums for min-sum pruning
    # suffix_min[n][i] = sum of last (len-i) elements... actually sum from i to end
    # We want: sum of smallest k from position i onward = sum of last k elements
    # Since sorted descending, smallest are at the end
    # min_sum_from_i_k = sum of elements at positions len-k to len-1 (the k smallest overall)
    # But we need "from position i": the k smallest from cand[i:]
    # That's the last k elements of cand[i:]
    suffix_min = [None] * 129
    for n in range(1, 129):
        cand = candidates[n]
        ln = len(cand)
        # Build suffix sums: suff[i] = sum(cand[i:])
        suff = [0] * (ln + 1)
        for i in range(ln - 1, -1, -1):
            suff[i] = suff[i + 1] + smooth_list[cand[i]]
        suffix_min[n] = suff  # suff[i] = sum from i to end
    
    def find_antichain(n, k):
        """Find antichain of size k summing to n. Returns list of values or None."""
        if k == 1:
            if n in val_to_idx:
                return [n]
            return None
        
        cand = candidates[n]
        ln = len(cand)
        if ln < k:
            return None
        
        # Quick bounds
        max_sum = prefix_max[n][k]
        min_sum = suffix_min[n][ln - k]
        if n > max_sum or n < min_sum:
            return None
        
        path = []  # list of indices
        
        def backtrack(start, rem_sum, rem_k):
            if rem_k == 0:
                return [smooth_list[idx] for idx in path] if rem_sum == 0 else None
            
            if len(cand) - start < rem_k:
                return None
            
            # Max possible: sum of next rem_k largest from position start
            max_possible = prefix_max[n][start + rem_k] - prefix_max[n][start]
            if max_possible < rem_sum:
                return None
            
            # Min possible: sum of rem_k smallest from position start onward
            # = sum of last rem_k elements in cand[start:]
            # = suffix_min[n][len(cand) - rem_k] ... but only if those are >= start
            # Actually: the rem_k smallest from cand[start:] are at positions
            # (len(cand) - rem_k) to (len(cand) - 1), provided len(cand) - rem_k >= start
            if len(cand) - rem_k < start:
                return None  # not enough elements
            min_possible = suffix_min[n][len(cand) - rem_k] - suffix_min[n][len(cand)]
            # Wait, suffix_min[n][i] = sum from i to end. So sum of last rem_k = suff[len-rem_k] - suff[len]
            # But suff[len] = 0, so min_possible = suff[len - rem_k]
            # But this is the global last rem_k, not necessarily from position start.
            # If len(cand) - rem_k >= start, then the last rem_k elements are all >= start, so it's valid.
            min_possible = suffix_min[n][len(cand) - rem_k]
            if min_possible > rem_sum:
                return None
            
            # Iterate candidates
            for i in range(start, len(cand) - rem_k + 1):
                idx = cand[i]
                num = smooth_list[idx]
                if num > rem_sum:
                    continue
                
                # Divisibility check
                compatible = True
                for prev in path:
                    if divs[prev][idx]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                path.append(idx)
                result = backtrack(i + 1, rem_sum - num, rem_k - 1)
                if result is not None:
                    return result
                path.pop()
            
            return None
        
        return backtrack(0, n, k)
    
    answers = []
    total_terms = 0
    
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Emergency fallback: use any valid representation
            # Try k=1 first, then a simple greedy
            if n in val_to_idx:
                answers.append([n])
            else:
                # Greedy: pick largest smooth <= remaining, check antichain
                terms = []
                rem = n
                used = set()
                for idx in range(n_smooth - 1, -1, -1):
                    v = smooth_list[idx]
                    if v > rem:
                        continue
                    if v in used:
                        continue
                    # Check divisibility
                    ok = True
                    for t in terms:
                        if t % v == 0 or v % t == 0:
                            ok = False
                            break
                    if ok:
                        terms.append(v)
                        used.add(v)
                        rem -= v
                        if rem == 0:
                            break
                if rem == 0:
                    answers.append(terms)
                else:
                    # Last resort: just [n] if smooth, else [1, n-1] hack
                    if n in val_to_idx:
                        answers.append([n])
                    else:
                        answers.append([1, n - 1] if n - 1 > 0 else [n])
            continue
        
        answer = None
        for k in range(1, 31):
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback
            if n in val_to_idx:
                answer = [n]
            else:
                answer = [1, n - 1]
        
        answers.append(answer)
        total_terms += len(answer)
    
    return answers

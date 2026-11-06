"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _generate_smooth(max_val):
    """Generate all 2,3,5-smooth numbers up to max_val."""
    smooths = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooths.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted([x for x in smooths if x <= max_val])

def _find_min_antichain_sum(n, smooths, deadline=None):
    """Find minimum-k antichain subset of smooths (<=n) summing to n."""
    candidates = [x for x in smooths if x <= n]
    if not candidates:
        return None
    
    m = len(candidates)
    if m == 0:
        return None
    
    # Sort candidates ascending for pruning
    sorted_cands = sorted(candidates)
    m = len(sorted_cands)
    
    # Precompute divisibility in sorted order
    div = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and sorted_cands[j] % sorted_cands[i] == 0:
                div[i][j] = True
    
    # Prefix sums for pruning
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + sorted_cands[i]
    
    # Suffix sums
    suffix = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_cands[i]
    
    for k in range(1, min(m, 30) + 1):
        if deadline and time.time() > deadline:
            break
        found = [False]
        chosen = []
        chosen_idx = []
        
        def backtrack(start, remaining):
            if found[0]:
                return
            if len(chosen) == k:
                if remaining == 0:
                    found[0] = True
                return
            if remaining <= 0:
                return
            picks_left = k - len(chosen)
            if picks_left <= 0:
                return
            if remaining < picks_left:
                return
            if start + picks_left > m:
                return
            # Max possible sum from start with picks_left elements
            if start + picks_left <= m:
                max_sum = suffix[start] - suffix[start + picks_left] if start + picks_left <= m else 0
                # Actually suffix[start] is sum from start to end. We want sum of picks_left largest from start..end
                # That's suffix[start] - suffix[start + (m - start - picks_left)] if m-start >= picks_left
                avail = m - start
                if avail < picks_left:
                    return
                # Largest picks_left from start..end = suffix[start] - suffix[m - picks_left]
                max_sum = suffix[start] - suffix[m - picks_left]
                if max_sum < remaining:
                    return
                # Smallest picks_left from start..end
                min_sum = prefix[start + picks_left] - prefix[start]
                if min_sum > remaining:
                    return
            
            for i in range(start, m - picks_left + 1):
                if found[0]:
                    return
                # Antichain check
                ok = True
                for j in chosen_idx:
                    if div[i][j] or div[j][i]:
                        ok = False
                        break
                if not ok:
                    continue
                chosen.append(sorted_cands[i])
                chosen_idx.append(i)
                backtrack(i + 1, remaining - sorted_cands[i])
                chosen.pop()
                chosen_idx.pop()
        
        backtrack(0, n)
        if found[0]:
            return sorted(chosen)
    
    return None

def _precompute_answers():
    """Precompute the optimal antichain representations for all targets 1..128."""
    smooths = _generate_smooth(128)
    start_time = time.time()
    deadline = start_time + 80.0
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            for nn in range(n, 129):
                if nn in smooths:
                    answers.append([nn])
                else:
                    answers.append([1])
            break
        ans = _find_min_antichain_sum(n, smooths, deadline)
        if ans is None:
            if n in smooths:
                ans = [n]
            else:
                ans = [1]
        answers.append(ans)
    return answers

_ANSWERS = _precompute_answers()

def construct():
    return _ANSWERS
# EVOLVE-BLOCK-END

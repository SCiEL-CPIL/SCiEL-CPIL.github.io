"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_subset_sum_bitset(values, target, min_k, max_k):
    """
    Check if there exists a subset of 'values' with size in [min_k, max_k]
    that sums to 'target'. Uses bitset DP.
    Returns True if found.
    """
    n = len(values)
    if n < min_k:
        return False
    
    # Compute bounds for the DP array
    pos_sum = sum(v for v in values if v > 0)
    neg_sum = sum(-v for v in values if v < 0)
    offset = neg_sum
    width = pos_sum + neg_sum + 1
    if width <= 0:
        # All values are 0
        return (target == 0) and (min_k <= n)
    
    mask = (1 << width) - 1
    
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in values:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= (shifted & mask)
    
    for k in range(min_k, max_k + 1):
        idx = offset + target
        if 0 <= idx < width:
            if dp[k] & (1 << idx):
                return True
    return False

def _is_sidon(seq):
    """Check if seq is a Sidon sequence (all pairwise sums distinct)."""
    n = len(seq)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = seq[i] + seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _build_sidon_prefix(limit):
    """
    Build a Sidon sequence within [1, limit] greedily.
    Returns the longest prefix that is Sidon.
    """
    seq = []
    used_sums = set()
    for x in range(1, limit + 1):
        # Check if adding x keeps it Sidon
        ok = True
        for y in seq:
            s = x + y
            if s in used_sums:
                ok = False
                break
        if ok:
            # Also check x+x
            s = x + x
            if s in used_sums:
                ok = False
        if ok:
            seq.append(x)
            for y in seq:
                used_sums.add(x + y)
            if len(seq) >= 48:
                break
    return seq

def _check_x_mean_of_subset(selected, x, min_size=2):
    """
    Check if x is the mean of any subset of at least min_size elements from selected.
    Equivalent to: exists S ⊆ selected, |S| >= min_size, sum(S) = |S| * x
    i.e., exists S, sum(y - x for y in S) = 0, |S| >= min_size
    """
    n = len(selected)
    if n < min_size:
        return False
    
    zs = [y - x for y in selected]
    return _check_subset_sum_bitset(zs, 0, min_size, n)

def _check_existing_violation(selected, x, min_size=2):
    """
    For each a in selected, check if a is now the mean of a subset of (selected \ {a}) ∪ {x}
    that includes x and has size >= min_size.
    """
    for a_idx, a in enumerate(selected):
        others = selected[:a_idx] + selected[a_idx+1:]
        m = len(others)
        if m < 1:
            continue
        # Need: exists S ⊆ others, |S| >= 1, sum(S) + x = (|S|+1)*a
        # i.e., sum(S) = (|S|+1)*a - x
        # For each k from 1 to m: target = (k+1)*a - x
        # We need to check if there's a subset of 'others' of size k summing to target.
        # Use bitset DP on 'others'
        pos_sum = sum(v for v in others if v > 0)
        neg_sum = sum(-v for v in others if v < 0)
        offset = neg_sum
        width = pos_sum + neg_sum + 1
        if width <= 0:
            continue
        mask = (1 << width) - 1
        
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        
        for y in others:
            for k in range(m, 0, -1):
                if dp[k-1] == 0:
                    continue
                if y >= 0:
                    shifted = dp[k-1] << y
                else:
                    shifted = dp[k-1] >> (-y)
                dp[k] |= (shifted & mask)
        
        for k in range(1, m + 1):
            if k + 1 < min_size:
                continue
            target = (k + 1) * a - x
            idx = offset + target
            if 0 <= idx < width:
                if dp[k] & (1 << idx):
                    return True
    return False

def _quick_pair_check(selected, x):
    """
    Quick check for 2-element subsets (size-2 means).
    """
    # 1. Is x mean of a pair in selected?
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            if selected[i] + selected[j] == 2 * x:
                return True
    # 2. Is any a in selected mean of pair (x, y)?
    sel_set = set(selected)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
    return False

def _full_check(C):
    """
    Full verification: no element is the mean of any subset of >= 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        n = len(others)
        if n < 2:
            continue
        if _check_x_mean_of_subset(others, a, min_size=2):
            return True
    return False

def _greedy_extend_from_sidon(sidon_seq, candidates, start_time, deadline):
    """
    Start with a Sidon sequence, then greedily add more elements.
    Since the base is Sidon, we only need to check subset sizes >= 3 for new elements,
    but we still need to check that existing elements aren't violated.
    """
    selected = list(sidon_seq)
    sel_set = set(selected)
    
    for x in candidates:
        if time.time() - start_time > deadline:
            break
        if len(selected) >= 48:
            break
        if x in sel_set:
            continue
        
        # Quick pair check (size-2 means)
        if _quick_pair_check(selected, x):
            continue
        
        # Check if x is mean of subset of selected with size >= 3
        # (size-2 already checked by pair check)
        if len(selected) >= 3:
            if _check_x_mean_of_subset(selected, x, min_size=3):
                continue
        
        # Check if any existing element is now violated by adding x
        # For existing a, a could be mean of subset including x of size >= 2
        # Since base is Sidon, size-2 violations involving x are caught by pair check
        # So we only need to check size >= 3 for existing elements
        if _check_existing_violation(selected, x, min_size=3):
            continue
        
        selected.append(x)
        sel_set.add(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = _DEADLINE - 3
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Strategy 1: Build Sidon prefix, then extend
    sidon = _build_sidon_prefix(512)
    if len(sidon) > len(best_set):
        best_set = sidon[:]
    
    # Extend the Sidon sequence with remaining candidates
    remaining = [x for x in candidates if x not in set(sidon)]
    extended = _greedy_extend_from_sidon(sidon, remaining, start_time, deadline)
    if len(extended) > len(best_set):
        best_set = extended[:]
    
    # Strategy 2: Try different Sidon-like constructions
    # Use a modified greedy that prioritizes spread-out numbers
    for trial in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Build a partial Sidon-like set with different biases
        seq = []
        used_sums = set()
        rng = random.Random(trial * 12345 + 67890)
        
        # Start with some spread-out numbers
        spread_starts = [1, 512, 256, 128, 384, 64, 448]
        rng.shuffle(spread_starts)
        
        for x in spread_starts:
            if x > 512 or x < 1:
                continue
            ok = True
            for y in seq:
                if x + y in used_sums:
                    ok = False
                    break
            if ok and (x + x) in used_sums:
                ok = False
            if ok:
                seq.append(x)
                for y in seq:
                    used_sums.add(x + y)
        
        # Now greedily add more
        for x in candidates:
            if time.time() - start_time > deadline:
                break
            if len(seq) >= 48:
                break
            if x in set(seq):
                continue
            ok = True
            for y in seq:
                if x + y in used_sums:
                    ok = False
                    break
            if ok and (x + x) in used_sums:
                ok = False
            if ok:
                seq.append(x)
                for y in seq:
                    used_sums.add(x + y)
        
        if len(seq) > len(best_set):
            best_set = seq[:]
    
    # Strategy 3: Random greedy with pair check + size>=3 check
    rng = random.Random(42)
    for trial in range(200):
        if time.time() - start_time > deadline - 2:
            break
        
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        
        selected = []
        sel_set = set()
        
        for x in rand_order:
            if time.time() - start_time > deadline:
                break
            if len(selected) >= 48:
                break
            if x in sel_set:
                continue
            
            if _quick_pair_check(selected, x):
                continue
            
            if len(selected) >= 3:
                if _check_x_mean_of_subset(selected, x, min_size=3):
                    continue
            
            if _check_existing_violation(selected, x, min_size=3):
                continue
            
            selected.append(x)
            sel_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification
    if best_set:
        if _full_check(best_set):
            # Fallback
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END

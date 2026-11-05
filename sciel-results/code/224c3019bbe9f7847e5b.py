"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import itertools
import sys

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None

def _check_safe(a, others):
    """Return True if no nonempty subset of `others` sums to Fraction(1, a)."""
    target = Fraction(1, a)
    m = len(others)
    if m == 0:
        return True
    
    # Sort others in descending order of value (ascending reciprocal) for better pruning
    # Actually, for subset sum with positive terms, sorting descending by term value helps prune faster
    # Terms are 1/x. Larger x means smaller term. So sort x ascending to get larger terms first?
    # No, sort x descending to get larger terms (smaller x) first.
    o = sorted(others, reverse=True)  # e.g., [24, 23, ..., 1] -> terms [1/24, 1/23, ..., 1]
    # Wait, 1/1 is the largest term. So sorting x descending puts 1/24 first (smallest term).
    # For pruning `acc > target`, it is better to have larger terms first so we exceed target sooner.
    # So we should sort x ascending: [1, 2, ..., 24] -> terms [1, 1/2, ..., 1/24].
    o = sorted(others)  # ascending x -> descending terms
    
    n = len(o)
    found = [False]
    
    # Precompute terms
    terms = [Fraction(1, x) for x in o]
    
    # DFS with memoization on (index, current_sum) is hard because sum is Fraction.
    # But n is small (<=23).
    # We can use a set of visited states (i, acc) to avoid recomputation?
    # Or just simple DFS with pruning.
    
    # Optimization: If the sum of all remaining terms is less than (target - acc), prune.
    # Precompute suffix sums of terms.
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + terms[i]
    
    def dfs(i, acc, cnt):
        if found[0]:
            return
        if cnt > 0 and acc == target:
            found[0] = True
            return
        if acc > target:
            return
        if i == n:
            return
        # Pruning: if acc + suffix_sums[i] < target, we can't reach target
        if acc + suffix_sums[i] < target:
            return
        
        # Option 1: Take term i
        dfs(i + 1, acc + terms[i], cnt + 1)
        if found[0]:
            return
        # Option 2: Skip term i
        dfs(i + 1, acc, cnt)
    
    dfs(0, Fraction(0), 0)
    return not found[0]

def _violates(a, S):
    """True if a in S violates: some subset of S\{a} sums to 1/a."""
    others = [x for x in S if x != a]
    return not _check_safe(a, others)

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    for a in S:
        if _violates(a, S):
            return False
    return True

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 85.0  # safety margin before 90s kill

    full = list(range(1, 25))
    
    # Phase 1: Greedy removal of violators
    S = list(full)
    changed = True
    while changed:
        if time.time() > _DEADLINE:
            break
        changed = False
        for a in sorted(S):
            if time.time() > _DEADLINE:
                break
            if _violates(a, S):
                S.remove(a)
                changed = True
                break  # restart scan after removal
    
    _BEST = list(S)
    
    # Phase 2: Try to add back removed elements
    removed = [x for x in full if x not in S]
    for a in sorted(removed, reverse=True):
        if time.time() > _DEADLINE:
            break
        trial = S + [a]
        if _is_valid_set(trial):
            S = trial
    
    _BEST = list(S)
    
    # Phase 3: Local search (1-swap)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            for b in list(S):
                if time.time() > _DEADLINE:
                    break
                trial = [x for x in S if x != b] + [a]
                if _is_valid_set(trial) and len(trial) > len(S):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    
    _BEST = list(S)
    
    # Phase 4: 2-swap local search (try swapping 2 out for 2 in to increase size or maintain)
    # Since we only care about max cardinality, and we are likely at a local optimum for 1-swap,
    # we try to find a valid set of size |S|+1 by replacing k elements.
    # Given time constraints, we try a limited 2-swap.
    if time.time() < _DEADLINE:
        current_size = len(S)
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        
        # Try to find a set of size current_size + 1
        # This is expensive. We do a randomized hill climbing or limited search.
        # For now, we stick to the deterministic 1-swap result if time is tight.
        # Let's try a few random restarts of the greedy phase if time permits.
        
        # Actually, let's try a smarter addition:
        # Iterate through out_set, and for each, try to add it and remove the "least useful" element?
        # No, that's complex.
        
        # Let's just ensure we return the best found.
        pass

    return list(_BEST)
# EVOLVE-BLOCK-END

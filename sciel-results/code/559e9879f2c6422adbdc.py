"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _check_validity(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [x for j, x in enumerate(sel) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add_fast(x, selected, pair_sums_mod=None, triple_sums_mod=None):
    """Fast check if x can be added to selected without violating constraints.
    Uses precomputed modular sums when available."""
    n = len(selected)
    
    # Check 1: x doesn't divide sum of any 3 from selected
    if n >= 3:
        if triple_sums_mod is not None:
            for r in triple_sums_mod[x]:
                if r == 0:
                    return False
        else:
            for t in combinations(selected, 3):
                if (t[0] + t[1] + t[2]) % x == 0:
                    return False
    
    # Check 2: no existing element a divides (pair_sum + x) for any pair from selected\{a}
    for a in selected:
        if pair_sums_mod is not None:
            # For element a, we need pairs from selected that don't include a
            # pair_sums_mod[a] contains residues of (si + sj) % a for all pairs
            # But we need to exclude pairs containing a
            # Simpler: just check all pairs, and if a is in the pair, skip
            for i in range(n):
                if selected[i] == a:
                    continue
                for j in range(i + 1, n):
                    if selected[j] == a:
                        continue
                    if (selected[i] + selected[j] + x) % a == 0:
                        return False
        else:
            others_without_a = [v for v in selected if v != a]
            if len(others_without_a) >= 2:
                for i in range(len(others_without_a)):
                    for j in range(i + 1, len(others_without_a)):
                        if (others_without_a[i] + others_without_a[j] + x) % a == 0:
                            return False
    return True

def _precompute_mods(selected):
    """Precompute modular sums for faster _can_add checks."""
    n = len(selected)
    if n < 2:
        return None, None
    
    # triple_sums_mod[x] = set of residues (si+sj+sk) % x for all triples, for x in 1..192
    # This is expensive, so we'll use a simpler approach: precompute pair sums mod a for each a
    
    pair_sums_mod = {}
    for a in selected:
        residues = set()
        for i in range(n):
            for j in range(i + 1, n):
                if selected[i] != a and selected[j] != a:
                    residues.add((selected[i] + selected[j]) % a)
        pair_sums_mod[a] = residues
    
    return pair_sums_mod, None

def _safety_score(x, selected):
    """Score how 'safe' it is to add x to selected. Higher = safer."""
    if not selected:
        return 0.0
    score = 0.0
    n = len(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        violations = 0
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (others[i] + others[j] + x) % a == 0:
                        violations += 1
        score -= violations * 10.0
        if n >= 3:
            for t in combinations(selected, 3):
                if (t[0] + t[1] + t[2]) % x == 0:
                    score -= 50.0
                    break
    score += x * 0.1
    return score

def _greedy_with_safety(order=None, time_limit=10.0):
    """Greedy with safety scoring: pick the safest candidate at each step."""
    if order is None:
        order = list(range(192, 0, -1))
    selected = []
    start = time.time()
    remaining = list(order)
    
    while remaining and time.time() - start < time_limit and len(selected) < 48:
        best_x = None
        best_score = -float('inf')
        for x in remaining:
            if _can_add_fast(x, selected):
                s = _safety_score(x, selected)
                if s > best_score:
                    best_score = s
                    best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        remaining.remove(best_x)
    
    return selected

def _greedy_large_first(order=None, time_limit=50.0):
    """Greedy: try to add elements in given order, skip if invalid."""
    if order is None:
        order = list(range(192, 0, -1))
    selected = []
    start = time.time()
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        if _can_add_fast(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit=20.0):
    """Improve by swapping elements."""
    if not _check_validity(selected):
        selected = list(selected)
        changed = True
        start_fix = time.time()
        while changed and time.time() - start_fix < 5.0:
            changed = False
            for i in range(len(selected)):
                a = selected[i]
                others = [selected[j] for j in range(len(selected)) if j != i]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        selected.remove(a)
                        changed = True
                        break
                if changed:
                    break
        return selected
    
    start = time.time()
    best = list(selected)
    
    for iteration in range(2000):
        if time.time() - start > time_limit:
            break
        
        candidates = [x for x in range(192, 0, -1) if x not in best]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        # Try to add a single element
        for x in candidates[:50]:
            if _can_add_fast(x, best):
                best.append(x)
                improved = True
                if len(best) >= 48:
                    return best
                break
        
        # Try single swap
        if not improved and len(best) >= 2:
            for i in range(len(best)):
                a = best[i]
                temp_best = best[:i] + best[i+1:]
                for x in candidates[:30]:
                    if _can_add_fast(x, temp_best):
                        best = temp_best + [x]
                        improved = True
                        break
                if improved:
                    break
        
        # Try double swap
        if not improved and len(best) >= 4:
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    temp_best = best[:i] + best[i+1:j] + best[j+1:]
                    for x in candidates[:20]:
                        if _can_add_fast(x, temp_best):
                            temp_best2 = temp_best + [x]
                            for y in candidates[:20]:
                                if y != x and _can_add_fast(y, temp_best2):
                                    best = temp_best2 + [y]
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Try a random perturbation: remove 2 random elements and try to add 2 different ones
            if len(best) >= 4 and random.random() < 0.3:
                remove_idx = random.sample(range(len(best)), 2)
                remove_idx.sort(reverse=True)
                for idx in remove_idx:
                    best.pop(idx)
                
                added = 0
                for x in candidates[:30]:
                    if _can_add_fast(x, best):
                        best.append(x)
                        added += 1
                        if added >= 2:
                            break
                
                # If we couldn't add back 2, try to recover
                if added < 2:
                    # Revert: this is complex, so just continue
                    pass
        
        if not improved:
            break
    
    return best

def _polish(selected, time_limit=10.0):
    """Final polish: try to add elements in order of safety score."""
    if not selected:
        return selected
    start = time.time()
    best = list(selected)
    
    while time.time() - start < time_limit and len(best) < 48:
        candidates = [x for x in range(192, 0, -1) if x not in best]
        if not candidates:
            break
        
        scored = []
        for x in candidates:
            if _can_add_fast(x, best):
                s = _safety_score(x, best)
                scored.append((s, x))
        
        if not scored:
            break
        
        scored.sort(reverse=True)
        added = False
        for s, x in scored[:10]:
            if _can_add_fast(x, best):
                best.append(x)
                added = True
                break
        
        if not added:
            break
    
    return best

def _divisibility_aware_order():
    """Generate an order that prioritizes numbers with fewer divisors/multiples in range."""
    # Numbers that are prime or have few factors are safer
    # Large primes are good candidates
    order = []
    for x in range(192, 0, -1):
        # Count how many numbers in [1,192] divide x or are divided by x
        divisors = sum(1 for d in range(1, x + 1) if x % d == 0)
        multiples = sum(1 for m in range(2 * x, 193, x))
        score = divisors + multiples * 2  # Multiples are more dangerous
        order.append((score, x))
    
    order.sort(key=lambda t: (t[0], -t[1]))  # Fewer connections first, then larger
    return [x for _, x in order]

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best_result = []
    
    # Strategy 1: Large-first greedy
    result = _greedy_large_first(order=list(range(192, 0, -1)), time_limit=8.0)
    if len(result) > len(best_result):
        best_result = result
    
    # Strategy 2: Safety-scored greedy
    if time.time() < deadline - 20:
        result = _greedy_with_safety(order=list(range(192, 0, -1)), time_limit=10.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 3: Divisibility-aware order
    if time.time() < deadline - 25:
        div_order = _divisibility_aware_order()
        result = _greedy_large_first(order=div_order, time_limit=6.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 4: Random restarts
    for attempt in range(25):
        if time.time() > deadline - 30:
            break
        
        random.seed(attempt * 137 + 42)
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        
        result = _greedy_large_first(order=candidates, time_limit=3.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 5: Hybrid orders
    for low in [100, 120, 140, 150, 160, 170, 180]:
        if time.time() > deadline - 25:
            break
        order = list(range(192, low - 1, -1)) + list(range(low - 1, 0, -1))
        result = _greedy_large_first(order=order, time_limit=3.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search improvement
    if time.time() < deadline - 15 and len(best_result) > 0:
        improved = _local_search(best_result, time_limit=15.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Polish
    if time.time() < deadline - 5 and len(best_result) > 0:
        improved = _polish(best_result, time_limit=8.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Final attempts
    if time.time() < deadline - 10:
        random.seed(999)
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        result = _greedy_large_first(order=candidates, time_limit=5.0)
        if len(result) > len(best_result):
            best_result = result
        if time.time() < deadline - 5:
            improved = _local_search(best_result, time_limit=5.0)
            if len(improved) > len(best_result):
                best_result = improved
            if time.time() < deadline - 2:
                improved = _polish(best_result, time_limit=3.0)
                if len(improved) > len(best_result):
                    best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END

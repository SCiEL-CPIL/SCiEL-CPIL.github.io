# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _build_forbidden_residues(selected):
    """
    For each a in selected, compute the set of residues r (mod a) such that
    there exist distinct s1, s2 in selected \ {a} with (s1 + s2) % a == (-r) % a.
    """
    bad_residues = {}
    n = len(selected)
    for i in range(n):
        a = selected[i]
        residues = set()
        for j in range(n):
            if j == i: continue
            s1 = selected[j]
            for k in range(n):
                if k == i or k == j: continue
                s2 = selected[k]
                r = -(s1 + s2) % a
                residues.add(r)
        bad_residues[a] = residues
    return bad_residues

def _can_add_fast(selected, x, bad_residues=None):
    """
    Fast check if x can be added to selected.
    1. No triple in selected sums to multiple of x.
    2. For each a in selected, x % a is not in bad_residues[a].
    """
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            return False
    
    if bad_residues is None:
        bad_residues = _build_forbidden_residues(selected)
        
    for a in selected:
        if x % a in bad_residues[a]:
            return False
            
    return True

def _divisor_score(x, selected):
    """
    Compute a score for x: how many triples in selected sum to a multiple of x.
    Lower is better (x is safer to add).
    """
    count = 0
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            count += 1
    return count

def _greedy_with_scores(order, deadline, use_scores=True):
    """Greedy construction with optional score-based reordering."""
    selected = []
    br = {}
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x, br):
            selected.append(x)
            br = _build_forbidden_residues(selected)
            
            # If using scores, re-sort remaining candidates by how many triples they divide
            if use_scores and len(selected) < 20:
                # Only do this for small sets to keep it fast
                remaining = [y for y in order if y not in set(selected) and y != x]
                if remaining:
                    scores = [(y, _divisor_score(y, selected)) for y in remaining[:50]]
                    scores.sort(key=lambda t: t[1])
                    # Reorder: put low-score items first
                    new_order = [s[0] for s in scores] + [y for y in remaining if y not in set(s[0] for s in scores)]
                    order = new_order
    
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    best_len = len(selected)
    
    br = _build_forbidden_residues(selected)
    candidates = [x for x in range(1, 193) if x not in set(selected)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding
        candidates.sort(reverse=True)
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x not in set(selected):
                if _can_add_fast(selected, x, br):
                    selected.append(x)
                    candidates.remove(x)
                    improved = True
                    best_len = len(selected)
                    br = _build_forbidden_residues(selected)
                    break
        
        # 2. Try swaps (remove 1, add 2)
        if not improved and time.time() < deadline:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_br = _build_forbidden_residues(remaining)
                
                added = []
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if len(added) == 2:
                        break
                    if x not in set(remaining) and x != removed:
                        test_set = remaining + added
                        if added:
                            test_br = _build_forbidden_residues(test_set)
                        else:
                            test_br = rem_br
                            
                        if _can_add_fast(test_set, x, test_br):
                            added.append(x)
                            if len(added) == 2:
                                break
                            
                if len(added) == 2:
                    new_selected = remaining + added
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        best_len = len(selected)
                        br = _build_forbidden_residues(selected)
                        improved = True
                        break

    return selected

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        result = _greedy_with_scores(rand_order, deadline, use_scores=True)
        if len(result) > len(best):
            best = result
    
    # Also try structured orderings
    structured_orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        list(range(96, 0, -1)) + list(range(97, 193)),
        [x for x in range(192, 0, -1) if x > 64],
        # Divisor-aware: sort by number of divisors (fewer divisors = safer)
        sorted(range(1, 193), key=lambda x: len([d for d in range(1, x+1) if x % d == 0])),
        # Primes first
        [x for x in range(1, 193) if all(x % d != 0 for d in range(2, int(x**0.5)+1))] + 
        [x for x in range(1, 193) if not all(x % d != 0 for d in range(2, int(x**0.5)+1))],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy_with_scores(order, deadline, use_scores=False)
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best = _random_greedy(deadline)
    
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
            
    if time.time() < deadline - 2:
        current_best = best
        current_len = len(best)
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            rand_order = list(range(1, 193))
            random.shuffle(rand_order)
            result = _greedy_with_scores(rand_order, deadline, use_scores=True)
            if len(result) > current_len:
                current_best = result
                current_len = len(result)
                if time.time() < deadline - 2:
                    improved = _local_improve(current_best, deadline)
                    if len(improved) > current_len:
                        current_best = improved
                        current_len = len(improved)
        best = current_best

    if not _is_valid(best):
        best = _greedy_with_scores(list(range(192, 0, -1)), deadline, use_scores=False)
    
    return best
# EVOLVE-BLOCK-END

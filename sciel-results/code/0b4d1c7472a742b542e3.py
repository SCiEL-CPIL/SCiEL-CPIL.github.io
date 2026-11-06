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
    Wait, the condition is (s1 + s2 + x) % a == 0 => x % a == -(s1+s2) % a.
    So for each a, we compute the set of bad residues for x.
    """
    bad_residues = {}
    n = len(selected)
    for i in range(n):
        a = selected[i]
        residues = set()
        # We need distinct s1, s2 from selected \ {a}
        # Iterate over pairs
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
    # Check 1: Triples in selected
    # This is O(N^3). We can optimize by precomputing triple sums?
    # But N is small.
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            return False
    
    # Check 2: Existing elements
    if bad_residues is None:
        bad_residues = _build_forbidden_residues(selected)
        
    for a in selected:
        if x % a in bad_residues[a]:
            return False
            
    return True

def _greedy(order, deadline, bad_residues=None):
    """Greedy construction with given order."""
    selected = []
    br = {} # Current bad residues
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x, br):
            selected.append(x)
            # Update bad residues
            # This is expensive to recompute fully. 
            # Given N <= 48, recomputing is O(N^3) which is ~100k ops. 
            # With 192 steps, 19M ops. Acceptable.
            br = _build_forbidden_residues(selected)
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    best_len = len(selected)
    
    # Precompute bad residues for current set
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
                    break # Restart loop to re-evaluate with new set
                    
        # 2. Try swaps (remove 1, add 2)
        if not improved and time.time() < deadline:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_br = _build_forbidden_residues(remaining)
                
                # Try to add 2 elements to remaining
                added = []
                # Sort candidates by some heuristic? Maybe just iterate.
                # To be efficient, we can check pairs?
                # Let's just try adding one by one.
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if len(added) == 2:
                        break
                    if x not in set(remaining) and x != removed:
                        if _can_add_fast(remaining + added, x, rem_br if not added else None):
                            # If we add x, we need to update rem_br for the next addition
                            # This is getting complex. Let's simplify: 
                            # Just check if we can add x to (remaining + added).
                            # If added is empty, use rem_br.
                            # If added is not empty, recompute.
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
                        break # Restart loop

    return selected

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    for attempt in range(50): # Increased attempts
        if time.time() > deadline:
            break
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        result = _greedy(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Also try structured orderings
    structured_orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        # Alternating high/low
        list(range(96, 0, -1)) + list(range(97, 193)),
        # Primes-like: numbers with few divisors
        [x for x in range(192, 0, -1) if x > 64],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin for 90s limit
    
    # Phase 1: Randomized greedy to find good initial solution
    best = _random_greedy(deadline)
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: More local improvement rounds
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
            
    # Phase 4: Simulated Annealing / Random Perturbation
    if time.time() < deadline - 2:
        current_best = best
        current_len = len(best)
        # Try to find a larger set by randomizing the order and running greedy again, 
        # but starting from a subset of the current best?
        # Or just run more random greedy with tighter deadline?
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Random subset of current best to restart?
            # Let's just do more random greedy
            rand_order = list(range(1, 193))
            random.shuffle(rand_order)
            result = _greedy(rand_order, deadline)
            if len(result) > current_len:
                current_best = result
                current_len = len(result)
                # Local improve the new best
                if time.time() < deadline - 2:
                    improved = _local_improve(current_best, deadline)
                    if len(improved) > current_len:
                        current_best = improved
                        current_len = len(improved)
        best = current_best

    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(192, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END

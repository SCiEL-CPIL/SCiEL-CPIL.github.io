"""Combined approach: precompute bad triples with efficient LCM table,
use multiple greedy strategies + local search to maximize cardinality.
Key: for triple {a,b,c} to be bad, lcm(a,b)==lcm(a,c)==lcm(b,c)==L.
This means each of a,b,c divides L, and for each pair, the lcm is exactly L.
"""
import time
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    deadline = time.time() + 70  # safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table for all pairs
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples: (a, b, c) with a < b < c where all pairwise LCMs equal
    bad_triples_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples_set.add((a, b, c))
    
    if time.time() > deadline:
        return [1]
    
    # Precompute: for each pair (a,b), which c values form a bad triple
    # forbidden_by_pair[(a,b)] = set of c where (min(a,b,c), mid, max) is bad
    forbidden_by_pair = {}
    for (a, b, c) in bad_triples_set:
        # For pair (a,b), c is forbidden
        forbidden_by_pair.setdefault((a, b), set()).add(c)
        # For pair (a,c), b is forbidden
        forbidden_by_pair.setdefault((a, c), set()).add(b)
        # For pair (b,c), a is forbidden
        forbidden_by_pair.setdefault((b, c), set()).add(a)
    
    def can_add(v, selected_set, selected_list):
        """Check if v can be added to selected without forming a bad triple."""
        for i in range(len(selected_list)):
            a = selected_list[i]
            # Check pair (a, v) against all other selected elements
            # Actually, we need to check: for every pair (x, y) in selected,
            # is {x, y, v} a bad triple?
            # More efficiently: for each x in selected, check if (x, v) pair
            # has any other selected element that forms a bad triple
            pass
        
        # Direct check: for every pair (x, y) in selected, check if v is forbidden
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                key1 = (a, b) if a < b else (b, a)
                if v in forbidden_by_pair.get(key1, ()):
                    return False
        return True
    
    def greedy(order):
        """Greedy: add numbers in given order if no bad triple is formed."""
        selected = set()
        sel_list = []
        for x in order:
            if can_add(x, selected, sel_list):
                selected.add(x)
                sel_list.append(x)
        return sel_list
    
    # Try multiple greedy orderings
    candidates = []
    
    # Strategy 1: Natural order
    candidates.append(greedy(range(1, N + 1)))
    
    # Strategy 2: Reverse order
    candidates.append(greedy(range(N, 0, -1)))
    
    # Strategy 3: Primes and 1
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    primes = [1] + [p for p in range(2, N + 1) if is_prime(p)]
    candidates.append(greedy(primes))
    
    # Strategy 4: Sorted by number of prime factors (ascending)
    def total_omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order4 = sorted(range(1, N + 1), key=lambda x: (total_omega(x), x))
    candidates.append(greedy(order4))
    
    # Strategy 5: Larger numbers first (reverse natural)
    candidates.append(greedy(range(N, 0, -1)))
    
    # Strategy 6: Powers of 2 and their multiples
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128]
    candidates.append(greedy(powers2))
    
    # Strategy 7: Odd numbers
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    candidates.append(greedy(odds))
    
    # Random orderings
    import random
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    # Pick the best
    best = max(candidates, key=len)
    best_set = set(best)
    
    # Local search: try to add elements, then try swaps
    def local_search(sel, deadline):
        sel_set = set(sel)
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each non-selected vertex
            for v in range(1, N + 1):
                if v in sel_set:
                    continue
                if can_add(v, sel_set, sel_list):
                    sel_set.add(v)
                    sel_list.append(v)
                    improved = True
            # Try removing one element and adding two
            if not improved:
                for remove_idx in range(len(sel_list)):
                    if time.time() > deadline:
                        break
                    removed = sel_list[remove_idx]
                    trial_set = set(sel_set)
                    trial_set.discard(removed)
                    trial_list = [x for x in sel_list if x != removed]
                    added = 0
                    for v in range(1, N + 1):
                        if v in trial_set:
                            continue
                        if can_add(v, trial_set, trial_list):
                            trial_set.add(v)
                            trial_list.append(v)
                            added += 1
                            if time.time() > deadline:
                                break
                    if added > 1:  # Net gain of at least 1
                        sel_set = trial_set
                        sel_list = trial_list
                        improved = True
                        break
        return sel_list
    
    if time.time() < deadline:
        best_grown = local_search(best, deadline)
        if len(best_grown) > len(best):
            best = best_grown
            best_set = set(best)
    
    # Validate
    def is_valid(sel):
        s = set(sel)
        if len(s) != len(sel):
            return False
        if len(s) == 0:
            return False
        for a, b, c in combinations(sel, 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True
    
    if not is_valid(best):
        # Fallback: just return a known valid set
        return [1]
    
    if not best:
        return [1]
    
    return best

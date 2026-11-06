import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    N = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples and build conflict structures
    bad_triples = []
    conflict_count = [0] * (N + 1)
    pair_conflicts = {}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1
                    pair_conflicts.setdefault((a, b), set()).add(c)
                    pair_conflicts.setdefault((a, c), set()).add(b)
                    pair_conflicts.setdefault((b, c), set()).add(a)
    
    if time.time() > deadline:
        return [1]
    
    # Build per-element conflict structure: for each v, map pair->set of conflicting third elements
    # Actually, we need: given v and current selected set S, check if any pair in S conflicts with v
    # More efficient: for each v, precompute the set of pairs (a,b) such that (a,b,v) is a bad triple
    v_conflicting_pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in bad_triples:
        v_conflicting_pairs[a].add((b, c) if b < c else (c, b))
        v_conflicting_pairs[b].add((a, c) if a < c else (c, a))
        v_conflicting_pairs[c].add((a, b))
    
    # Also build: for each v, the set of elements that conflict with v in the sense that
    # there exists a pair containing v that is bad with that element
    # Actually the key insight: v cannot be added if there exist a,b in S such that (a,b,v) is bad
    # So for each v, we need to check if any pair from S is in v_conflicting_pairs[v]
    
    def can_add(v, selected_set, selected_list):
        """Check if v can be added without forming a bad triple."""
        if not v_conflicting_pairs[v]:
            return True
        # Check all pairs in selected_list
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                key = (a, b) if a < b else (b, a)
                if key in v_conflicting_pairs[v]:
                    return False
        return True
    
    # Even faster: precompute for each pair (a,b) the set of v's that conflict
    # Then can_add(v, S) checks: for each pair in S, is v in conflict_set[pair]?
    # That's what pair_conflicts already gives us.
    
    # Let's use a different approach: maintain a "conflict graph" 
    # For efficiency, track for each v which pairs in S would conflict
    # Use bitsets or just iterate over selected_list pairs
    
    def greedy(order):
        """Greedy: add numbers in given order if no bad triple is formed."""
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Multiple greedy strategies
    candidates = []
    
    # Strategy 1: Natural order
    candidates.append(greedy(list(range(1, N + 1))))
    
    # Strategy 2: Reverse order
    candidates.append(greedy(list(range(N, 0, -1))))
    
    # Strategy 3: Sorted by conflict count (ascending)
    order3 = sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x))
    candidates.append(greedy(order3))
    
    # Strategy 4: Sorted by conflict count (descending)
    order4 = sorted(range(1, N + 1), key=lambda x: (-conflict_count[x], x))
    candidates.append(greedy(order4))
    
    # Strategy 5: Primes and 1
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    primes = [1] + [p for p in range(2, N + 1) if is_prime(p)]
    candidates.append(greedy(primes))
    
    # Strategy 6: Powers of 2
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128]
    candidates.append(greedy(powers2))
    
    # Strategy 7: Odd numbers
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    candidates.append(greedy(odds))
    
    # Strategy 8: Numbers with few prime factors
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
    
    order8 = sorted(range(1, N + 1), key=lambda x: (total_omega(x), x))
    candidates.append(greedy(order8))
    
    # Strategy 9: Sorted by number of prime factors (distinct)
    def distinct_omega(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order9 = sorted(range(1, N + 1), key=lambda x: (distinct_omega(x), x))
    candidates.append(greedy(order9))
    
    # Strategy 10: Even numbers
    evens = [x for x in range(2, N + 1, 2)]
    candidates.append(greedy(evens))
    
    # Random restarts with different seeds - more of them since we have time
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    # Pick best candidate
    best = max(candidates, key=len)
    best_set = set(best)
    
    # Local search: try to add elements (fast version)
    def local_search_add(sel_list, deadline):
        sel_set = set(sel_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel_set:
                    continue
                if can_add(v, sel_set, sel_list):
                    sel_list.append(v)
                    sel_set.add(v)
                    improved = True
        return sel_list
    
    # Local search: try swaps (remove one, add more)
    def local_search_swap(sel_list, deadline):
        sel_set = set(sel_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_idx in range(len(sel_list)):
                if time.time() > deadline:
                    break
                removed = sel_list[remove_idx]
                trial_list = [x for x in sel_list if x != removed]
                trial_set = set(trial_list)
                added = 0
                for v in range(1, N + 1):
                    if v in trial_set:
                        continue
                    if can_add(v, trial_set, trial_list):
                        trial_list.append(v)
                        trial_set.add(v)
                        added += 1
                if added > 1:
                    sel_list = trial_list
                    sel_set = trial_set
                    improved = True
                    break
        return sel_list
    
    # Local search: 2-swap (remove two, add more)
    def local_search_2swap(sel_list, deadline):
        sel_set = set(sel_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(sel_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    r1, r2 = sel_list[i], sel_list[j]
                    trial_list = [x for x in sel_list if x != r1 and x != r2]
                    trial_set = set(trial_list)
                    added = 0
                    for v in range(1, N + 1):
                        if v in trial_set:
                            continue
                        if can_add(v, trial_set, trial_list):
                            trial_list.append(v)
                            trial_set.add(v)
                            added += 1
                    if added > 2:
                        sel_list = trial_list
                        sel_set = trial_set
                        improved = True
                        break
                if improved:
                    break
        return sel_list
    
    if time.time() < deadline - 5:
        best = local_search_add(best, deadline - 5)
    
    if time.time() < deadline - 2:
        best = local_search_swap(best, deadline - 2)
    
    if time.time() < deadline:
        best = local_search_2swap(best, deadline)
    
    # Final validation
    def is_valid(sel):
        s = set(sel)
        if len(s) != len(sel) or len(s) == 0:
            return False
        for a, b, c in combinations(sel, 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True
    
    if not is_valid(best):
        return [1]
    
    if not best:
        return [1]
    
    return best

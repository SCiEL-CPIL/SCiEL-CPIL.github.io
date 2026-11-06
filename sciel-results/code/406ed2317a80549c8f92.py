import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad triples and conflict counts
    bad_triples = []
    conflict_count = [0] * (N + 1)
    # For each pair (a,b), track which c's would form a bad triple
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
                    # Record conflicts for each pair
                    pair_conflicts.setdefault((a, b), set()).add(c)
                    pair_conflicts.setdefault((a, c), set()).add(b)
                    pair_conflicts.setdefault((b, c), set()).add(a)
    
    if time.time() > deadline:
        return [1]
    
    def can_add(v, selected_list):
        """Check if v can be added to selected_list without forming a bad triple."""
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                key = (a, b) if a < b else (b, a)
                if v in pair_conflicts.get(key, ()):
                    return False
        return True
    
    def greedy(order):
        """Greedy: add numbers in given order if no bad triple is formed."""
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    # Multiple greedy strategies
    candidates = []
    
    # Strategy 1: Natural order
    candidates.append(greedy(list(range(1, N + 1))))
    
    # Strategy 2: Reverse order
    candidates.append(greedy(list(range(N, 0, -1))))
    
    # Strategy 3: Sorted by conflict count (ascending) - least constrained first
    order3 = sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x))
    candidates.append(greedy(order3))
    
    # Strategy 4: Sorted by conflict count (descending) - most constrained first
    order4 = sorted(range(1, N + 1), key=lambda x: (-conflict_count[x], x))
    candidates.append(greedy(order4))
    
    # Strategy 5: Primes and 1 (pairwise coprime, safe)
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
    
    # Random restarts with different seeds
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    # Pick best candidate
    best = max(candidates, key=len)
    
    # Local search: try to add elements
    def local_search_add(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel_list:
                    continue
                if can_add(v, sel_list):
                    sel_list.append(v)
                    improved = True
        return sel_list
    
    # Local search: try swaps (remove one, add more)
    def local_search_swap(sel, deadline):
        sel_list = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_idx in range(len(sel_list)):
                if time.time() > deadline:
                    break
                removed = sel_list[remove_idx]
                trial = [x for x in sel_list if x != removed]
                added = 0
                for v in range(1, N + 1):
                    if v in trial:
                        continue
                    if can_add(v, trial):
                        trial.append(v)
                        added += 1
                if added > 1:
                    sel_list = trial
                    improved = True
                    break
        return sel_list
    
    if time.time() < deadline:
        best = local_search_add(best, deadline)
    
    if time.time() < deadline:
        best = local_search_swap(best, deadline)
    
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

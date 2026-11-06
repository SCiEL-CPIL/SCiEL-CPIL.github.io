import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    N = 128

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # For each element v, precompute set of pairs (a,b) with a<b such that (a,b,v) is bad
    # We'll store as: conflict_pairs[v] = list of (a, b) where a < b and (a,b,v) is a bad triple
    conflict_pairs = [[] for _ in range(N + 1)]
    # Also: for each pair (a,b), the set of v that conflict
    pair_forbidden = {}  # (a,b) -> set of v
    for (a, b, c) in bad_triples:
        # v=a conflicts with pair (b,c)
        conflict_pairs[a].append((b, c))
        pair_forbidden.setdefault((b, c), set()).add(a)
        # v=b conflicts with pair (a,c)
        conflict_pairs[b].append((a, c))
        pair_forbidden.setdefault((a, c), set()).add(b)
        # v=c conflicts with pair (a,b)
        conflict_pairs[c].append((a, b))
        pair_forbidden.setdefault((a, b), set()).add(c)

    # Convert to sets for fast lookup
    conflict_pairs_set = [set(p) for p in conflict_pairs]

    if time.time() > deadline:
        return [1]

    def can_add_fast(v, selected_list, pair_block_count):
        """
        O(1) check: pair_block_count[v] counts how many pairs in selected_list
        would form a bad triple with v. If 0, v can be added.
        """
        return pair_block_count[v] == 0

    def add_element(v, selected_list, selected_set, pair_block_count):
        """Add v to selection, updating pair_block_count for all other elements."""
        selected_list.append(v)
        selected_set.add(v)
        # For each existing element x in selected_list (except v), 
        # the pair (x, v) may forbid some elements
        # Actually: for each x in selected_list[:-1], check if (x,v) is a key in pair_forbidden
        # If so, increment pair_block_count for each forbidden element
        for x in selected_list[:-1]:
            key = (x, v) if x < v else (v, x)
            if key in pair_forbidden:
                for w in pair_forbidden[key]:
                    pair_block_count[w] += 1

    def remove_element(v, selected_list, selected_set, pair_block_count):
        """Remove v from selection, updating pair_block_count."""
        selected_list.remove(v)
        selected_set.discard(v)
        for x in selected_list:
            key = (x, v) if x < v else (v, x)
            if key in pair_forbidden:
                for w in pair_forbidden[key]:
                    pair_block_count[w] -= 1

    def greedy(order):
        """Greedy with O(1) can_add using pair_block_count."""
        selected_list = []
        selected_set = set()
        pair_block_count = [0] * (N + 1)
        for x in order:
            if x in selected_set:
                continue
            if pair_block_count[x] == 0:
                add_element(x, selected_list, selected_set, pair_block_count)
        return selected_list

    # Multiple greedy strategies
    candidates = []

    # Strategy 1: Natural order
    candidates.append(greedy(list(range(1, N + 1))))

    # Strategy 2: Reverse order
    candidates.append(greedy(list(range(N, 0, -1))))

    # Strategy 3: Sorted by conflict count (ascending) - fewer conflicts first
    conflict_count = [0] * (N + 1)
    for v in range(1, N + 1):
        conflict_count[v] = len(conflict_pairs_set[v])
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

    # Strategy 9: Sorted by distinct prime factors
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

    # Random restarts
    random.seed(42)
    for trial in range(300):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy(order))

    # Pick best candidate
    best = max(candidates, key=len)
    best_set = set(best)

    # Local search: add elements (now fast with O(1) check)
    def local_search_add(sel_list, deadline):
        sel_set = set(sel_list)
        pair_block_count = [0] * (N + 1)
        # Rebuild pair_block_count
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                key = (a, b) if a < b else (b, a)
                if key in pair_forbidden:
                    for w in pair_forbidden[key]:
                        pair_block_count[w] += 1
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if v in sel_set:
                    continue
                if pair_block_count[v] == 0:
                    add_element(v, sel_list, sel_set, pair_block_count)
                    improved = True
        return sel_list

    # Local search: 1-swap (remove one, try to add multiple)
    def local_search_1swap(sel_list, deadline):
        sel_set = set(sel_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_idx in range(len(sel_list)):
                if time.time() > deadline:
                    break
                removed = sel_list[remove_idx]
                # Remove and rebuild
                trial_list = [x for x in sel_list if x != removed]
                trial_set = set(trial_list)
                # Rebuild pair_block_count for trial
                pbc = [0] * (N + 1)
                for i in range(len(trial_list)):
                    for j in range(i + 1, len(trial_list)):
                        a, b = trial_list[i], trial_list[j]
                        key = (a, b) if a < b else (b, a)
                        if key in pair_forbidden:
                            for w in pair_forbidden[key]:
                                pbc[w] += 1
                # Try to add as many as possible
                added = 0
                for v in range(1, N + 1):
                    if v in trial_set:
                        continue
                    if pbc[v] == 0:
                        trial_list.append(v)
                        trial_set.add(v)
                        # Update pbc
                        for x in trial_list[:-1]:
                            key = (x, v) if x < v else (v, x)
                            if key in pair_forbidden:
                                for w in pair_forbidden[key]:
                                    pbc[w] += 1
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
        n = len(sel_list)
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    r1, r2 = sel_list[i], sel_list[j]
                    trial_list = [x for x in sel_list if x != r1 and x != r2]
                    trial_set = set(trial_list)
                    pbc = [0] * (N + 1)
                    for ii in range(len(trial_list)):
                        for jj in range(ii + 1, len(trial_list)):
                            a, b = trial_list[ii], trial_list[jj]
                            key = (a, b) if a < b else (b, a)
                            if key in pair_forbidden:
                                for w in pair_forbidden[key]:
                                    pbc[w] += 1
                    added = 0
                    for v in range(1, N + 1):
                        if v in trial_set:
                            continue
                        if pbc[v] == 0:
                            trial_list.append(v)
                            trial_set.add(v)
                            for x in trial_list[:-1]:
                                key = (x, v) if x < v else (v, x)
                                if key in pair_forbidden:
                                    for w in pair_forbidden[key]:
                                        pbc[w] += 1
                            added += 1
                    if added > 2:
                        sel_list = trial_list
                        sel_set = trial_set
                        n = len(sel_list)
                        improved = True
                        break
                if improved:
                    break
        return sel_list

    # Apply local searches
    if time.time() < deadline - 10:
        best = local_search_add(best, deadline - 10)

    if time.time() < deadline - 5:
        best = local_search_1swap(best, deadline - 5)

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
        # Fallback: rebuild from scratch with validation
        return [1]

    if not best:
        return [1]

    return best

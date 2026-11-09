"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80  # Safety margin
    
    N = 128
    # Precompute GCD table for fast access
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad[a][b] = set of x such that (a,b,x) is a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for x in range(1, N + 1):
                if x == a or x == b:
                    continue
                if gcd_table[a][x] == gab and gcd_table[b][x] == gab:
                    bad[a][b].add(x)
                    bad[b][a].add(x)

    def can_add(selected, x):
        """
        Check if x can be added to selected without creating a bad triple.
        A bad triple (a, b, x) exists if gcd(a,b) == gcd(a,x) == gcd(b,x).
        """
        n = len(selected)
        # Iterate over pairs in selected
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                if x in bad[a][b]:
                    return False
        return True

    def greedy_build(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected

    def local_search_fill(selected):
        """
        Try to add any remaining element that fits.
        This is a local search with 'add' moves.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        changed = True
        while changed:
            changed = False
            for x in all_nums - selected_set:
                if can_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    changed = True
        return selected

    def local_search_swap(selected):
        """
        Try to swap elements to potentially unlock more additions or increase size.
        A swap (remove a, add b) keeps size same. It might allow future adds.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        candidates = list(all_nums - selected_set)
        
        max_swaps = 200
        swaps_done = 0
        
        while swaps_done < max_swaps and time.time() < deadline - 5:
            improved_this_round = False
            for a in selected:
                if time.time() > deadline - 5:
                    break
                for b in candidates:
                    if b in selected_set:
                        continue
                    # Try removing a and adding b
                    temp_selected = [x for x in selected if x != a]
                    
                    if can_add(temp_selected, b):
                        new_selected = temp_selected + [b]
                        # Try to fill again
                        new_selected = local_search_fill(new_selected)
                        
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            selected_set = set(selected)
                            candidates = list(all_nums - selected_set)
                            improved_this_round = True
                            break
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
            swaps_done += 1
            
        return selected

    nums = list(range(1, N + 1))
    best = []

    # Heuristic orderings
    orderings = []
    
    # 1. Ascending
    orderings.append(nums[:])
    # 2. Descending
    orderings.append(nums[::-1])
    
    # 3. By number of divisors (descending)
    def div_count(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    orderings.append(sorted(nums, key=lambda x: -div_count(x)))
    orderings.append(sorted(nums, key=lambda x: div_count(x)))
    
    # 4. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)

    # 5. Random restarts
    rng = random.Random(42)
    for _ in range(40):
        o = nums[:]
        rng.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        # Greedy build
        cand = greedy_build(order)
        
        # Local search fill (add any possible)
        cand = local_search_fill(cand)
        
        # Local search swap (try to improve if stuck)
        if time.time() < deadline - 5:
            cand = local_search_swap(cand)
        
        if len(cand) > len(best):
            best = cand
            
    # Ensure non-empty
    if not best:
        best = [1]
        
    return best
# EVOLVE-BLOCK-END

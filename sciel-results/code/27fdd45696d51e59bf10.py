import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin under 90s

    # Precompute divisibility: div[a][s] = True if a divides s
    # a in [1,256], s in [2,512]
    div_table = [[False] * 513 for _ in range(257)]
    for a in range(1, 257):
        for s in range(2*a, 513, a):
            div_table[a][s] = True

    def valid(S):
        """Check if set S is valid: for all a<b<=c in S, a does not divide b+c."""
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(i+1, len(L)):
                b = L[j]
                for k in range(j, len(L)):
                    c = L[k]
                    if div_table[a][b+c]:
                        return False
        return True

    def can_add(S, x):
        """Check if adding x to S keeps it valid. S is a set."""
        L = sorted(S)
        # Case 1: x is smallest: x < b <= c, x | (b+c)
        for i, b in enumerate(L):
            if b <= x:
                continue
            for c in L[i:]:
                if div_table[x][b+c]:
                    return False
        # Case 2: x is middle: a < x <= c, a | (x+c)
        for a in L:
            if a >= x:
                continue
            for c in L:
                if c < x:
                    continue
                if div_table[a][x+c]:
                    return False
        # Case 3: x is largest: a < b <= x, a | (b+x)
        for i, a in enumerate(L):
            for b in L[i+1:]:
                if b > x:
                    break
                if div_table[a][b+x]:
                    return False
        return True

    # Removed conflict_count as it is too expensive for repeated use in sorting.
    # We will rely on simple heuristics for candidate ordering in local search.

    def greedy_from_tail(tail_start, order):
        """Greedy: start with [tail_start, 256], add elements in given order."""
        S = set(range(tail_start, 257))
        for candidate in order:
            if candidate in S:
                continue
            if can_add(S, candidate):
                S.add(candidate)
        return S

    def local_search(S, max_time):
        """Try to improve S by removing one element and greedily refilling."""
        S = set(S)
        best = S
        best_size = len(S)
        end = start_time + max_time
        
        # Precompute a list of all numbers 1..256 for fast candidate generation
        all_nums = list(range(1, 257))
        
        while time.time() < end:
            improved = False
            L = sorted(S)
            
            # Try single removals
            for i in range(len(L)):
                if time.time() > end:
                    break
                a = L[i]
                S2 = set(S)
                S2.remove(a)
                
                # Try adding elements not in S2.
                # Heuristic: Try larger numbers first (often less restrictive for 'a' being small)
                # or just iterate through all. To speed up, we only try a subset or break early if no improvement found in first few.
                # Actually, we must try all to be sure we maximize the refill.
                # But we can speed up can_add.
                
                for candidate in all_nums:
                    if candidate in S2:
                        continue
                    if can_add(S2, candidate):
                        S2.add(candidate)
                        
                if len(S2) > best_size:
                    best = S2
                    best_size = len(S2)
                    S = set(S2)
                    improved = True
                    break
            if improved:
                continue
            # If no single removal improves, stop. Double removal is too expensive.
            break
            
        return best

    # Try multiple tail starts and orders
    best = set(range(171, 257))
    best_size = len(best)

    tail_starts = [160, 165, 170, 175, 180, 185, 190, 195, 200, 210, 220, 230]
    orders = [
        list(range(1, 171)),
        list(range(170, 0, -1)),
    ]

    # Add randomized orders with different seeds (Reduced count to save time)
    rng = random.Random(42)
    for _ in range(3):
        o = list(range(1, 171))
        rng.shuffle(o)
        orders.append(o)

    for ts in tail_starts:
        if time.time() > deadline - 20:
            break
        for order in orders:
            if time.time() > deadline - 15:
                break
            S = greedy_from_tail(ts, order)
            if len(S) > best_size:
                best = S
                best_size = len(S)

    # Local search on the best
    # Allow most of the remaining time for local search as it's the main improvement driver
    if time.time() < deadline - 2:
        best = local_search(best, max_time=deadline - 3)
        best_size = len(best)

    # Verify validity (should always be true, but just in case)
    if not valid(best):
        # Fallback to a known valid set
        best = set(range(171, 257))

    return sorted(best)

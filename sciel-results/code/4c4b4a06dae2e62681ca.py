"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin (90s limit, leave buffer)
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Precompute bad triples: for each element x, which pairs (y,z) with y<z form a bad triple with x
    # bad_with[x] = set of (y, z) such that gcd(x,y)==gcd(y,z)==gcd(z,x)
    bad_with = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                g_bc = gcd_mat[b][c]
                g_ac = gcd_mat[a][c]
                if g_ab == g_bc == g_ac:
                    bad_with[a].add((b, c))
                    bad_with[b].add((a, c))
                    bad_with[c].add((a, b))
    
    def is_valid_addition(selected, x):
        """Check if adding x to selected creates no bad triple."""
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                if (sel_list[i], sel_list[j]) in bad_with[x]:
                    return False
        return True
    
    def count_bad_triples(selected):
        """Count bad triples in selected set."""
        sel = list(selected)
        count = 0
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                for k in range(j + 1, len(sel)):
                    a, b, c = sel[i], sel[j], sel[k]
                    if gcd_mat[a][b] == gcd_mat[b][c] == gcd_mat[a][c]:
                        count += 1
        return count
    
    # Greedy with different orderings
    def greedy(order):
        selected = []
        for x in order:
            if is_valid_addition(selected, x):
                selected.append(x)
        return set(selected)
    
    # Try multiple orderings
    best = set()
    
    # Natural order
    best = greedy(list(range(1, N + 1)))
    
    # Reverse order
    if time.time() < deadline - 5:
        s = greedy(list(range(N, 0, -1)))
        if len(s) > len(best):
            best = s
    
    # Order by number of prime factors (more composite first)
    if time.time() < deadline - 5:
        def num_prime_factors(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        order = sorted(range(1, N + 1), key=lambda x: (-num_prime_factors(x), x))
        s = greedy(order)
        if len(s) > len(best):
            best = s
    
    # Order by number of divisors
    if time.time() < deadline - 5:
        def num_divisors(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count
        order = sorted(range(1, N + 1), key=lambda x: (-num_divisors(x), x))
        s = greedy(order)
        if len(s) > len(best):
            best = s
    
    # Local search: try to add elements, or swap to improve
    def local_search(start_set, max_iter=500):
        current = set(start_set)
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 3:
            improved = False
            iteration += 1
            # Try adding each missing element
            for x in range(1, N + 1):
                if x not in current:
                    if is_valid_addition(current, x):
                        current.add(x)
                        improved = True
            # Try swaps: remove one, add one
            for remove in list(current):
                for add in range(1, N + 1):
                    if add not in current:
                        trial = set(current)
                        trial.discard(remove)
                        trial.add(add)
                        # Check validity
                        valid = True
                        trial_list = list(trial)
                        for i in range(len(trial_list)):
                            for j in range(i + 1, len(trial_list)):
                                if (trial_list[i], trial_list[j]) in bad_with[add]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Only swap if it helps (same size but different composition might help later)
                            # Actually, we only want to swap if we can then add more
                            pass  # Swaps that keep same size don't directly help cardinality
        return current
    
    if time.time() < deadline - 10:
        improved_set = local_search(best, max_iter=200)
        if len(improved_set) > len(best):
            best = improved_set
    
    # Randomized greedy with restarts
    import random
    random.seed(42)
    for restart in range(50):
        if time.time() >= deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy(order)
        # Quick local improvement
        for x in range(1, N + 1):
            if x not in s and is_valid_addition(s, x):
                s.add(x)
        if len(s) > len(best):
            best = s
    
    # Final local search on best
    if time.time() < deadline - 3:
        final = local_search(best, max_iter=300)
        if len(final) > len(best):
            best = final
    
    return list(best)
# EVOLVE-BLOCK-END

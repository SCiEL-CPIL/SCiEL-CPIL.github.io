"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s wall-clock
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute lcm > 256 compatibility
    # compatible[i][j] = True if lcm(i, j) > 256
    # We only care about numbers 2..256, index 0..254 for 2..256
    N = 255  # numbers 2..256
    nums = list(range(2, 257))
    
    # Precompute compatibility matrix
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = nums[i], nums[j]
            compat[i][j] = compat[j][i] = (lcm(a, b) > 256)
    
    # Key insight: if x is in the set, then no multiple kx (k>=2, kx<=256) can be in the set
    # because lcm(x, kx) = kx <= 256
    # So for each x, choosing x excludes all its multiples.
    # Conversely, if we want to maximize sum of 1/x, we prefer smaller x.
    
    # Strategy: Use a greedy + local search approach
    # 1. Start with a greedy solution (pick smallest compatible numbers)
    # 2. Try local improvements (swap, add, remove)
    
    # Greedy: sort by 1/x descending (i.e., x ascending), pick if compatible with all selected
    def greedy():
        selected = []
        for i in range(N):
            if all(compat[i][j] for j in selected):
                selected.append(i)
        return selected
    
    # Better greedy: try multiple orderings
    def greedy_order(order):
        selected = []
        for i in order:
            if all(compat[i][j] for j in selected):
                selected.append(i)
        return selected
    
    # Score function
    def score(selected):
        s = Fraction(0, 1)
        for i in selected:
            s += Fraction(1, nums[i])
        return s
    
    # Try greedy with natural order
    best = greedy()
    best_score = score(best)
    
    # Try greedy with different orderings
    # Order by: primes first (they have fewer conflicts)
    def is_prime(n):
        if n < 2:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                return False
        return True
    
    primes_order = [i for i in range(N) if is_prime(nums[i])]
    composites_order = [i for i in range(N) if not is_prime(nums[i])]
    
    # Try primes first
    sol = greedy_order(primes_order + composites_order)
    s = score(sol)
    if s > best_score:
        best_score = s
        best = sol
    
    # Try composites first
    sol = greedy_order(composites_order + primes_order)
    s = score(sol)
    if s > best_score:
        best_score = s
        best = sol
    
    # Try order by number of conflicts (fewer conflicts first)
    conflict_count = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j and not compat[i][j]:
                conflict_count[i] += 1
    
    sorted_by_conflicts = sorted(range(N), key=lambda i: (conflict_count[i], nums[i]))
    sol = greedy_order(sorted_by_conflicts)
    s = score(sol)
    if s > best_score:
        best_score = s
        best = sol
    
    # Local search: try to improve by swapping elements
    def local_search(current, max_iters=10000):
        current = list(current)
        cur_score = score(current)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try removing one element and adding another
            for i in range(len(current)):
                candidate = [j for j in range(N) if j not in current and all(compat[j][k] for k in current)]
                # Try adding each candidate
                for add_idx in candidate:
                    new_set = current + [add_idx]
                    new_score = cur_score + Fraction(1, nums[add_idx])
                    if new_score > cur_score:
                        current = new_set
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try swapping: remove one, add another
            if not improved:
                for i in range(len(current)):
                    without_i = [j for j in current if j != current[i]]
                    candidates = [j for j in range(N) if j not in current and all(compat[j][k] for k in without_i)]
                    for add_idx in candidates:
                        new_score = cur_score - Fraction(1, nums[current[i]]) + Fraction(1, nums[add_idx])
                        if new_score > cur_score:
                            current = without_i + [add_idx]
                            cur_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        return current, cur_score
    
    best, best_score = local_search(best)
    
    # Try to add more elements
    while time.time() < deadline - 5:
        added = False
        for i in range(N):
            if i not in best and all(compat[i][j] for j in best):
                best.append(i)
                best_score += Fraction(1, nums[i])
                added = True
                break
        if not added:
            break
    
    # Convert back to values
    result = [nums[i] for i in best]
    return result
# EVOLVE-BLOCK-END

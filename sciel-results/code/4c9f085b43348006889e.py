"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def _get_divisors(n):
    """Return all positive divisors of n in descending order."""
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs, reverse=True)

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50.0  # safety margin before 60s limit
    
    divisors = _get_divisors(n)
    # Filter divisors <= target (larger ones can't be used)
    divisors = [d for d in divisors if d <= target]
    
    # The current seed uses greedy from largest to smallest.
    # Let's see what the greedy solution looks like and its size.
    # Greedy: pick largest divisor <= remainder, repeat.
    def greedy_solve():
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        if rem == 0:
            return chosen
        return None
    
    greedy_result = greedy_solve()
    if greedy_result is not None:
        best_size = len(greedy_result)
        best_solution = greedy_result
    else:
        best_size = 256  # max allowed
        best_solution = []
    
    # Try to find a smaller solution using BFS/DFS with pruning.
    # Since the number of divisors is ~60, and we want min terms,
    # we can try iterative deepening: try size 1, 2, 3, ...
    # But 60 choose k grows fast. However, the sum is fixed.
    # A better approach: since we want to minimize count, and divisors are distinct,
    # we can use a meet-in-the-middle or just smart backtracking.
    
    # Actually, let's first check if the greedy solution is already optimal.
    # The greedy solution for 5040 divisors summing to 4979:
    # 5040 divisors in descending: 5040, 2520, 1680, 1260, 1008, 840, 720, 560, 504, 420, 360, 336, 280, 252, 240, 210, 180, 168, 140, 126, 120, 105, 90, 84, 72, 70, 63, 56, 45, 42, 36, 35, 30, 28, 24, 21, 20, 18, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    
    # Greedy: 4979
    # 2520 (rem=2459), 1680 (rem=779), 720 (rem=59), 56 (rem=3), 3 (rem=0)
    # So greedy gives 5 divisors: [2520, 1680, 720, 56, 3]
    
    # Can we do it in 4?
    # We need 4 distinct divisors summing to 4979.
    # Let's do a targeted search for k=1,2,3,4 before the greedy size.
    
    def find_min_terms(k_max):
        """Find minimum number of terms, checking up to k_max."""
        for k in range(1, k_max + 1):
            if time.time() > deadline - 5.0:
                break
            # Use backtracking with pruning
            found = None
            def backtrack(start, remaining, current):
                nonlocal found
                if found is not None:
                    return
                if time.time() > deadline - 5.0:
                    return
                if len(current) == k:
                    if remaining == 0:
                        found = current[:]
                    return
                if remaining <= 0:
                    return
                # Pruning: max possible sum with (k - len(current)) more divisors
                # from divisors[start:]
                max_possible = sum(divisors[start:start + (k - len(current))]) if start + (k - len(current)) <= len(divisors) else sum(divisors[start:])
                if max_possible < remaining:
                    return
                # Min possible sum
                if start >= len(divisors):
                    return
                for i in range(start, len(divisors)):
                    if divisors[i] > remaining:
                        continue
                    current.append(divisors[i])
                    backtrack(i + 1, remaining - divisors[i], current)
                    current.pop()
                    if found is not None:
                        return
            backtrack(0, target, [])
            if found is not None:
                return found
        return None
    
    # Try to find a solution with fewer terms than greedy
    if greedy_result is not None:
        search_max = len(greedy_result) - 1
    else:
        search_max = 10
    
    better_solution = find_min_terms(search_max)
    
    if better_solution is not None:
        return better_solution
    elif greedy_result is not None:
        return greedy_result
    else:
        # Fallback: should not happen, but return empty or raise
        raise ValueError("No solution found")
# EVOLVE-BLOCK-END

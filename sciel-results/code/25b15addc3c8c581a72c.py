"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # We want to maximize the size of a subset S of {1,...,24} such that
    # for no x in S does 1/x equal the sum of 1/y for some subset of S\{x}.
    #
    # Strategy:
    # 1. Start with all 24 elements.
    # 2. Iteratively remove the "most problematic" element (one that participates
    #    in the most reciprocal-subsum violations) until no violations remain.
    # 3. Then try to add back removed elements in a greedy manner if possible.
    # 4. Also try local search: swap elements to increase size if possible.

    def is_valid(S):
        """Check if set S is reciprocal-subsum-free."""
        S_set = set(S)
        for x in S_set:
            target = Fraction(1, x)
            others = [y for y in S_set if y != x]
            # Check all non-empty subsets of others
            # For efficiency, use meet-in-the-middle for larger sets
            if len(others) <= 20:
                # Direct enumeration for small sets
                n = len(others)
                for mask in range(1, 1 << n):
                    s = Fraction(0, 1)
                    for i in range(n):
                        if mask & (1 << i):
                            s += Fraction(1, others[i])
                    if s == target:
                        return False
            else:
                # Meet-in-the-middle
                half = len(others) // 2
                left = others[:half]
                right = others[half:]
                
                left_sums = {}
                n_left = len(left)
                for mask in range(1, 1 << n_left):
                    s = Fraction(0, 1)
                    for i in range(n_left):
                        if mask & (1 << i):
                            s += Fraction(1, left[i])
                    if s not in left_sums:
                        left_sums[s] = True
                
                # Check if target is in left_sums
                if target in left_sums:
                    return False
                
                # Check combinations of left and right
                n_right = len(right)
                for mask in range(1, 1 << n_right):
                    s_right = Fraction(0, 1)
                    for i in range(n_right):
                        if mask & (1 << i):
                            s_right += Fraction(1, right[i])
                    needed = target - s_right
                    if needed in left_sums:
                        return False
                    # Also check if s_right alone equals target
                    if s_right == target:
                        return False
        return True

    def count_violations(S):
        """Count how many elements in S are involved in violations."""
        S_set = set(S)
        violations = set()
        for x in S_set:
            target = Fraction(1, x)
            others = [y for y in S_set if y != x]
            if len(others) <= 20:
                n = len(others)
                found = False
                for mask in range(1, 1 << n):
                    s = Fraction(0, 1)
                    for i in range(n):
                        if mask & (1 << i):
                            s += Fraction(1, others[i])
                    if s == target:
                        violations.add(x)
                        found = True
                        break
            else:
                half = len(others) // 2
                left = others[:half]
                right = others[half:]
                
                left_sums = set()
                n_left = len(left)
                for mask in range(1, 1 << n_left):
                    s = Fraction(0, 1)
                    for i in range(n_left):
                        if mask & (1 << i):
                            s += Fraction(1, left[i])
                    left_sums.add(s)
                
                found = False
                if target in left_sums:
                    violations.add(x)
                    found = True
                else:
                    n_right = len(right)
                    for mask in range(1, 1 << n_right):
                        s_right = Fraction(0, 1)
                        for i in range(n_right):
                            if mask & (1 << i):
                                s_right += Fraction(1, right[i])
                        needed = target - s_right
                        if needed in left_sums:
                            violations.add(x)
                            found = True
                            break
                        if s_right == target:
                            violations.add(x)
                            found = True
                            break
        return violations

    # Start with all 24 elements
    current = list(range(1, 25))
    
    # Iteratively remove the element involved in the most violations
    while not is_valid(current):
        if time.time() > deadline:
            break
        violations = count_violations(current)
        if not violations:
            break
        # Remove the element that appears in the most violation contexts
        # For simplicity, remove one violating element at a time
        # Pick the smallest violating element to try to keep larger elements
        to_remove = min(violations)
        current.remove(to_remove)
    
    best = list(current)
    
    # Try to add back removed elements
    removed = [x for x in range(1, 25) if x not in best]
    for x in removed:
        if time.time() > deadline:
            break
        candidate = best + [x]
        if is_valid(candidate):
            best = candidate
    
    # Local search: try swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(best):
            if time.time() > deadline:
                break
            for y in range(1, 25):
                if y in best:
                    continue
                candidate = [z if z != x else y for z in best]
                if is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return sorted(best)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a, b, c) with a < b < c, distinct, 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  =>  bc = a(b+c)  =>  (b-a)(c-a) = a^2
    # For a in 1..180, factor a^2, find b = a + d, c = a + a^2/d for divisors d of a^2 with d > 0
    # We need b, c in [1, 180], b != c, a != b, a != c
    forbidden = set()  # set of frozensets of 3 elements
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d of a2
        divisors = []
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        for d in divisors:
            b = a + d
            c_val = a2 // d
            c = a + c_val
            if b <= N and c <= N and b >= 1 and c >= 1:
                if b != c and a != b and a != c:
                    triple = frozenset((a, b, c))
                    forbidden.add(triple)
    
    # Convert to list for faster iteration
    forbidden_list = list(forbidden)
    
    # For fast checking: for each pair (b,c), check if a = bc/(b+c) is integer and in set
    # Better: for each element x, precompute which other elements form forbidden triples with x
    # Actually, let's use a different approach: for each pair (i,j), compute if there exists k such that {i,j,k} is forbidden
    
    # Precompute: for each pair (i,j) with i<j, find k such that {i,j,k} is a forbidden triple
    # 1/a = 1/b + 1/c means if we know two of {a,b,c}, the third is determined
    # If a is the "small" one: b*c = a*(b+c)
    # If b is known: a = b*c/(b+c) (if integer)
    # If c is known: a = b*c/(b+c) (if integer)
    
    # Let's build a conflict graph: for each pair (i,j), list of k values that form a forbidden triple
    pair_conflicts = {}  # (i,j) -> set of k values
    for triple in forbidden_list:
        elems = list(triple)
        for idx in range(3):
            k = elems[idx]
            i, j = elems[(idx+1)%3], elems[(idx+2)%3]
            if i > j:
                i, j = j, i
            if (i, j) not in pair_conflicts:
                pair_conflicts[(i, j)] = set()
            pair_conflicts[(i, j)].add(k)
    
    # Also build: for each element, which pairs conflict with it
    # Actually, let's use a simpler approach for local search
    
    # Start with the seed solution: {91, 92, ..., 180}
    current = set(range(91, 181))
    best = set(current)
    best_size = len(best)
    
    def is_valid(s):
        """Check if set s has no forbidden triple."""
        # For each forbidden triple, check if all 3 elements are in s
        for triple in forbidden_list:
            if triple.issubset(s):
                return False
        return True
    
    def count_violations(s):
        """Count number of forbidden triples fully contained in s."""
        count = 0
        for triple in forbidden_list:
            if triple.issubset(s):
                count += 1
        return count
    
    # Verify initial solution is valid
    assert is_valid(current), "Initial solution should be valid"
    
    # Local search with repair
    random.seed(42)
    
    # Strategy: try to add elements from 1..90 that aren't in the set
    # For each candidate, check if adding it creates violations
    # If it does, try to remove elements to fix
    
    def try_add(s, x):
        """Try to add x to s, return (new_set, success)"""
        if x in s:
            return s, True
        # Check what violations would be created
        new_s = s | {x}
        if is_valid(new_s):
            return new_s, True
        return s, False
    
    # More sophisticated: local search
    # We'll try to add small numbers and remove blocking large numbers
    
    # For each element x not in current, find which elements in current block it
    # (i.e., which elements form forbidden triples with x and another element in current)
    
    def blocking_elements(s, x):
        """Find elements in s that, together with x, form a forbidden triple with another element in s."""
        blockers = set()
        for triple in forbidden_list:
            if x in triple:
                other = [y for y in triple if y != x]
                # Check if both others are in s, or one is in s and the other is x
                # Actually, we need: if we add x, and both other elements are in s, we get a violation
                # Also, if one other is in s and the third (not x, not that one) is... wait
                # A violation is a triple fully in s ∪ {x}
                # So for triple containing x: if both other elements are in s, adding x creates violation
                # For triple not containing x: no effect
                # But also: for triples containing x, if one other is in s, we don't have a violation yet
                # We need BOTH others in s
                if all(y in s for y in other):
                    blockers.update(other)
        return blockers
    
    # Local search: iterate, try to add each missing element, if blocked, try swapping
    iterations = 0
    max_iterations = 500000
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Try to add elements not in current
        missing = [x for x in range(1, N + 1) if x not in current]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() - start_time > DEADLINE:
                break
            if x in current:
                continue
            
            # Check if x can be added without conflict
            if is_valid(current | {x}):
                current = current | {x}
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
                improved = True
            else:
                # Try to find a swap: remove some element y from current, add x
                # y should be one that blocks x
                blockers = blocking_elements(current, x)
                # Also consider other elements that might be "exchangeable"
                # Try each blocker
                for y in list(blockers):
                    if time.time() - start_time > DEADLINE:
                        break
                    new_s = (current - {y}) | {x}
                    if len(new_s) == len(current):  # swap
                        if is_valid(new_s):
                            current = new_s
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(best)
                            improved = True
                            break
                # If no blocker swap worked, try random swaps
                if not improved and time.time() - start_time < DEADLINE:
                    # Try a few random swaps
                    for _ in range(5):
                        if time.time() - start_time > DEADLINE:
                            break
                        y = random.choice(list(current))
                        if y == x:
                            continue
                        new_s = (current - {y}) | {x}
                        if is_valid(new_s):
                            current = new_s
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(best)
                            improved = True
                            break
        
        # If no improvement in this round, try some random perturbations
        if not improved:
            # Try random 1-swaps
            for _ in range(100):
                if time.time() - start_time > DEADLINE:
                    break
                x = random.choice([x for x in range(1, N+1) if x not in current]) if any(x not in current for x in range(1,N+1)) else None
                if x is None:
                    break
                y = random.choice(list(current))
                new_s = (current - {y}) | {x}
                if is_valid(new_s):
                    # Accept with some probability (simulated annealing-like)
                    if len(new_s) >= len(current):
                        current = new_s
                    else:
                        # Only accept if it might lead to improvement
                        pass  # Don't accept size-decreasing moves easily
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Final: try to add any remaining elements
    for x in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        if x not in best and is_valid(best | {x}):
            best = best | {x}
    
    return list(best)
# EVOLVE-BLOCK-END

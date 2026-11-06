"""Rewritten program: prime-powers base + greedy fill with precomputed conflict hypergraph."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill (90s hard limit)
    
    N = 128
    universe = list(range(1, N + 1))
    
    # --- Precompute all pairwise LCMs ---
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_table[(a, b)] = _lcm(a, b)
    
    # --- Precompute conflict hypergraph ---
    # conflict[x] = set of (a, b) pairs such that {a, b, x} forms a bad triple
    # i.e., lcm(a,b) == lcm(a,x) == lcm(b,x)
    conflict = {x: set() for x in range(1, N + 1)}
    # For each pair (a,b), find all x such that lcm(a,b)==lcm(a,x)==lcm(b,x)
    # Precompute: for each x, which pairs (a,b) are incompatible with x
    for x in range(1, N + 1):
        for a in range(1, x):
            lax = _lcm(a, x)
            for b in range(a + 1, x):
                if _lcm(a, b) == lax and _lcm(b, x) == lax:
                    conflict[x].add((a, b))
    
    # --- Identify prime powers ---
    def is_prime_power(n):
        if n <= 1:
            return False
        # Check if n = p^k for some prime p
        for p in range(2, n + 1):
            if p * p > n:
                break
            if n % p == 0:
                # n has factor p; check if all factors are p
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                break
        # n itself is prime
        if n > 1:
            # check primality
            is_p = True
            for d in range(2, int(n**0.5) + 1):
                if n % d == 0:
                    is_p = False
                    break
            if is_p:
                return True
        return False
    
    prime_powers = [x for x in range(2, N + 1) if is_prime_power(x)]
    
    # --- Greedy: start with all prime powers, then add others in best order ---
    selected = set(prime_powers)
    
    remaining = [x for x in range(1, N + 1) if x not in selected]
    
    # Sort remaining by number of conflicts with selected set (fewer conflicts first)
    def conflict_count_with_selected(x):
        # Count how many pairs in selected would form a bad triple with x
        sel_list = list(selected)
        count = 0
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                if (min(a,b), max(a,b)) in conflict[x]:
                    count += 1
        return count
    
    # Iteratively add the element with fewest conflicts
    best_result = list(selected)
    
    while remaining and time.time() < deadline:
        # Find candidate with minimum conflicts
        best_x = None
        best_conf = float('inf')
        for x in remaining:
            c = conflict_count_with_selected(x)
            if c < best_conf:
                best_conf = c
                best_x = x
            elif c == best_conf and x < best_x:
                best_x = x
        
        if best_conf == 0:
            # Can add without removing anything
            selected.add(best_x)
            remaining.remove(best_x)
            if len(selected) > len(best_result):
                best_result = list(selected)
        else:
            # Try to add by removing conflicting elements
            # Find all pairs in selected that conflict with best_x
            conflicting_pairs = []
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a, b = sel_list[i], sel_list[j]
                    if (min(a,b), max(a,b)) in conflict[best_x]:
                        conflicting_pairs.append((a, b))
            
            if len(conflicting_pairs) <= 1:
                # Remove at most one element
                if len(conflicting_pairs) == 0:
                    selected.add(best_x)
                else:
                    a, b = conflicting_pairs[0]
                    # Remove the one that has more conflicts overall (less valuable)
                    # Simple heuristic: remove the larger one (composite likely less "core")
                    to_remove = max(a, b)
                    selected.discard(to_remove)
                    selected.add(best_x)
                remaining.remove(best_x)
                if len(selected) > len(best_result):
                    best_result = list(selected)
            else:
                # Too many conflicts; try next candidate
                remaining.remove(best_x)
    
    # If we ended up with an empty set (shouldn't happen), fall back
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END

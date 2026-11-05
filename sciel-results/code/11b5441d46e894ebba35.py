"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache
from collections import defaultdict
import random

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    """
    N = 180
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples {a, b, c} with a, b, c distinct and 1/a = 1/b + 1/c
    forbidden = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted((a, b, c)))
                    forbidden.add(triple)
    
    # Build conflict structure
    # conflicts_with[x] = list of pairs (u, v) such that {x, u, v} is a forbidden triple
    conflicts_with = defaultdict(list)
    for (a, b, c) in forbidden:
        conflicts_with[a].append((b, c))
        conflicts_with[b].append((a, c))
        conflicts_with[c].append((a, b))

    def is_valid(S):
        """Check if S contains any forbidden triple."""
        for x in S:
            for (u, v) in conflicts_with[x]:
                if u in S and v in S:
                    return False
        return True

    def greedy(order):
        S_local = set()
        for x in order:
            can_add = True
            for (u, v) in conflicts_with[x]:
                if u in S_local and v in S_local:
                    can_add = False
                    break
            if can_add:
                S_local.add(x)
        return S_local

    # Initialize with multiple greedy strategies
    best_set = set()
    best_size = 0
    
    # Strategy 1: Base set [91..180] + greedy add [1..90]
    S_base = set(range(91, N + 1))
    for x in range(90, 0, -1):
        can_add = True
        for (u, v) in conflicts_with[x]:
            if u in S_base and v in S_base:
                can_add = False
                break
        if can_add:
            S_base.add(x)
    if len(S_base) > best_size:
        best_set = S_base
        best_size = len(S_base)
        
    # Strategy 2: Random Greedy
    random.seed(42)
    for _ in range(50):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
    # Local Search: Hill Climbing with Swaps
    # We try to swap an element in S with an element not in S to maintain/increase size.
    # Since we want to maximize size, and swaps are 1-for-1, we rely on the swap
    # opening up possibilities for further additions or just finding a valid state
    # with the same size but "better" structure. However, to get a jump, we should
    # look for 1-for-2 swaps if possible, but that's rare.
    # Instead, we do a "repair" local search:
    # 1. Try to add any element not in S.
    # 2. If blocked, try to remove a conflicting element from S to make room.
    
    current_set = set(best_set)
    current_size = best_size
    
    # Optimization: Precompute for each x not in S, the blocking pairs in S
    # This is dynamic, so we recompute or check on the fly.
    
    # To speed up, we use a time limit
    import time
    start_time = time.time()
    time_limit = 50 # seconds
    
    # Local Search Loop
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add elements
        for x in range(1, N + 1):
            if x in current_set:
                continue
            # Check if x can be added
            can_add = True
            for (u, v) in conflicts_with[x]:
                if u in current_set and v in current_set:
                    can_add = False
                    break
            if can_add:
                current_set.add(x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
                # Restart adding from 1? Or continue?
                # Continue is fine, but might miss some.
                # Break to restart the loop with the new state?
                # Let's break to ensure we re-evaluate all additions with the new set
                break
        
        # 2. Try swaps if no additions were made
        if not improved:
            # Try to swap x in current_set with y not in current_set
            # We prioritize swapping "bad" elements (high conflict) with "good" ones.
            # For simplicity, iterate through all pairs.
            # This is O(N^2 * Conflicts). N=180, N^2=32400. Conflicts small.
            # Might be slow in Python. Limit iterations.
            
            # Heuristic: Only try to swap if the element to be removed is "large" 
            # (tail) and the one to be added is "small" (head), or vice versa.
            
            # To avoid O(N^2) full scan every iteration, we pick a random element to remove
            # and try to find a replacement.
            
            # Let's try a more directed swap:
            # For each y not in S, find blocking pairs.
            # If we remove one element from a blocking pair, does it unblock y?
            # If yes, and the resulting set is valid, do the swap.
            
            # We'll do a limited number of random swap attempts per pass.
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random y not in S
                not_in_S = [i for i in range(1, N + 1) if i not in current_set]
                if not not_in_S:
                    break
                y = random.choice(not_in_S)
                
                # Find blocking pairs for y in current_set
                blockers = []
                for (u, v) in conflicts_with[y]:
                    if u in current_set and v in current_set:
                        blockers.append((u, v))
                
                if not blockers:
                    # Should have been added in step 1, but maybe state changed?
                    current_set.add(y)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                    break
                
                # Try to remove one element from a random blocking pair
                u, v = random.choice(blockers)
                
                # Try swapping u with y
                # Remove u, add y
                new_set = set(current_set)
                new_set.remove(u)
                new_set.add(y)
                
                # Check validity of new_set
                # Only need to check triples involving y or u
                # Actually, removing u can only break triples, adding y can only create them.
                # So we only need to check if y creates any triple in new_set.
                valid = True
                for (a, b) in conflicts_with[y]:
                    if a in new_set and b in new_set:
                        valid = False
                        break
                
                if valid:
                    # Also check if removing u broke anything? No, removal is safe.
                    # But we need to ensure the rest of the set is still valid.
                    # Since current_set was valid, and we only removed u and added y,
                    # any new conflict must involve y.
                    # So checking y's conflicts is sufficient.
                    
                    current_set = new_set
                    # Size remains same
                    improved = True
                    # We don't update best_size here as size is same, 
                    # but this state might allow future additions.
                    break

    return sorted(best_set)

# EVOLVE-BLOCK-END

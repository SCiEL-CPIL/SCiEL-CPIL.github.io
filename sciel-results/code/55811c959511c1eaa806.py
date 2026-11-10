import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit

    best_set = []
    best_size = 0

    # Precompute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = reduce(lcm, range(1, 23))
    weights = {a: M // a for a in range(1, 23)}
    
    # Helper to check if adding weight w to current_sums is valid
    # current_sums is a set of ints
    def try_add_int(current_sums, w):
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
        if len(new_sums) == 2 * len(current_sums):
            return True, new_sums
        return False, None

    # Build sums for a given set
    def build_sums(s):
        sums = {0}
        for x in s:
            w_x = weights[x]
            new_sums = set(sums)
            for val in sums:
                new_sums.add(val + w_x)
            sums = new_sums
        return sums

    # 1. Multiple deterministic greedy orderings
    orderings = [
        list(range(1, 23)),           # ascending
        list(range(22, 0, -1)),       # descending
        list(range(1, 23, 2)) + list(range(2, 23, 2)),  # odds then evens
        list(range(2, 23, 2)) + list(range(1, 23, 2)),  # evens then odds
    ]
    
    for ordering in orderings:
        current_set = []
        current_sums = {0}
        for a in ordering:
            w = weights[a]
            ok, new_sums = try_add_int(current_sums, w)
            if ok:
                current_set.append(a)
                current_sums = new_sums
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    # 2. Backtracking search to find larger sets
    def backtrack(remaining, current_set, current_sums, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return False
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if not remaining:
            return len(current_set) > best_size - 1  # prune if can't beat best
        
        # Try to add each remaining element
        for i, a in enumerate(remaining):
            if time.time() > deadline:
                return False
            
            w = weights[a]
            ok, new_sums = try_add_int(current_sums, w)
            if ok:
                remaining_copy = remaining[:i] + remaining[i+1:]
                if backtrack(remaining_copy, current_set + [a], new_sums, depth + 1):
                    return True
        
        return False

    # Run backtracking with a timeout
    remaining_all = list(range(1, 23))
    backtrack(remaining_all, [], {0}, 0)

    # 3. Randomized Restarts with Integer Arithmetic
    while time.time() < deadline - 5:
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {0}
        
        for a in candidates:
            if time.time() > deadline - 5:
                break
            
            w = weights[a]
            ok, new_sums = try_add_int(current_sums, w)
            if ok:
                current_set.append(a)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]

    # 4. Final polish: try to add any element to best_set
    if time.time() < deadline - 2:
        best_sums = build_sums(best_set)
        for y in range(1, 23):
            if y in best_set:
                continue
            w_y = weights[y]
            ok, new_sums = try_add_int(best_sums, w_y)
            if ok:
                best_set.append(y)
                best_sums = new_sums
                best_size = len(best_set)

    # 5. Swap-based local search: try replacing one element with another
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_sums = build_sums(best_set)
            best_set_set = set(best_set)
            
            for i, x in enumerate(best_set):
                if time.time() > deadline - 2:
                    break
                
                # Remove x
                temp_set = best_set[:i] + best_set[i+1:]
                temp_sums = build_sums(temp_set)
                
                # Try adding each y not in best_set
                for y in range(1, 23):
                    if y in best_set_set:
                        continue
                    w_y = weights[y]
                    ok, new_sums = try_add_int(temp_sums, w_y)
                    if ok:
                        # Check if this improves (same size, but different composition might help future additions)
                        # For now, accept any swap that maintains validity
                        new_set = temp_set + [y]
                        new_set_set = set(new_set)
                        
                        # Try to add more elements to the swapped set
                        final_set = new_set[:]
                        final_sums = new_sums
                        for z in range(1, 23):
                            if z in new_set_set:
                                continue
                            w_z = weights[z]
                            ok2, final_sums2 = try_add_int(final_sums, w_z)
                            if ok2:
                                final_set.append(z)
                                final_sums = final_sums2
                                new_set_set.add(z)
                        
                        if len(final_set) > best_size:
                            best_size = len(final_set)
                            best_set = final_set
                            improved = True
                            break
                        elif len(final_set) == best_size:
                            # Accept swap if it leads to a different set (might help future improvements)
                            if set(final_set) != set(best_set):
                                best_set = final_set
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break

    if best_set:
        return best_set
    else:
        return [1]

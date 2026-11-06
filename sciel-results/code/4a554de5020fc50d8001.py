"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_largest(S, deadline):
        """Greedy: add elements in decreasing order of absolute value."""
        result = []
        result_set = set()
        for x in sorted(S, key=lambda v: -abs(v)):
            if time.time() > deadline:
                break
            # Check if adding x violates sum-free property
            valid = True
            for a in result:
                if a + x in result_set or x + a in result_set:
                    valid = False
                    break
                if 2 * x in result_set:
                    valid = False
                    break
            if valid:
                result.append(x)
                result_set.add(x)
        return result

    def greedy_with_backtrack(S, deadline):
        """Greedy with local search improvements."""
        # Start with greedy largest
        result = greedy_largest(S, deadline)
        result_set = set(result)
        
        # Try to add elements that were rejected
        for x in sorted(S, key=lambda v: -abs(v)):
            if time.time() > deadline:
                break
            if x in result_set:
                continue
            # Check if x can be added
            valid = True
            for a in result:
                if a + x in result_set:
                    valid = False
                    break
                if 2 * x in result_set:
                    valid = False
                    break
            if valid:
                result.append(x)
                result_set.add(x)
        
        # Local search: try swapping elements to improve size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding two
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                removed = result[i]
                temp_result = result[:i] + result[i+1:]
                temp_set = set(temp_result)
                
                # Try to add two elements that conflict with removed
                candidates = []
                for x in S:
                    if x in temp_set:
                        continue
                    if x == removed:
                        continue
                    # Check if x conflicts with temp_set
                    valid = True
                    for a in temp_set:
                        if a + x in temp_set:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                
                # Try pairs from candidates
                added_any = False
                for ci in range(len(candidates)):
                    if time.time() > deadline:
                        return result
                    c1 = candidates[ci]
                    for cj in range(ci+1, len(candidates)):
                        c2 = candidates[cj]
                        if c1 == c2:
                            continue
                        # Check if c1 and c2 can both be added
                        test_set = set(temp_result + [c1, c2])
                        valid = True
                        for a in test_set:
                            for b in test_set:
                                if a + b in test_set and (a + b) != a and (a + b) != b:
                                    # This is a violation
                                    if a + b in test_set:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            result = temp_result + [c1, c2]
                            result_set = set(result)
                            improved = True
                            added_any = True
                            break
                    if added_any or time.time() > deadline:
                        break
                if added_any or time.time() > deadline:
                    break
        return result

    # Process each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Return best effort
            results.append(greedy_largest(instance(size, seed), deadline))
            continue
        
        S = instance(size, seed)
        
        # Try multiple strategies and pick best
        candidates = []
        
        # Strategy 1: Greedy largest
        c1 = greedy_largest(S, deadline)
        candidates.append(c1)
        
        # Strategy 2: Odd numbers (parity filter)
        c2 = [x for x in S if x % 2]
        candidates.append(c2)
        
        # Strategy 3: Greedy with backtracking
        if time.time() < deadline - 5:
            c3 = greedy_with_backtrack(S, deadline)
            candidates.append(c3)
        
        # Pick the largest sum-free subset
        best = max(candidates, key=lambda s: len(s))
        
        # Verify it's actually sum-free
        if not is_sum_free(best):
            # Fallback to parity filter which is guaranteed sum-free
            best = [x for x in S if x % 2]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END

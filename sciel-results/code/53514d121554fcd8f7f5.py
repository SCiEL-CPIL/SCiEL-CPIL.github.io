"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys

    deadline = time.time() + 110  # 10s safety margin before 120s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_add(S, candidates):
        """Try to add elements from candidates to S while maintaining sum-free property."""
        S_set = set(S)
        result = list(S)
        for x in candidates:
            if x in S_set:
                continue
            # Check if adding x violates sum-free:
            # Need: no a in S_set such that a + x in S_set (i.e., a+x exists)
            # And: no a in S_set such that x + a in S_set (same as above)
            # And: 2*x not in S_set (since a=b=x is allowed)
            # Actually: we need to check if there exist a,b in S_set ∪ {x} such that a+b=x or a+b in S_set
            # When adding x, new violations are:
            # 1. a + x = c for some a,c in S_set => a = c - x, so c - x in S_set and c in S_set
            # 2. x + x = c for some c in S_set => 2*x in S_set
            # 3. a + b = x for some a,b in S_set => a + b = x with a,b in S_set
            
            ok = True
            # Check condition 2: 2x in S_set
            if 2 * x in S_set:
                ok = False
            else:
                # Check condition 1: exists a,c in S_set with a + x = c, i.e., c - x = a
                # Equivalently: exists c in S_set such that c - x in S_set
                # Check condition 3: exists a,b in S_set with a + b = x
                if ok:
                    for c in S_set:
                        if c - x in S_set:
                            ok = False
                            break
                if ok:
                    for a in S_set:
                        if x - a in S_set and a != x - a:  # a + (x-a) = x, a != x-a to avoid double count but set handles it
                            # Actually a + b = x with a,b in S_set
                            # We already have a, check if x-a is in S_set
                            # But we need a != x-a? No, a=b is allowed, so 2a=x is fine as long as a in S_set
                            # Wait, if a = x/2 and x is even, then a+a=x, so if a in S_set, violation
                            # Let me just check all pairs
                            pass
                    # Simpler: check all pairs a,b in S_set
                    for a in S_set:
                        b = x - a
                        if b in S_set:
                            ok = False
                            break
            
            if ok:
                result.append(x)
                S_set.add(x)
        return result

    def local_search(S, candidates, max_iterations=500):
        """Local search: try to add elements, then remove and re-add to improve."""
        S_set = set(S)
        result = list(S)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for x in candidates:
                if x in S_set:
                    continue
                # Try adding x
                temp_set = S_set | {x}
                # Check sum-free for temp_set
                valid = True
                for a in temp_set:
                    for b in temp_set:
                        if a + b in temp_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    result.append(x)
                    S_set = temp_set
                    improved = True
                    break
            if not improved:
                break
        return result

    def solve_one(S, seed):
        """Find a large sum-free subset of S."""
        n = len(S)
        best = []
        
        # Strategy 1: Take all odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(best):
            best = odds
        
        # Strategy 2: Greedy construction
        # Sort by absolute value, try to add each
        S_set = set()
        greedy = []
        for x in sorted(S, key=abs):
            # Check if adding x keeps sum-free
            valid = True
            for a in greedy:
                if a + x in S_set:
                    valid = False
                    break
                if x - a in S_set:
                    valid = False
                    break
            if 2 * x in S_set:
                valid = False
            if valid:
                greedy.append(x)
                S_set.add(x)
        if len(greedy) > len(best):
            best = greedy
        
        # Strategy 3: Local search starting from best
        if time.time() < deadline - 5:
            candidates = [x for x in S if x not in set(best)]
            improved = local_search(best, candidates, max_iterations=200)
            if len(improved) > len(best):
                best = improved
        
        # Strategy 4: Random restarts with local search
        rng = Random(seed)
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            # Start with random subset
            random_subset = []
            for x in S:
                if rng.random() < 0.5:
                    random_subset.append(x)
            # Local search
            candidates = [x for x in S if x not in set(random_subset)]
            improved = local_search(random_subset, candidates, max_iterations=100)
            if len(improved) > len(best):
                best = improved
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            # Fallback: just return odds
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            # Verify sum-free
            if not is_sum_free(result):
                # Fallback to odds
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
# EVOLVE-BLOCK-END

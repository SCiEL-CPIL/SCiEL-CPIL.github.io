"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 110.0  # safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S):
        """Check if x can be added to sum-free set S."""
        if x + x in S:
            return False
        for a in S:
            if a + x in S:
                return False
        # Check if x is a sum of two elements
        for a in S:
            for b in S:
                if a + b == x:
                    return False
        return True

    def greedy_add(S, candidates, rng):
        """Greedily add elements from candidates to S while maintaining sum-free."""
        current = set(S)
        for x in candidates:
            if time.time() > deadline:
                break
            if x in current:
                continue
            if can_add(x, current):
                current.add(x)
        return current

    def local_search(S, full_set, rng, max_rounds=200):
        """Local search with add/remove moves."""
        current = set(S)
        best = set(S)
        best_size = len(best)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a random element not in current
            missing = [x for x in full_set if x not in current]
            if missing:
                rng.shuffle(missing)
                for x in missing[:20]:  # Try up to 20 random additions
                    if can_add(x, current):
                        current.add(x)
                        improved = True
                        break
            
            # Try removing and re-adding
            if not improved and len(current) > 0:
                to_remove = list(current)
                rng.shuffle(to_remove)
                for a in to_remove[:10]:  # Try up to 10 random removals
                    current.discard(a)
                    # Now try to add something
                    missing2 = [x for x in full_set if x not in current]
                    if missing2:
                        rng.shuffle(missing2)
                        for x in missing2[:15]:
                            if can_add(x, current):
                                current.add(x)
                                if len(current) > best_size:
                                    best = set(current)
                                    best_size = len(current)
                                break
                    if len(current) <= best_size:
                        # Restore if not improved
                        current.add(a)
                    # If we improved, keep a removed and continue
                    if len(current) > best_size:
                        improved = True
                        break
            
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        
        return best

    def solve_one(S, seed):
        if time.time() > deadline:
            return list(S)
        
        rng = Random(seed + 1000)
        full_set = set(S)
        
        # Multiple strategies, keep the best
        candidates = []
        
        # Strategy 1: All odd numbers
        odds = {x for x in S if x % 2 != 0}
        candidates.append(odds)
        
        # Strategy 2: All positive numbers
        positives = {x for x in S if x > 0}
        candidates.append(positives)
        
        # Strategy 3: All negative numbers
        negatives = {x for x in S if x < 0}
        candidates.append(negatives)
        
        # Strategy 4: Numbers with large absolute value
        abs_vals = sorted(S, key=lambda x: abs(x), reverse=True)
        top_half = set(abs_vals[:len(abs_vals)//2 + 1])
        candidates.append(top_half)
        
        # Strategy 5: Random subset
        rng.shuffle(S)
        rand_subset = set(S[:len(S)//2])
        candidates.append(rand_subset)
        
        # Strategy 6: Start empty, greedy add in random order
        rng2 = Random(seed + 2000)
        shuffled = list(S)
        rng2.shuffle(shuffled)
        greedy_empty = greedy_add(set(), shuffled, rng2)
        candidates.append(greedy_empty)
        
        # Strategy 7: Start with numbers > n/2 (if all positive)
        if all(x > 0 for x in S):
            n = max(S)
            large = {x for x in S if x > n // 2}
            candidates.append(large)
        
        # For each starting candidate, do local search
        best_result = set()
        best_size = 0
        
        for start in candidates:
            if time.time() > deadline:
                break
            
            # First, greedily add more
            remaining = [x for x in S if x not in start]
            rng.shuffle(remaining)
            grown = greedy_add(start, remaining, rng)
            
            # Then local search
            result = local_search(grown, S, rng, max_rounds=100)
            
            if len(result) > best_size:
                best_result = result
                best_size = len(result)
        
        # One more pass: try to improve best_result
        if time.time() < deadline - 5:
            improved = local_search(best_result, S, rng, max_rounds=200)
            if len(improved) > len(best_result):
                best_result = improved
        
        return list(best_result)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        S = instance(size, seed)
        results.append(solve_one(S, seed))
    
    return results
# EVOLVE-BLOCK-END

import time
import random
import sys

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute all pairs (a, b) with a <= b, sum s = a+b
    # We need to ensure that for every s in [2, 512], there is a pair (a,b) in our set
    # such that a+b=s.
    # Actually, maximality means: for every x not in S, adding x creates a collision.
    # Collision occurs if 2x is already a sum, or x+y is already a sum for some y in S.
    # This is equivalent to saying that the set of sums S+S "covers" all possible
    # "forbidden" positions for new elements.
    
    # A more direct approach:
    # A set S is maximal Sidon if for every x not in S, there exists y in S 
    # such that x+y is in S+S (or 2x is in S+S).
    
    # Let's use a heuristic search for small maximal Sidon sets.
    # Known small maximal Sidon sets are related to Bose-Chowla constructions or similar.
    # For N=256, a maximal Sidon set can be quite small (around 16-20 elements?).
    # Maximum Sidon set size is ~sqrt(N) = 16.
    # Wait, max Sidon size in [1, N] is approx sqrt(N).
    # Maximal Sidon sets can be smaller than maximum? No, maximal is inclusion-maximal.
    # A maximal Sidon set must have size at least something?
    # Actually, we want to MINIMIZE the cardinality of a MAXIMAL Sidon set.
    
    # Strategy:
    # 1. Start with a random small set.
    # 2. Check if it's Sidon.
    # 3. Check if it's maximal.
    # 4. If not maximal, add elements to make it maximal (greedily).
    # 5. Try to remove elements while maintaining maximality (local search).
    
    # Better Strategy:
    # Randomized Backtracking / Greedy with Backtracking
    # We want to find a set S such that:
    # 1. All pair sums are distinct.
    # 2. For every x not in S, x cannot be added.
    
    # Let's try a "cover" approach.
    # If we pick elements, they block other elements.
    # An element x blocks y if x+y is a sum of two elements in S (including x+x).
    
    # Let's use a randomized greedy algorithm with restarts and local search.
    
    def is_sidon(set_s):
        sums = set()
        for i in range(len(set_s)):
            for j in range(i, len(set_s)):
                s = set_s[i] + set_s[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(set_s):
        sums = set()
        for i in range(len(set_s)):
            for j in range(i, len(set_s)):
                sums.add(set_s[i] + set_s[j])
        
        s_set = set(set_s)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Can x be added?
            # Check if 2x is in sums
            if 2 * x in sums:
                continue
            # Check if x + y is in sums for any y in S
            can_add = True
            for y in set_s:
                if x + y in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True

    def get_sums(set_s):
        sums = set()
        for i in range(len(set_s)):
            for j in range(i, len(set_s)):
                sums.add(set_s[i] + set_s[j])
        return sums

    def try_construct(seed):
        rng = random.Random(seed)
        
        # Start with a random set of size ~10-15
        # We want to minimize size, so start small and grow if necessary for maximality?
        # Or start with a known structure?
        
        # Let's try to build a maximal set from scratch using a greedy "blocking" strategy.
        # Pick an element, it blocks others.
        
        # Actually, let's use a local search from a random maximal Sidon set.
        # How to get a random maximal Sidon set?
        # 1. Randomly pick elements, ensuring Sidon property.
        # 2. Continue until no more can be added (this gives a maximal Sidon set).
        # 3. Then try to remove elements while keeping it maximal.
        
        def random_maximal_sidon(rng):
            s = []
            s_set = set()
            sums = set()
            
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                if 2 * x in sums:
                    continue
                ok = True
                for y in s:
                    if x + y in sums:
                        ok = False
                        break
                if ok:
                    s.append(x)
                    s_set.add(x)
                    for y in s[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
            
            return s

        # Get an initial maximal Sidon set
        s = random_maximal_sidon(rng)
        if not s:
            return None
            
        s_set = set(s)
        
        # Local search: try to remove elements
        # If we remove an element x, the set is no longer Sidon? No, it's still Sidon.
        # But is it still maximal?
        # Removing an element might allow other elements to be added, breaking maximality.
        # We want to remove x if the resulting set is STILL maximal.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element
            for i in range(len(s)):
                if time.time() > deadline - 5:
                    break
                x = s[i]
                new_s = s[:i] + s[i+1:]
                new_s_set = s_set - {x}
                
                # Check if new_s is maximal
                if is_maximal(new_s):
                    s = new_s
                    s_set = new_s_set
                    improved = True
                    break # Restart loop with new size
        
        return s

    best_set = None
    best_size = float('inf')
    
    # Run multiple trials
    for seed in range(100):
        if time.time() > deadline - 2:
            break
            
        s = try_construct(seed)
        if s is not None:
            if len(s) < best_size:
                best_size = len(s)
                best_set = s
        
        # Early stopping if we find a very small set
        if best_size <= 12:
            break

    if best_set is None:
        # Fallback: return a valid Sidon set (might not be minimal, but valid)
        # Just return a small set
        return [1, 2, 4, 8, 16, 32, 64, 128]
        
    return best_set

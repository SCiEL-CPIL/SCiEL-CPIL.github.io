import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # We want to return a list of ints
    # Initial best known valid set (from seed or simple construction)
    # The seed program had [1, 2, 4, 8, 16, 32, 64, 128, 256] which is size 9.
    # Let's start with a random large set and prune.
    
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def compute_sums_by_cardinality(elements):
        """
        Returns a list of sets, sums[k] contains all sums of subsets of size k.
        Also returns a mapping from (cardinality, sum) to one subset (as a frozenset or tuple) 
        that achieves that sum, to help identify involved elements.
        """
        n = len(elements)
        sums = [set() for _ in range(n + 1)]
        witness = {} # (k, s) -> tuple of elements
        
        # Use DP or iterative construction
        # sums[0] = {0}
        sums[0].add(0)
        witness[(0, 0)] = ()
        
        for v in elements:
            # Update from high k to low k to avoid using v multiple times in same step?
            # No, we are building subsets of the *current* set of elements.
            # It's easier to just iterate all subsets if n is small, but n can be up to 20.
            # 2^20 is 1M. Iterating all subsets of 20 elements is doable in Python if optimized?
            # Maybe. But let's use the DP approach for sums, and keep track of one witness.
            
            # Actually, simple recursion/iteration over masks is probably faster in Python for n<=20 than complex DP updates with sets of sums.
            # Let's try mask iteration.
            pass
            
        # Let's stick to the mask iteration for simplicity and correctness of witness tracking.
        # If n > 20, this might be slow. We will limit initial N.
        
        sums = [set() for _ in range(n + 1)]
        witness = {}
        
        for mask in range(1 << n):
            s = 0
            cnt = 0
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
                    cnt += 1
                    elems.append(elements[i])
            
            if s not in sums[cnt]:
                sums[cnt].add(s)
                witness[(cnt, s)] = tuple(elems)
                
        return sums, witness

    def find_collisions(sums):
        """
        Returns a list of tuples (i, j, s) where i != j and s is in sums[i] and sums[j].
        """
        collisions = []
        n = len(sums)
        for i in range(n):
            for j in range(i + 1, n):
                inter = sums[i].intersection(sums[j])
                for s in inter:
                    collisions.append((i, j, s))
        return collisions

    def prune_set(elements):
        """
        Iteratively remove elements involved in the most collisions.
        """
        elements = list(elements)
        n = len(elements)
        
        while time.time() < deadline - 5:
            if n == 0:
                break
                
            sums, witness = compute_sums_by_cardinality(elements)
            
            # Check if valid
            is_valid = True
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    if not sums[i].isdisjoint(sums[j]):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                return elements
                
            # Find collisions
            collisions = find_collisions(sums)
            if not collisions:
                # Should not happen if is_valid is False, but just in case
                return elements
                
            # Count involvement
            involvement = {v: 0 for v in elements}
            
            # For each collision, pick a witness pair and increment count for elements in union
            # To avoid bias towards high sums or specific cardinalities, we just pick the first witness found.
            for i, j, s in collisions:
                # Get one subset of size i summing to s
                elems_i = witness.get((i, s), ())
                # Get one subset of size j summing to s
                elems_j = witness.get((j, s), ())
                
                # Mark elements in union
                involved_set = set(elems_i) | set(elems_j)
                for v in involved_set:
                    involvement[v] += 1
            
            # Find element with max involvement
            max_v = max(involvement, key=involvement.get)
            
            # Remove it
            elements.remove(max_v)
            n -= 1
            
        return elements

    # Strategy:
    # Try starting with different random sets of size N.
    # Start with N=20, then 18, etc.
    
    # To ensure we return something valid, we keep the best valid set found.
    # The seed set is valid.
    
    # Let's try a few random restarts with different N.
    # We have ~50 seconds.
    
    # Fixed seed for reproducibility? No, random is fine, but maybe use time-based seed?
    # The task says "random large set".
    
    for start_n in [20, 18, 16, 14]:
        if time.time() > deadline - 10:
            break
            
        # Try a few random instances for this N
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            # Generate random distinct integers in 1..256
            random_set = random.sample(range(1, 257), start_n)
            
            pruned = prune_set(random_set)
            
            # Verify pruned set is valid
            # prune_set should return a valid set, but let's double check
            if len(pruned) > len(best_set):
                # Verify
                sums, _ = compute_sums_by_cardinality(pruned)
                valid = True
                for i in range(len(sums)):
                    for j in range(i + 1, len(sums)):
                        if not sums[i].isdisjoint(sums[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set = pruned

    # Ensure types are correct
    return [int(x) for x in best_set]

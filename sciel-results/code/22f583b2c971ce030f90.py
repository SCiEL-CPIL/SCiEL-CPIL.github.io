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
    best_size = len(best_set)
    
    def compute_sums_by_cardinality(elements):
        """
        Returns a list of sets, sums[k] contains all sums of subsets of size k.
        """
        n = len(elements)
        sums = [set() for _ in range(n + 1)]
        sums[0].add(0)
        
        # Iterative DP for subset sums by cardinality
        # sums[k] holds sums of subsets of size k from the processed elements
        # We process elements one by one.
        # To avoid using an element multiple times, we update from high k to low k?
        # No, standard subset sum DP:
        # new_sums[k] = old_sums[k] U {s + v for s in old_sums[k-1]}
        
        current_sums = [set() for _ in range(n + 1)]
        current_sums[0].add(0)
        
        for v in elements:
            new_sums = [set(s) for s in current_sums]
            for k in range(1, n + 1):
                if current_sums[k-1]:
                    for s in current_sums[k-1]:
                        new_sums[k].add(s + v)
            current_sums = new_sums
        return current_sums

    def count_collisions(elements):
        """
        Returns the number of pairs of sums (s1, s2) with different cardinalities that are equal.
        Actually, we just need to know if there are ANY collisions.
        For optimization, we want a cost function.
        Cost = sum over all k != j of |sums[k] intersect sums[j]|.
        """
        sums = compute_sums_by_cardinality(elements)
        n = len(elements)
        cost = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] and sums[j]:
                    inter = sums[i].intersection(sums[j])
                    cost += len(inter)
        return cost

    def find_best_replacement(elements, index, deadline):
        """
        Try to replace elements[index] with a new value to reduce collisions.
        Returns (new_elements, new_cost) if improvement found, else (None, None).
        """
        current_cost = count_collisions(elements)
        if current_cost == 0:
            return elements, 0
        
        n = len(elements)
        current_val = elements[index]
        best_new_val = None
        best_new_cost = current_cost
        best_new_elements = None
        
        # Try a few random replacements
        # We can't try all 256, but we can try a sample.
        # Or we can be smart: try values that are "far" from current to change sums significantly.
        
        candidates = random.sample(range(1, 257), min(50, 256))
        candidates = [v for v in candidates if v not in elements and v != current_val]
        
        for v in candidates:
            new_elements = elements.copy()
            new_elements[index] = v
            new_cost = count_collisions(new_elements)
            if new_cost < best_new_cost:
                best_new_cost = new_cost
                best_new_val = v
                best_new_elements = new_elements
                if new_cost == 0:
                    break
                    
        if best_new_elements:
            return best_new_elements, best_new_cost
        return None, None

    def search_for_size(n, time_limit):
        """
        Try to find a valid set of size n.
        """
        if n <= best_size:
            return None
            
        for restart in range(20):
            if time.time() > time_limit:
                break
                
            # Initialize random set
            elements = random.sample(range(1, 257), n)
            
            # Greedy improvement
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                # Pick a random index to try to fix
                indices = list(range(n))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > time_limit:
                        break
                        
                    new_elements, new_cost = find_best_replacement(elements, idx, time_limit)
                    if new_elements:
                        elements = new_elements
                        improved = True
                        if new_cost == 0:
                            return elements
                            
            # If we didn't find a perfect set, check if current is better than best?
            # No, we only care about valid sets.
            
        return None

    # Try to find larger sets
    # We know size 9 is valid.
    # Try 10, 11, 12... up to maybe 20.
    # Time budget per size: ~2 seconds.
    
    for target_n in range(20, 9, -1):
        if time.time() > deadline - 5:
            break
            
        start_t = time.time()
        limit = start_t + 2.0 # 2 seconds per size
        
        result = search_for_size(target_n, limit)
        
        if result:
            # Verify
            sums = compute_sums_by_cardinality(result)
            valid = True
            for i in range(len(sums)):
                for j in range(i + 1, len(sums)):
                    if not sums[i].isdisjoint(sums[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_set = result
                best_size = target_n
                # If we found a valid set, we can stop searching for larger sizes?
                # No, we might find 20. But if we found 10, we should try 11?
                # The loop goes 20 down to 10.
                # If we found 15, we still try 14? No, we want MAX size.
                # So if we find a valid set of size N, we should update best and continue to N-1?
                # No, we want the LARGEST.
                # If we find size 15, we should break?
                # Wait, the loop is 20, 19, 18...
                # If we find 15, we know 16-20 failed.
                # We should check 14? No, 15 > 14.
                # So if we find ANY valid set in the loop, we should update best.
                # But we want the MAX.
                # So if we find size 15, we can return it immediately?
                # Yes, because we are iterating downwards. 15 is the largest we've found so far.
                # And we already tried 16-20 and failed.
                # So 15 is the max possible?
                # Not necessarily. Maybe 16 failed due to bad luck, but another 16 exists.
                # But we spent 2s on 16.
                # Given the time constraint, returning the first valid found in descending order is a good heuristic.
                return [int(x) for x in best_set]

    # Ensure types are correct
    return [int(x) for x in best_set]

import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    best_size = 0
    best_set = []
    
    # Max possible sum for 30 elements of max 256
    MAX_SUM = 256 * 30
    
    def is_valid_addition(current_sums_set, x):
        """
        Check if adding x to a set with subset sums current_sums_set is valid.
        current_sums_set is a set of integers.
        Returns True if valid, False otherwise.
        """
        # Generate new sums
        new_sums = set()
        for s in current_sums_set:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # We need to check if any new sum conflicts with any existing sum or other new sums.
        # Conflict: a | b for a != b.
        
        # 1. Check new sums against existing sums
        # For each ns in new_sums:
        #    Does any s in current_sums_set divide ns?
        #    Does ns divide any s in current_sums_set?
        
        # 2. Check new sums against each other
        #    For ns1, ns2 in new_sums, ns1 != ns2:
        #       Does ns1 divide ns2?
        
        # Optimization:
        # Since current_sums_set is already valid, we only check interactions involving new_sums.
        
        # Let's use a boolean array for current sums for fast lookup?
        # Or just iterate.
        # Given the size, iteration might be okay, but array is safer for speed.
        
        # Let's just do direct checks for simplicity first, then optimize if needed.
        # Actually, let's use the divisor/multiple check with a set.
        
        for ns in new_sums:
            # Check against existing sums
            for s in current_sums_set:
                if s == ns:
                    # This can happen if x=0, but x>=1, so s+x > s. 
                    # However, x itself might be in current_sums_set?
                    # No, x is a new element, not in the set.
                    # But x could be equal to an existing sum?
                    # The problem says "no two distinct resulting values may divide one another".
                    # If x is already a sum, then we have two subsets with sum x.
                    # "Two different subsets with the same sum still make the set invalid".
                    # So if ns is in current_sums_set, it's a conflict (equal sums).
                    # Wait, the rule is: "no two distinct resulting values may divide one another".
                    # Equal values are not distinct. But the text says:
                    # "Two different subsets with the same sum still make the set invalid... their union has sum 2s, which s divides."
                    # So equal sums are forbidden.
                    # Thus, if ns is in current_sums_set, return False.
                    return False
                if ns > s:
                    if ns % s == 0:
                        return False
                else:
                    if s % ns == 0:
                        return False
            
            # Check against other new sums
            # To avoid O(N^2) inside the loop, we can just check all pairs in new_sums
            # But we can do it more efficiently by sorting new_sums.
            
        # Check new sums against each other
        sorted_new = sorted(new_sums)
        for i in range(len(sorted_new)):
            ns_i = sorted_new[i]
            # Check if ns_i divides any subsequent ns_j
            for j in range(i + 1, len(sorted_new)):
                ns_j = sorted_new[j]
                if ns_j % ns_i == 0:
                    return False
        # Note: we don't need to check if ns_j divides ns_i because ns_j > ns_i.
        
        return True

    def greedy_construct(order):
        current_set = []
        current_sums = set()
        
        for x in order:
            if is_valid_addition(current_sums, x):
                current_set.append(x)
                # Update sums
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                current_sums.update(new_sums)
                
        return current_set

    # Run multiple restarts with different random orders
    # We can also try specific heuristics like descending order
    candidates = list(range(1, 257))
    
    # Try descending order first
    best_set = greedy_construct(sorted(candidates, reverse=True))
    best_size = len(best_set)
    
    # Try ascending order
    s2 = greedy_construct(sorted(candidates))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # Random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        s = greedy_construct(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        # Optional: Local search?
        # Try to swap an element in best_set with one not in best_set to see if we can keep size or increase?
        # Since we maximize size, if we find a larger set, we update.
        # If we find same size, we might keep the one with larger sum? Doesn't matter for score.
        
    return best_set

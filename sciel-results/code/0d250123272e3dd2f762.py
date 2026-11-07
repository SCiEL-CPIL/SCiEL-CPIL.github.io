import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # State:
    # S: list of ints
    # sum_map: dict {sum_value: set_of_cardinalities}
    # Actually, we just need to know if a sum exists for a specific cardinality.
    # sum_card_map: dict {sum_value: set of cardinalities}
    # If we add x, we generate new sums.
    
    def build_sum_map(S):
        """
        Returns a dictionary: {sum_value: set of cardinalities}
        """
        k = len(S)
        smap = {0: {0}}
        for i in range(k):
            # We can build incrementally, but for simplicity, let's just use combinations
            # or bitmask. 
            # For k up to 20, bitmask is fine.
            pass
            
        # Efficient build using DP/Bitmask
        # smap[s] contains cardinalities of subsets summing to s.
        # Initialize with 0:0
        # For each element x in S:
        #   new_smap = copy of smap
        #   for s, cards in smap.items():
        #       new_s = s + x
        #       new_cards = {c+1 for c in cards}
        #       if new_s in new_smap:
        #           new_smap[new_s].update(new_cards)
        #       else:
        #           new_smap[new_s] = new_cards
        #   smap = new_smap
        #   # Check validity: if any sum has cards from different "levels" that conflict?
        #   # No, validity is: for any sum s, if it appears in cardinality c1 and c2, is that allowed?
        #   # Rule: "Whenever two subsets have different cardinalities their sums must differ."
        #   # This means: If sum s is achieved by subset A (card c1) and subset B (card c2),
        #   # and c1 != c2, then it is INVALID.
        #   # So, for a valid set, each sum value can only be associated with ONE cardinality.
        #   # Wait. "Repeated sums within the SAME cardinality are allowed."
        #   # So if sum s is achieved by two subsets of size c, it's OK.
        #   # If sum s is achieved by one subset of size c1 and another of size c2 (c1!=c2), it's NOT OK.
        #   # Therefore, in a valid set, for every sum s, the set of cardinalities `smap[s]` must have size 1.
        
        smap = {0: {0}}
        for x in S:
            new_smap = {k: set(v) for k, v in smap.items()}
            for s, cards in smap.items():
                new_s = s + x
                new_cards = {c + 1 for c in cards}
                if new_s in new_smap:
                    new_smap[new_s].update(new_cards)
                else:
                    new_smap[new_s] = new_cards
            smap = new_smap
            
        # Validate
        for s, cards in smap.items():
            if len(cards) > 1:
                return None # Invalid
        return smap

    # Initial valid set
    S = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    sum_map = build_sum_map(S)
    if sum_map is None:
        S = list(range(128, 136))
        sum_map = build_sum_map(S)
    
    best_set = S[:]
    best_size = len(S)
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try to add a new element
        added = False
        candidates = [x for x in range(1, 257) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            # New sums are s + x for all s in sum_map
            # New cardinalities are c + 1
            # We need to check if any new sum (s+x) already exists in sum_map with a DIFFERENT cardinality.
            # Note: The new sums might also collide with each other?
            # If s1 + x == s2 + x => s1 == s2.
            # If s1 == s2, then the new sums are identical.
            # Their cardinalities are c1+1 and c2+1.
            # Since s1==s2 in the old map, c1 and c2 are in sum_map[s1].
            # If sum_map[s1] has size 1 (valid old set), then c1 == c2.
            # So c1+1 == c2+1.
            # So new sums collide with same cardinality. This is allowed.
            # So we only need to check collision with EXISTING sums.
            
            valid = True
            for s, old_cards in sum_map.items():
                new_s = s + x
                new_c = min(old_cards) + 1 # Since old_cards is a single value set, min is the value
                
                # Check if new_s is in sum_map
                if new_s in sum_map:
                    existing_cards = sum_map[new_s]
                    # If the existing cardinality is different from new_c, it's a conflict.
                    # Note: new_c is a single value. existing_cards is a set (size 1 for valid set).
                    if new_c not in existing_cards:
                        valid = False
                        break
                
                # Also, we must ensure that the new sum doesn't conflict with OTHER new sums?
                # As argued above, if old set is valid, new sums don't conflict with each other in a forbidden way.
            
            if valid:
                # Update S and sum_map
                S.append(x)
                # Rebuild sum_map? Or update incrementally?
                # Incremental update:
                new_sum_map = {k: set(v) for k, v in sum_map.items()}
                for s, cards in sum_map.items():
                    new_s = s + x
                    new_cards = {c + 1 for c in cards}
                    if new_s in new_sum_map:
                        new_sum_map[new_s].update(new_cards)
                    else:
                        new_sum_map[new_s] = new_cards
                sum_map = new_sum_map
                
                best_set = S[:]
                best_size = len(S)
                added = True
                break # Restart adding process with new set
        
        if not added:
            # Try to swap?
            # Pick a random element from S to remove, and try to add two?
            # Or just restart with a different random seed for candidates?
            # To avoid getting stuck in a local optimum, we can try to remove a random element
            # and then try to add again.
            
            if len(S) > 5:
                # Remove a random element
                idx = random.randint(0, len(S) - 1)
                removed = S.pop(idx)
                # Rebuild sum_map for the smaller set
                sum_map = build_sum_map(S)
                # The loop will try to add elements again.
                # This allows escaping local optima.
                # We don't update best_set here, only if we grow back.
            else:
                # If we are very small, just restart completely?
                # Or break?
                pass

    return [int(x) for x in best_set]

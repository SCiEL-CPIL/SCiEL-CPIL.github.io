import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    def is_valid(S):
        """Check if S has no forbidden triple (a,b,c distinct, a+b, a+c, b+c in S)."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                if (a + b) not in S:
                    continue
                # Need a third distinct c in S such that a+c in S and b+c in S
                for k in range(n):
                    if k == i or k == j:
                        c = S_list[k]
                        if (a + c) in S and (b + c) in S:
                            return False
        return True
    
    def can_add(S, x):
        """Check if adding x to S keeps it valid."""
        if x in S:
            return True
        # After adding x, we need to check that no forbidden triple exists.
        # A forbidden triple involving x: {x, a, b} distinct, with x+a, x+b, a+b all in S_new = S ∪ {x}.
        # Since x is new, the sums involving x are x+a, x+b (and a+b which was already in S or not).
        # The six values: x, a, b, x+a, x+b, a+b.
        # For the triple {x, a, b}: need x+a ∈ S_new, x+b ∈ S_new, a+b ∈ S_new.
        # x+a ∈ S_new means x+a ∈ S or x+a == x (impossible since a≠0).
        # So x+a ∈ S, x+b ∈ S, and a+b ∈ S_new means a+b ∈ S or a+b == x.
        # Also need a, b distinct from x and from each other.
        
        # Also need to check triples that don't involve x but were already checked... 
        # Actually, if S was valid before, adding x only creates new forbidden triples that include x.
        # So we only need to check triples containing x.
        
        # Find all a ∈ S such that x+a ∈ S
        valid_as = [a for a in S if (x + a) in S]
        
        len_va = len(valid_as)
        for i in range(len_va):
            a = valid_as[i]
            for j in range(len_va):
                if i == j:
                    continue
                b = valid_as[j]
                if a == b:
                    continue
                # Check if a+b ∈ S_new, i.e., a+b ∈ S or a+b == x
                if (a + b) in S or (a + b) == x:
                    return False
        return True
    
    def try_add(S, candidates):
        """Try to add candidates to S greedily."""
        added = 0
        for x in candidates:
            if can_add(S, x):
                S.add(x)
                added += 1
        return added
    
    def local_search_add(S, candidates, time_limit):
        """Local search: repeatedly try to add elements until none can be added or time runs out."""
        S = set(S)
        candidates = list(set(candidates) - S)
        while time.time() < time_limit and candidates:
            progress = False
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > time_limit:
                    break
                if can_add(S, x):
                    S.add(x)
                    candidates.remove(x)
                    progress = True
                    break
            if not progress:
                break
        return S
    
    def local_search_swap(S, remove_candidates, add_candidates, time_limit):
        """Try swapping: remove an element from S, add an element not in S, to increase size."""
        # This is for maintaining size while trying different configurations
        # For now, focus on addition-based search
        return S
    
    # Initial valid set: upper 96 integers {33, ..., 128}
    # Verify: for any three distinct a,b,c in [33,128], a+b ≥ 33+34=67, but we need a+b, a+c, b+c all in S.
    # The minimum sum of two distinct elements in S is 33+34=67, which is in S.
    # So we need to check if there exist a<b<c in S with a+b, a+c, b+c all in S.
    # Let's just verify computationally that {33,...,128} is valid.
    
    S0 = set(range(33, 129))
    # Quick check
    assert is_valid(S0), "Initial set {33,...,128} should be valid"
    
    best_set = set(S0)
    best_size = len(S0)
    
    # Try to add elements from {1, ..., 32}
    small_candidates = list(range(1, 33))
    
    # Multiple restarts with different orderings
    while time.time() < deadline:
        S = set(best_set)
        
        # Shuffle and try to add
        candidates = small_candidates.copy()
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # If we've added all we can, try more sophisticated local search
        if len(S) == best_size:
            # Try random restarts with different initial subsets
            # Sometimes a different choice of which elements to include leads to a better final set
            if time.time() < deadline - 5:
                # Try starting from a smaller base and building up
                base_size = random.randint(50, 96)
                base = random.sample(range(33, 129), base_size)
                S_base = set(base)
                # Verify base is valid
                if not is_valid(S_base):
                    continue
                
                # Now try to add from the rest
                remaining = list(set(range(33, 129)) - S_base)
                remaining += [x for x in range(1, 33) if x not in S_base]
                random.shuffle(remaining)
                for x in remaining:
                    if time.time() > deadline:
                        break
                    if can_add(S_base, x):
                        S_base.add(x)
                
                if len(S_base) > best_size:
                    best_size = len(S_base)
                    best_set = set(S_base)
    
    # Final verification
    result = list(best_set)
    assert all(isinstance(x, int) for x in result)
    assert all(1 <= x <= 128 for x in result)
    assert len(result) == len(set(result))
    assert is_valid(set(result)), "Final set must be valid"
    
    return result

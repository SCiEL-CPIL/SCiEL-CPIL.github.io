# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0  # Safety margin before 30s limit
    
    def can_add(x, S, sums):
        # Check 2x
        if 2 * x in sums:
            return False
        # Check x + a for a in S
        for a in S:
            if x + a in sums:
                return False
        return True

    def get_sums_sorted(S):
        """Return sorted list of all pairwise sums a+b with a<=b."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sorted(sums)

    def is_maximal_fast(S):
        """Check if S is inclusion-maximal."""
        if not S:
            return False
        sums_set = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums_set.add(S[i] + S[j])
        S_set = set(S)
        # Try to find any addable element
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Quick check: if 2*x is in sums, x cannot be added
            if 2 * x in sums_set:
                continue
            # Check if x + a is in sums for any a in S
            addable = True
            for a in S:
                if x + a in sums_set:
                    addable = False
                    break
            if addable:
                return False  # x can be added, so not maximal
        return True

    def greedy_construct_lookahead(order):
        """
        Greedy construction that picks the element which blocks the most other candidates.
        order is the initial list of candidates.
        """
        S = []
        sums = set()
        candidates = list(order)
        
        while candidates and time.time() < deadline:
            best_x = None
            max_blocked = -1
            
            # Evaluate each candidate
            # Optimization: We can break early if we find a perfect block? No, we want max block.
            # We can prune candidates that are no longer valid? 
            # Yes, we should filter candidates to only those that are currently addable.
            valid_candidates = []
            for x in candidates:
                if can_add(x, S, sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # If only one, pick it
            if len(valid_candidates) == 1:
                best_x = valid_candidates[0]
            else:
                # Evaluate blocking power
                for x in valid_candidates:
                    # Simulate adding x
                    new_sums = sums.copy()
                    new_sums.add(2 * x)
                    for a in S:
                        new_sums.add(x + a)
                    
                    # Count how many other valid_candidates are now blocked
                    blocked_count = 0
                    for y in valid_candidates:
                        if y == x:
                            continue
                        # Check if y is blocked by new_sums
                        if 2 * y in new_sums:
                            blocked_count += 1
                            continue
                        is_blocked = False
                        for a in S:
                            if y + a in new_sums:
                                is_blocked = True
                                break
                        if not is_blocked:
                            # Check if y + x is in new_sums? 
                            # y+x is in new_sums if it was in sums (impossible as y,x valid) 
                            # or if y+x == 2z (impossible) or y+x == x+a => y=a (impossible) 
                            # Wait, the condition for y being addable to S U {x} is:
                            # 2y not in new_sums AND for all a in S U {x}, y+a not in new_sums.
                            # We checked 2y.
                            # We checked y+a for a in S.
                            # We need to check y+x.
                            # y+x is in new_sums? 
                            # new_sums contains x+a for a in S. 
                            # Does new_sums contain y+x? 
                            # Only if y+x = 2x (y=x, no) or y+x = x+a (y=a, no).
                            # So y+x is NOT in new_sums.
                            # Wait, the Sidon condition for S U {x} U {y} requires:
                            # 2y not in sums(S U {x})
                            # y+a not in sums(S U {x}) for all a in S U {x}
                            # We checked y+a for a in S.
                            # We must check y+x.
                            # Is y+x in sums(S U {x})?
                            # sums(S U {x}) = sums(S) U {2x} U {x+a for a in S}.
                            # y+x is in this set if:
                            # 1. y+x in sums(S). But y was valid for S, so y+a not in sums(S) for any a in S. 
                            #    Specifically, if we set a such that y+a = y+x => a=x. x is not in S.
                            #    So y+x is NOT in sums(S).
                            # 2. y+x == 2x => y=x. No.
                            # 3. y+x == x+a => y=a. No.
                            # So y+x is NOT in sums(S U {x}).
                            # Therefore, if y passed the checks for a in S, it is still addable to S U {x}.
                            # So my logic above was correct.
                            pass 
                        
                        if is_blocked:
                            blocked_count += 1
                            
                    if blocked_count > max_blocked:
                        max_blocked = blocked_count
                        best_x = x
            
            # Add best_x to S
            S.append(best_x)
            sums.add(2 * best_x)
            for a in S[:-1]:
                sums.add(best_x + a)
            
            # Remove best_x from candidates
            candidates = [x for x in candidates if x != best_x]
            
            # Optional: Remove candidates that are now invalid? 
            # The loop at the start of the while does this implicitly by filtering valid_candidates.
            # But keeping the list small is good.
            
        return S

    def prune_aggressive(S):
        """Try to remove elements while maintaining maximality."""
        S = list(S)
        if not S:
            return S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            
            indices = list(range(n))
            # Try removals in order of increasing value
            indices_sorted = sorted(indices, key=lambda i: S[i])
            
            for idx in indices_sorted:
                if time.time() > deadline:
                    break
                new_S = S[:idx] + S[idx+1:]
                if not new_S:
                    continue
                if is_maximal_fast(new_S):
                    S = new_S
                    improved = True
                    break
            
            if not improved and n > 2:
                indices_sorted2 = sorted(indices, key=lambda i: -S[i])
                for idx in indices_sorted2:
                    if time.time() > deadline:
                        break
                    new_S = S[:idx] + S[idx+1:]
                    if not new_S:
                        continue
                    if is_maximal_fast(new_S):
                        S = new_S
                        improved = True
                        break
        
        return S

    def local_search(S):
        """Perform local search: try single-element swaps to reduce size."""
        S = list(S)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            
            for i in range(n):
                if time.time() > deadline:
                    break
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                if is_maximal_fast(new_S):
                    S = new_S
                    improved = True
                    break
        
        return S

    def prune_double_removal(S):
        """Try removing two elements at a time."""
        S = list(S)
        if not S:
            return S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 2:
                break
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    new_S = S[:i] + S[i+1:j] + S[j+1:]
                    if not new_S:
                        continue
                    if is_maximal_fast(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        
        return S

    # Phase 1: Deterministic Descending Greedy (Fallback)
    order_desc = list(range(N, 0, -1))
    S_init = greedy_construct_lookahead(order_desc)
    S_init = prune_aggressive(S_init)
    best_set = S_init
    best_size = len(S_init)
    
    # Phase 2: Randomized Restarts
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        order = list(all_elements)
        rng.shuffle(order)
        
        S = greedy_construct_lookahead(order)
        
        S_pruned = prune_aggressive(S)
        S_double = prune_double_removal(S_pruned)
        S_local = local_search(S_double)
        
        if len(S_local) < best_size:
            best_size = len(S_local)
            best_set = S_local
    
    # Final verification
    if not is_maximal_fast(best_set):
        best_set = greedy_construct_lookahead(order_desc)
    
    return best_set

# EVOLVE-BLOCK-END

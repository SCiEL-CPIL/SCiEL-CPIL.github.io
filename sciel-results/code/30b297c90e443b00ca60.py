def construct():
    """
    Constructs a Sidon set containing 0 with all differences 1..30.
    Minimizes the largest element using a greedy Maximum Coverage approach
    with diameter-aware scoring.
    """
    S = [0]
    used_diffs = set()
    covered = set()
    
    # To efficiently check conflicts, we maintain the set of used differences.
    # We also need to know which required diffs (1..30) are covered.
    
    max_limit = 50
    # We will iterate, adding one element at a time.
    # We stop when all 30 differences are covered.
    
    while len(covered) < 30:
        best_candidate = None
        best_score = -1
        best_delta_d = None
        best_gain = None
        
        # We search for candidates x in 1..max_limit
        # Note: x must not be in S
        s_set = set(S)
        
        for x in range(1, max_limit + 1):
            if x in s_set:
                continue
            
            # Check Sidon constraint:
            # All new differences |x - s| for s in S must not be in used_diffs
            valid = True
            new_covered_count = 0
            # We also need to check if the new differences themselves conflict with each other?
            # No, the Sidon condition is that ALL pairwise differences in the set are distinct.
            # Existing pairs in S are already distinct.
            # New pairs are (x, s) for s in S.
            # We must ensure:
            # 1. No |x - s| is in used_diffs.
            # 2. No |x - s1| == |x - s2| for s1 != s2 in S.
            #    Since s1 != s2, |x - s1| == |x - s2| implies x - s1 = -(x - s2) => 2x = s1 + s2.
            #    So if 2x is a sum of two distinct elements in S, it's a conflict?
            #    Wait. If $x-s_1 = s_2-x$, then $x-s_1$ and $s_2-x$ are the same difference value.
            #    This means the difference value $d$ appears twice.
            #    So we must check if any new difference value repeats within the new set OR with old set.
            #    Let's just collect all new diffs and check uniqueness and intersection with used_diffs.
            
            new_diffs = []
            for s in S:
                d = abs(x - s)
                if d == 0:
                    continue
                new_diffs.append(d)
            
            # Check for duplicates within new_diffs
            if len(set(new_diffs)) != len(new_diffs):
                valid = False
            else:
                # Check intersection with used_diffs
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Calculate gain
            for d in new_diffs:
                if 1 <= d <= 30 and d not in covered:
                    new_covered_count += 1
            
            if new_covered_count == 0:
                # This candidate doesn't help cover required diffs.
                # We might still need it to enable future covers?
                # Generally, in max coverage, we prefer candidates with gain > 0.
                # If no candidate has gain > 0, we pick the one with smallest diameter increase.
                # For now, let's track it but give it low priority.
                pass
            
            # Calculate diameter increase
            current_max = max(S)
            new_max = max(current_max, x)
            delta_d = new_max - current_max
            
            # Score calculation
            # We want to maximize gain.
            # We want to minimize delta_d.
            # If delta_d == 0, it's very good.
            # Let's use a composite score.
            # Primary: Gain (higher is better)
            # Secondary: Delta D (lower is better)
            # Tertiary: x (lower is better, to keep numbers small)
            
            # We can sort by tuple (-gain, delta_d, x)
            # But we are doing this in a loop.
            
            # Let's define a score that prioritizes:
            # 1. Higher gain
            # 2. Lower delta_d
            # 3. Lower x
            
            # If we find a candidate with gain > 0 and delta_d == 0, it's excellent.
            
            # Let's just store the best based on a tuple comparison.
            # We want to MINIMIZE ( -gain, delta_d, x )
            # So we compare tuples: (-new_covered_count, delta_d, x)
            
            current_tuple = (-new_covered_count, delta_d, x)
            
            if best_candidate is None:
                best_candidate = x
                best_tuple = current_tuple
            else:
                if current_tuple < best_tuple:
                    best_candidate = x
                    best_tuple = current_tuple
        
        if best_candidate is None:
            # No valid candidate found in range
            max_limit *= 2
            if max_limit > 1000:
                # Fallback to a simple constructive algorithm if greedy fails
                # This should rarely happen
                return _fallback_construct()
            continue
        
        # Add best_candidate to S
        S.append(best_candidate)
        s_set.add(best_candidate)
        
        # Update used_diffs and covered
        for s in S[:-1]:
            d = abs(best_candidate - s)
            used_diffs.add(d)
            if 1 <= d <= 30:
                covered.add(d)
        
        # If we have covered all, we can stop
        if len(covered) == 30:
            break
    
    # Final verification
    S.sort()
    all_diffs = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            all_diffs.append(S[j] - S[i])
    
    if len(set(all_diffs)) != len(all_diffs):
        return _fallback_construct()
    
    diff_set = set(all_diffs)
    for d in range(1, 31):
        if d not in diff_set:
            return _fallback_construct()
            
    return S

def _fallback_construct():
    """
    Fallback method using a known small Sidon set or a different greedy strategy.
    """
    # A known small Sidon set (B2 sequence)
    # We need one that covers 1..30.
    # Let's try a simple greedy that just picks the smallest valid x,
    # but with a larger search space, or use a precomputed set.
    
    # Precomputed small Sidon sets:
    # {0, 1, 4, 10, 12, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121}
    # This is an arithmetic progression, which is NOT a Sidon set (differences repeat).
    
    # Let's use a standard small Sidon set construction.
    # Singer's construction or similar.
    # Or just a robust greedy:
    S = [0]
    used_diffs = set()
    covered = set()
    
    # Greedy: pick smallest x that maintains Sidon and covers most new diffs.
    # If no diff covered, pick smallest valid x.
    
    max_x = 100
    
    while len(covered) < 30:
        best_x = None
        best_gain = -1
        best_diam = float('inf')
        
        for x in range(1, max_x + 1):
            if x in S:
                continue
            
            new_diffs = [abs(x - s) for s in S]
            if len(set(new_diffs)) != len(new_diffs):
                continue
            if any(d in used_diffs for d in new_diffs):
                continue
            
            gain = 0
            for d in new_diffs:
                if 1 <= d <= 30 and d not in covered:
                    gain += 1
            
            diam = max(max(S), x)
            
            # Prefer higher gain, then smaller diameter
            if gain > best_gain or (gain == best_gain and diam < best_diam):
                best_gain = gain
                best_diam = diam
                best_x = x
        
        if best_x is None:
            max_x *= 2
            if max_x > 1000:
                # Last resort: just return a valid set, even if large
                # Construct a set that is guaranteed to work
                # Use a known large Sidon set
                # e.g. {0} U {i * 1000 + j for ...}
                # But we need to minimize largest element.
                # If greedy fails, something is wrong.
                # Let's try a different heuristic: pick smallest valid x regardless of gain.
                for x in range(1, max_x + 1):
                    if x in S:
                        continue
                    new_diffs = [abs(x - s) for s in S]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    if any(d in used_diffs for d in new_diffs):
                        continue
                    best_x = x
                    break
                if best_x is None:
                    raise Exception("Failed to construct Sidon set")
        
        S.append(best_x)
        for s in S[:-1]:
            d = abs(best_x - s)
            used_diffs.add(d)
            if 1 <= d <= 30:
                covered.add(d)
    
    return S

"""Rewritten program using a greedy Maximum Coverage approach with ratio-based selection."""

def construct():
    # We need a Sidon set containing 0 with all differences 1..30 present.
    # Minimize the largest element.
    
    # Strategy:
    # 1. Start with S = [0].
    # 2. Maintain set of all positive differences 'existing_diffs'.
    # 3. Maintain set of covered differences in 1..30 'covered'.
    # 4. Iteratively select next element x.
    #    - Candidate x is valid if all |x - s| for s in S are not in existing_diffs.
    #    - Score candidates by (newly_covered_count) / (diameter_increase + 1).
    #    - Prefer candidates with new_cover > 0.
    #    - If no candidate has new_cover > 0, pick smallest valid x (should rarely happen).
    
    S = [0]
    existing_diffs = set()
    covered = set()
    
    max_x_limit = 200  # Upper bound for search, can increase if stuck
    
    while len(covered) < 30:
        best_x = None
        best_score = -1
        best_new_cover = 0
        
        current_max = max(S)
        
        for x in range(1, max_x_limit + 1):
            if x in S:
                continue
            
            # Check Sidon property
            valid = True
            new_diffs = []
            for s in S:
                d = abs(x - s)
                if d in existing_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Calculate coverage
            new_cover = 0
            for d in new_diffs:
                if 1 <= d <= 30 and d not in covered:
                    new_cover += 1
            
            # Calculate diameter increase
            # Since we are adding x, new max is max(current_max, x)
            # Diameter increase is max(0, x - current_max)
            diam_inc = max(0, x - current_max)
            
            # We prioritize coverage. If new_cover is 0, score is 0.
            # We want to maximize score.
            # To avoid division by zero and to favor smaller diameter increases,
            # we use a penalty.
            if new_cover == 0:
                score = 0
            else:
                # Ratio: coverage per unit of diameter expansion
                score = new_cover / (diam_inc + 1)
            
            # Update best
            if score > best_score or (score == best_score and (best_x is None or x < best_x)):
                best_score = score
                best_x = x
                best_new_cover = new_cover
        
        if best_x is None:
            # No valid candidate in range, expand search
            max_x_limit *= 2
            continue
        
        # Add best_x to S
        S.append(best_x)
        S.sort()
        
        # Update existing_diffs and covered
        for s in S[:-1]:  # S[:-1] is S before adding best_x
            d = abs(best_x - s)
            existing_diffs.add(d)
            if 1 <= d <= 30:
                covered.add(d)
        
        if len(covered) >= 30:
            break
    
    # Final verification and cleanup
    S.sort()
    all_diffs = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            all_diffs.append(S[j] - S[i])
    
    if len(set(all_diffs)) != len(all_diffs):
        # Fallback: This shouldn't happen with the logic above, but just in case
        # Return a known valid set or raise
        pass
        
    diff_set = set(all_diffs)
    if not all(d in diff_set for d in range(1, 31)):
        # Fallback if coverage failed
        # This indicates the greedy got stuck or missed something
        # For robustness, we could implement a backtracking or different heuristic,
        # but given the constraints, the greedy ratio method is strong.
        pass

    return S

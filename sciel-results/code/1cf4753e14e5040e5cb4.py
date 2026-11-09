import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    best_max = float('inf')
    best_set = None
    
    # Beam Search
    # State: (max_val, num_missing, S_tuple, used_diffs_set, covered_mask)
    # We prioritize small max_val and high coverage.
    
    S_init = (0,)
    used_diffs_init = set()
    covered_mask_init = 0
    max_val_init = 0
    
    beam = [(max_val_init, 30, S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)
    
    def get_candidates(S, used_diffs, covered_mask):
        candidates = set()
        S_list = list(S)
        max_s = max(S_list) if S_list else 0
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # 1. Candidates that directly cover a missing difference
        # x = s + d or x = s - d (but x > max_s to keep sorted order simple? 
        # Actually, we can insert anywhere. But for beam search simplicity, 
        # appending larger values is a common heuristic for Sidon sets.
        # However, to minimize max, we might want to fill gaps.
        # Let's generate x such that x - s = d for some s in S.
        for d in missing_diffs:
            for s in S_list:
                x = s + d
                if x > max_s and x <= 10**6:
                    candidates.add(x)
                # Also x = s - d? If x > max_s, s-d > max_s implies d < s-max_s.
                # But usually Sidon sets are constructed by adding larger elements.
                # Let's stick to x > max_s for now to avoid complex re-sorting and 
                # because "truncated perfect difference sets" often look like 
                # initial segments of quadratic residues or similar, which are increasing.
                
        # 2. Random candidates near the max to explore local structure
        if S_list:
            for _ in range(5):
                # Try values slightly larger than max
                rand_x = max_s + rng.randint(1, 20)
                if rand_x <= 10**6:
                    candidates.add(rand_x)
        
        valid_candidates = []
        
        for x in candidates:
            if x <= max_s:
                continue
            if x > 10**6:
                continue
            
            # Check Sidon property incrementally
            # New diffs are x - s for s in S_list
            new_diffs = []
            valid = True
            for s in S_list:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Update mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            new_missing = 30 - bin(new_mask).count('1')
            
            # Score: prefer smaller x, then more coverage
            valid_candidates.append((x, new_mask, new_diffs, new_missing))
            
        # Sort by x, then by coverage
        valid_candidates.sort(key=lambda item: (item[0], item[3]))
        return valid_candidates[:10] # Limit candidates per state

    beam_width = 2000
    max_steps = 32
    
    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break
            
        new_beam = []
        for max_val, num_missing, S, used_diffs, covered_mask in beam:
            if time.time() - start_time > deadline:
                break
                
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
                continue
            
            if len(S) >= 32:
                continue
                
            candidates = get_candidates(S, used_diffs, covered_mask)
            for x, new_mask, new_diffs, new_missing in candidates:
                new_S = S + (x,)
                new_used_diffs = used_diffs.union(new_diffs)
                new_max = x
                
                new_beam.append((new_max, new_missing, new_S, new_used_diffs, new_mask))
                
                if time.time() - start_time > deadline:
                    break
                    
        if not new_beam:
            break
            
        # Prune beam
        new_beam.sort(key=lambda item: (item[0], item[1]))
        
        seen = set()
        unique_beam = []
        for state in new_beam:
            s_tuple = state[2]
            if s_tuple not in seen:
                seen.add(s_tuple)
                unique_beam.append(state)
            if len(unique_beam) >= beam_width:
                break
                
        beam = unique_beam
        
        # If we found a good solution, we can prune states that are worse
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break

    if best_set:
        # Local Search Polish
        S = best_set[:]
        improved = True
        polish_deadline = deadline - 10 # Leave 10s for safety
        
        while improved and (time.time() - start_time) < polish_deadline:
            improved = False
            S.sort()
            n = len(S)
            
            # Try to reduce each element from largest to smallest
            for i in range(n - 1, 0, -1):
                if (time.time() - start_time) > polish_deadline:
                    break
                    
                s_i = S[i]
                S_rest = S[:i] + S[i+1:]
                S_rest_set = set(S_rest)
                
                # Determine which diffs are covered ONLY by s_i in S
                # Diffs involving s_i: s_i - s, s - s_i for s in S_rest
                # Diffs not involving s_i: diffs of S_rest
                diffs_rest = get_diffs(S_rest)
                
                # Diffs provided by s_i
                diffs_with_i = set()
                for s in S_rest:
                    diffs_with_i.add(abs(s_i - s))
                
                # Diffs that are in target_diffs AND in diffs_with_i AND NOT in diffs_rest
                # These are the "critical" diffs that s_i must continue to cover (or be covered by new element)
                critical_diffs = target_diffs & diffs_with_i - diffs_rest
                
                # We want to replace s_i with s_j < s_i
                # s_j must be > max(S_rest[:i]) to maintain sorted order if we just replace?
                # No, we can insert anywhere. But to minimize max, we want s_j as small as possible.
                # However, s_j must not create duplicate diffs with S_rest.
                
                # Lower bound for s_j:
                # If i > 0, s_j > S[i-1] is not strictly required if we re-sort, 
                # but s_j must be distinct from S_rest.
                # Let's try values from S[i-1]+1 to s_i-1.
                # To be efficient, try a few strategic values.
                
                lower_bound = S[i-1] + 1 if i > 0 else 1
                upper_bound = s_i - 1
                
                if upper_bound < lower_bound:
                    continue
                
                # Try a subset of candidates
                # 1. s_i - 1, s_i - 2, ...
                # 2. Values that cover critical_diffs
                # If critical_diffs is not empty, s_j must generate them.
                # s_j - s = d or s - s_j = d => s_j = s + d or s_j = s - d
                # Since s_j < s_i, and s < s_i (as s in S_rest, s_i is largest), 
                # s_j = s + d is likely.
                
                candidate_vals = set()
                
                # Add some small offsets
                for offset in range(1, min(10, upper_bound - lower_bound + 1)):
                    candidate_vals.add(s_i - offset)
                
                # Add values derived from critical diffs
                if critical_diffs:
                    for d in critical_diffs:
                        for s in S_rest:
                            # s_j = s + d
                            v1 = s + d
                            if lower_bound <= v1 <= upper_bound:
                                candidate_vals.add(v1)
                            # s_j = s - d (if s > d)
                            v2 = s - d
                            if lower_bound <= v2 <= upper_bound:
                                candidate_vals.add(v2)
                
                # Add random
                if upper_bound > lower_bound:
                    for _ in range(3):
                        candidate_vals.add(rng.randint(lower_bound, upper_bound))
                
                for new_val in sorted(candidate_vals, reverse=True): # Try larger first? 
                    # Actually, we want to minimize max, so if we replace s_i with new_val < s_i,
                    # the new max is max(S_rest, new_val). Since new_val < s_i, max is likely max(S_rest) or new_val.
                    # To minimize the new max, we should prefer smaller new_val?
                    # No, the max of the set becomes max(max(S_rest), new_val).
                    # Since new_val < s_i, and s_i was the max, the new max is at most s_i.
                    # To minimize the *overall* max, we just need any valid new_val < s_i.
                    # But maybe a smaller new_val allows further reduction of other elements?
                    # Let's try values in descending order to keep the set "dense" near the top,
                    # or ascending to push it down?
                    # Let's try descending (closest to s_i) first, as it's less likely to break Sidon property with lower elements?
                    # Actually, moving further away (smaller) might create conflicts with lower elements?
                    # Let's just iterate and pick the first valid one.
                    
                    if new_val in S_rest_set:
                        continue
                    
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
                
                if improved:
                    break  # Restart outer loop with new S

        return S
    else:
        # Fallback: Greedy
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in missing:
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S

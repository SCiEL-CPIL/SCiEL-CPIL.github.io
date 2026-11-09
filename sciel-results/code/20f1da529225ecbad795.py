import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    best_max = float('inf')
    best_set = None
    
    # Beam Search Parameters
    beam_width = 1000
    max_candidates_per_state = 50
    
    # State: (max_val, num_missing, S, used_diffs, covered_mask)
    # We store S as a tuple for hashing/comparison if needed, but list is fine.
    # used_diffs as a frozenset for hashing.
    
    # Initial state
    S_init = (0,)
    used_diffs_init = frozenset()
    covered_mask_init = 0
    max_val_init = 0
    
    beam = [(max_val_init, 30, S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)
    
    def get_candidates(S, used_diffs, covered_mask):
        candidates = set()
        for d in range(1, 31):
            if not (covered_mask & (1 << (d - 1))):
                for s in S:
                    x = s + d
                    candidates.add(x)
        
        # Add some random candidates to escape local minima
        if S:
            last = max(S)
            for _ in range(10):
                rand_x = last + rng.randint(1, 50)
                if rand_x <= 10**6:
                    candidates.add(rand_x)
        
        valid_candidates = []
        S_list = list(S)
        last = S_list[-1] if S_list else -1
        
        for x in candidates:
            if x <= last:
                continue
            if x > 10**6:
                continue
            
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
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            new_missing = 30 - bin(new_mask).count('1')
            # Score: prefer more coverage, then smaller x
            valid_candidates.append((x, new_mask, new_diffs, new_missing))
            
        # Sort and limit
        valid_candidates.sort(key=lambda item: (item[3], item[0]))
        return valid_candidates[:max_candidates_per_state]

    for step in range(32):
        if time.time() - start_time > deadline:
            break
            
        new_beam = []
        for max_val, num_missing, S, used_diffs, covered_mask in beam:
            if time.time() - start_time > deadline:
                break
                
            # If we already cover all, we can still try to reduce max?
            # No, we are adding elements. Max will increase.
            # If covered_mask == full_mask, we have a solution.
            # We should record it and stop expanding this branch?
            # Or keep it in the beam to see if we can find a smaller max?
            # No, adding elements increases max.
            # So if covered, we just update best and don't expand.
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
                continue
            
            candidates = get_candidates(S, used_diffs, covered_mask)
            for x, new_mask, new_diffs, new_missing in candidates:
                new_S = S + (x,)
                new_used_diffs = used_diffs.union(new_diffs)
                new_max = x # Since x > S[-1]
                
                new_beam.append((new_max, new_missing, new_S, new_used_diffs, new_mask))
                
                if time.time() - start_time > deadline:
                    break
                    
        if not new_beam:
            break
            
        # Sort and prune beam
        # Sort by (num_missing, max_val)
        new_beam.sort(key=lambda item: (item[1], item[0]))
        
        # Deduplicate states?
        # State is defined by S. If two paths lead to same S, they are identical.
        # We can use a set of tuples S to dedupe.
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
        
        # If we found a solution in this step, we can potentially stop if we are confident?
        # But we want the MIN max.
        # The beam keeps the best states.
        # If we have a solution with max M, any state in the beam with max > M and missing > 0 is worse?
        # Not necessarily, it might lead to a solution with max < M? No, max only increases.
        # So if we have a solution with max M, we can prune any state in the beam with max_val >= M.
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break

    if best_set:
        return best_set
    else:
        # Fallback: Return a valid set even if not optimal
        # Greedy without backtracking
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
                # Add a large number to unblock?
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

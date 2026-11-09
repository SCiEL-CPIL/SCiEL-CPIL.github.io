import time
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
    # We can afford a larger beam if we are fast.
    # Let's start with 500 and see.
    beam_width = 500
    max_candidates_per_state = 50
    
    # State: (max_val, num_missing, S, used_diffs, covered_mask)
    S_init = (0,)
    used_diffs_init = frozenset()
    covered_mask_init = 0
    max_val_init = 0
    
    beam = [(max_val_init, 30, S_init, used_diffs_init, covered_mask_init)]
    
    def get_candidates(S, used_diffs, covered_mask):
        candidates = set()
        for d in range(1, 31):
            if not (covered_mask & (1 << (d - 1))):
                for s in S:
                    x = s + d
                    candidates.add(x)
        
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
                
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
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
            
        new_beam.sort(key=lambda item: (item[1], item[0]))
        
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
        
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break

    if best_set:
        return best_set
    else:
        # Fallback
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

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
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
    
    def try_polish(S, time_limit):
        """Try to reduce the maximum element via local search."""
        S = sorted(S)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(S)
            # Try to reduce the largest element
            if n > 1:
                s_max = S[-1]
                S_rest = S[:-1]
                S_rest_set = set(S_rest)
                
                # Find which target diffs are ONLY covered by s_max
                diffs_rest = get_diffs(S_rest)
                diffs_with_max = set()
                for s in S_rest:
                    diffs_with_max.add(s_max - s)
                critical = (target_diffs & diffs_with_max) - diffs_rest
                
                # Try to replace s_max with a smaller value
                # We need new_val > max(S_rest) to keep it as max? 
                # No, we just need it to be distinct.
                # But if we want to minimize max, we want new_val < s_max.
                # If new_val < max(S_rest), then the new max is max(S_rest).
                # If new_val > max(S_rest), the new max is new_val.
                
                lower_bound = max(S_rest) + 1 # Actually, we can insert anywhere.
                # Let's just try values from 1 to s_max - 1.
                # To be efficient, we only try values that help cover critical diffs 
                # or are close to current max.
                
                candidate_vals = set()
                # Values that cover critical diffs
                if critical:
                    for d in critical:
                        for s in S_rest:
                            v1 = s + d
                            v2 = s - d
                            if 1 <= v1 < s_max:
                                candidate_vals.add(v1)
                            if 1 <= v2 < s_max:
                                candidate_vals.add(v2)
                
                # Try values just below s_max
                for d in range(1, min(50, s_max)):
                    candidate_vals.add(s_max - d)
                
                # Try values just above max(S_rest)
                max_rest = max(S_rest)
                for d in range(1, min(50, s_max - max_rest)):
                    candidate_vals.add(max_rest + d)
                
                for new_val in sorted(candidate_vals):
                    if new_val in S_rest_set:
                        continue
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
            
            if (time.time() - start_time) > time_limit:
                break
                
        return S

    # --- Phase 1: Beam Search ---
    # State: (S_tuple, used_diffs_set, covered_mask)
    # We keep a beam of the most promising states.
    
    S_init = (0,)
    used_diffs_init = set()
    covered_mask_init = 0
    
    beam = [(S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)
    
    def get_candidates(S, used_diffs, covered_mask, max_val):
        candidates = set()
        S_list = list(S)
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # 1. Candidates from missing diffs
        for d in missing_diffs:
            for s in S_list:
                x = s + d
                if x > max_val and x <= 10**6:
                    candidates.add(x)
        
        # 2. Candidates to fill gaps (small x)
        if S_list:
            # Try small x values that might be valid
            # We can't just try 1,2,3... as they are likely invalid.
            # But we can try values derived from existing diffs.
            pass
            
        # 3. Random candidates near max
        for _ in range(10):
            rand_x = max_val + rng.randint(1, 100)
            if rand_x <= 10**6:
                candidates.add(rand_x)
        
        valid_candidates = []
        
        for x in candidates:
            if x <= max_val:
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
            
            valid_candidates.append((x, new_mask, new_diffs, new_missing))
            
        valid_candidates.sort(key=lambda item: (item[0], item[3]))
        return valid_candidates[:20]

    beam_width = 5000
    max_steps = 32
    
    for step in range(max_steps):
        if time.time() - start_time > deadline - 20: # Leave 20s for polish
            break
            
        new_beam = []
        for S, used_diffs, covered_mask in beam:
            if time.time() - start_time > deadline - 20:
                break
                
            if covered_mask == full_mask:
                if max(S) < best_max:
                    best_max = max(S)
                    best_set = list(S)
                continue
            
            if len(S) >= 32:
                continue
                
            candidates = get_candidates(S, used_diffs, covered_mask, max(S))
            for x, new_mask, new_diffs, new_missing in candidates:
                new_S = S + (x,)
                new_used_diffs = used_diffs.union(new_diffs)
                new_beam.append((new_S, new_used_diffs, new_mask))
                
                if time.time() - start_time > deadline - 20:
                    break
                    
        if not new_beam:
            break
            
        # Prune beam: keep unique sets, sort by max then missing
        new_beam.sort(key=lambda item: (max(item[0]), bin(item[2]).count('0')))
        
        seen = set()
        unique_beam = []
        for state in new_beam:
            s_tuple = state[0]
            if s_tuple not in seen:
                seen.add(s_tuple)
                unique_beam.append(state)
            if len(unique_beam) >= beam_width:
                break
                
        beam = unique_beam
        
        if best_max < float('inf'):
            beam = [s for s in beam if max(s[0]) < best_max]
            if not beam:
                break

    # --- Phase 2: Polish ---
    if best_set:
        polish_deadline = deadline - 5
        S = try_polish(best_set, polish_deadline)
        if max(S) < best_max:
            best_max = max(S)
            best_set = S
        
        # Try a few random restarts of polish if time permits
        while time.time() - start_time < polish_deadline and best_set:
            # Mutate best_set slightly and polish
            S_mut = best_set[:]
            # Remove a random element (not 0) and re-add
            if len(S_mut) > 1:
                idx = rng.randint(1, len(S_mut) - 1)
                val = S_mut.pop(idx)
                # Try to re-insert a nearby value
                new_val = val + rng.randint(-50, 50)
                if new_val > 0 and new_val not in S_mut:
                    S_mut.append(new_val)
                    S_mut.sort()
                    if is_sidon(S_mut) and covers_all(S_mut):
                        S_new = try_polish(S_mut, polish_deadline)
                        if max(S_new) < best_max:
                            best_max = max(S_new)
                            best_set = S_new

    if best_set:
        return best_set
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


import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    # Precompute structures for a given n
    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        k4s = list(combinations(range(n), 4))
        num_k4 = len(k4s)
        # k4_incidence[k4_idx] = list of triple indices in that k4
        k4_incidence = []
        triple_to_k4 = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4s):
            ts = []
            for t in combinations(k4, 3):
                # Find index of t in triples
                # Since triples is sorted lexicographically, we can map it
                # But for speed, let's just use a dict or calculate index
                # Actually, calculating index from (a,b,c) is possible but dict is easier for setup
                pass
            
            # Better: build triple_idx map
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        for i, k4 in enumerate(k4s):
            ts = []
            for t in combinations(k4, 3):
                t_idx = triple_idx[t]
                ts.append(t_idx)
                triple_to_k4[t_idx].append(i)
            k4_incidence.append(ts)
            
        k5s = list(combinations(range(n), 5))
        num_k5 = len(k5s)
        k5_incidence = []
        triple_to_k5 = [[] for _ in range(num_triples)]
        
        for i, k5 in enumerate(k5s):
            ts = []
            for t in combinations(k5, 3):
                t_idx = triple_idx[t]
                ts.append(t_idx)
                triple_to_k5[t_idx].append(i)
            k5_incidence.append(ts)
                
        return num_triples, k4_incidence, k5_incidence, triple_to_k4, triple_to_k5

    def try_solve(n, time_limit):
        if n < 4: return None
        
        num_triples, k4_incidence, k5_incidence, triple_to_k4, triple_to_k5 = get_structures(n)
        num_k4 = len(k4_incidence)
        num_k5 = len(k5_incidence)
        
        if num_k4 == 0 or num_k5 == 0:
             # Trivial cases
             return [0] * num_triples

        # Initialize random coloring
        # 0: Red, 1: Blue
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Precompute incidence matrices for fast updates
        # M4: shape (num_k4, num_triples), entries 1 if triple in k4
        # M5: shape (num_k5, num_triples), entries 1 if triple in k5
        M4 = np.zeros((num_k4, num_triples), dtype=np.int8)
        for i, ts in enumerate(k4_incidence):
            for t in ts:
                M4[i, t] = 1
                
        M5 = np.zeros((num_k5, num_triples), dtype=np.int8)
        for i, ts in enumerate(k5_incidence):
            for t in ts:
                M5[i, t] = 1

        # Current counts
        # red_k4_count[i] = sum(M4[i] * (colors == 0))
        # blue_k5_count[i] = sum(M5[i] * (colors == 1))
        
        is_red = (colors == 0).astype(np.int8)
        is_blue = (colors == 1).astype(np.int8)
        
        red_k4_count = M4 @ is_red
        blue_k5_count = M5 @ is_blue
        
        # Bad sets
        bad_k4_mask = (red_k4_count == 4)
        bad_k5_mask = (blue_k5_count == 10)
        
        bad_k4_count = np.sum(bad_k4_mask)
        bad_k5_count = np.sum(bad_k5_mask)
        
        if bad_k4_count == 0 and bad_k5_count == 0:
            return colors.tolist()

        # Local Search
        # We will pick a bad constraint, pick a random variable in it, flip it.
        # Update counts using vectorized operations.
        
        # To speed up random selection from bad sets:
        # We can just use np.where(bad_k4_mask)[0]
        
        while time.time() < time_limit:
            if bad_k4_count == 0 and bad_k5_count == 0:
                return colors.tolist()
                
            # Decide which constraint to fix
            # Prioritize the one with more bad constraints? Or random?
            # Let's use a weighted random choice based on counts
            if bad_k4_count == 0:
                fix_type = 'k5'
            elif bad_k5_count == 0:
                fix_type = 'k4'
            else:
                # Random choice, maybe weighted
                if random.random() < 0.5:
                    fix_type = 'k4'
                else:
                    fix_type = 'k5'
                    
            if fix_type == 'k4':
                # Pick a random bad K4
                bad_indices = np.where(bad_k4_mask)[0]
                if len(bad_indices) == 0: continue
                k4_idx = bad_indices[random.randint(0, len(bad_indices)-1)]
                
                # Pick a random triple in this K4 to flip
                # We need the list of triples for this K4
                ts = k4_incidence[k4_idx]
                if not ts: continue
                t_idx = ts[random.randint(0, len(ts)-1)]
                
                # Flip color
                # If it was Red (0) -> Blue (1)
                # If it was Blue (1) -> Red (0)
                # For a bad K4 (all red), we MUST flip a Red one to Blue.
                # So t_idx must be Red.
                if colors[t_idx] != 0:
                    # If we picked a blue one in a red K4? That shouldn't happen if K4 is all red.
                    # But wait, bad K4 means all 4 are red. So any triple in it is red.
                    # So this check is redundant but safe.
                    continue
                
                # Update colors
                colors[t_idx] = 1
                is_red[t_idx] = 0
                is_blue[t_idx] = 1
                
                # Update counts
                # For all K4s containing t_idx:
                # red_k4_count[k4_i] -= 1
                affected_k4s = triple_to_k4[t_idx]
                for k4_i in affected_k4s:
                    old_cnt = red_k4_count[k4_i]
                    new_cnt = old_cnt - 1
                    red_k4_count[k4_i] = new_cnt
                    
                    was_bad = (old_cnt == 4)
                    is_bad = (new_cnt == 4)
                    if was_bad and not is_bad:
                        bad_k4_mask[k4_i] = False
                        bad_k4_count -= 1
                    elif not was_bad and is_bad:
                        bad_k4_mask[k4_i] = True
                        bad_k4_count += 1
                        
                # For all K5s containing t_idx:
                # blue_k5_count[k5_i] += 1
                affected_k5s = triple_to_k5[t_idx]
                for k5_i in affected_k5s:
                    old_cnt = blue_k5_count[k5_i]
                    new_cnt = old_cnt + 1
                    blue_k5_count[k5_i] = new_cnt
                    
                    was_bad = (old_cnt == 10)
                    is_bad = (new_cnt == 10)
                    if was_bad and not is_bad:
                        bad_k5_mask[k5_i] = False
                        bad_k5_count -= 1
                    elif not was_bad and is_bad:
                        bad_k5_mask[k5_i] = True
                        bad_k5_count += 1
                        
            else: # fix_type == 'k5'
                bad_indices = np.where(bad_k5_mask)[0]
                if len(bad_indices) == 0: continue
                k5_idx = bad_indices[random.randint(0, len(bad_indices)-1)]
                
                ts = k5_incidence[k5_idx]
                if not ts: continue
                t_idx = ts[random.randint(0, len(ts)-1)]
                
                # Bad K5 means all blue. Flip Blue to Red.
                if colors[t_idx] != 1:
                    continue
                
                colors[t_idx] = 0
                is_red[t_idx] = 1
                is_blue[t_idx] = 0
                
                # Update counts
                affected_k4s = triple_to_k4[t_idx]
                for k4_i in affected_k4s:
                    old_cnt = red_k4_count[k4_i]
                    new_cnt = old_cnt + 1
                    red_k4_count[k4_i] = new_cnt
                    
                    was_bad = (old_cnt == 4)
                    is_bad = (new_cnt == 4)
                    if was_bad and not is_bad:
                        bad_k4_mask[k4_i] = False
                        bad_k4_count -= 1
                    elif not was_bad and is_bad:
                        bad_k4_mask[k4_i] = True
                        bad_k4_count += 1
                        
                affected_k5s = triple_to_k5[t_idx]
                for k5_i in affected_k5s:
                    old_cnt = blue_k5_count[k5_i]
                    new_cnt = old_cnt - 1
                    blue_k5_count[k5_i] = new_cnt
                    
                    was_bad = (old_cnt == 10)
                    is_bad = (new_cnt == 10)
                    if was_bad and not is_bad:
                        bad_k5_mask[k5_i] = False
                        bad_k5_count -= 1
                    elif not was_bad and is_bad:
                        bad_k5_mask[k5_i] = True
                        bad_k5_count += 1
                        
        return None

    # Try to solve for n=34
    # We have ~110s.
    
    # If 34 fails, try 33, etc.
    
    for n in [34, 33, 32, 31, 30]:
        t_start = time.time()
        remaining = deadline - t_start
        if remaining < 5:
            break
            
        # Allocate time based on n
        # 34 gets most time
        if n == 34:
            t_limit = deadline - 10
        else:
            t_limit = deadline - 5
            
        colors = try_solve(n, t_limit)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: Use the seed solution for n=9 (known valid)
    # n=9, color 0 if sum%3==0 else 1.
    
    def check_validity(n, colors):
        triples = list(combinations(range(n), 3))
        t_idx = {t: i for i, t in enumerate(triples)}
        # Check Red K4
        for k4 in combinations(range(n), 4):
            cnt = 0
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        if colors[t_idx[t]] == 0:
                            cnt += 1
            if cnt == 4:
                return False
        # Check Blue K5
        for k5 in combinations(range(n), 5):
            cnt = 0
            for t in combinations(k5, 3):
                if colors[t_idx[t]] == 1:
                    cnt += 1
            if cnt == 10:
                return False
        return True

    n = 9
    seed_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    if check_validity(n, seed_colors):
        return {'n': n, 'colors': seed_colors}
        
    # If even seed fails (unlikely), return a trivial valid one
    # n=1
    return {'n': 1, 'colors': []}

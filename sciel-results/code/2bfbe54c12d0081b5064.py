import math
import time
import random

def compute_score(u):
    """Compute the score for a given set U."""
    if not u:
        return 0.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    # Sumset
    sumset = set()
    for a in u:
        for b in u:
            sumset.add(a + b)
    
    # Diffset
    diffset = set()
    for a in u:
        for b in u:
            diffset.add(a - b)
    
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def compute_diff_guided_score(u):
    """Compute a diff-guided score that prioritizes |U-U| growth over |U+U| growth."""
    if not u:
        return 0.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    sumset = set()
    for a in u:
        for b in u:
            sumset.add(a + b)
    
    diffset = set()
    for a in u:
        for b in u:
            diffset.add(a - b)
    
    r = len(diffset) / len(sumset)
    # Add a small bonus for larger |U-U| to guide search
    diff_bonus = math.log(len(diffset) + 1) * 0.001
    return 1 + math.log(r) / math.log(m) + diff_bonus

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known good Golomb rulers as starting points
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]
    
    best_set = None
    best_score = 0.0
    
    # Evaluate initial candidates
    for u in initial_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = u
    
    if best_set is None:
        return [0, 1, 3]
    
    # Local search with diff-guided strategy
    current_set = list(best_set)
    current_score = best_score
    
    # Track current diffset and sumset sizes for guided search
    def get_sets(u):
        s = set()
        d = set()
        for a in u:
            for b in u:
                s.add(a + b)
                d.add(a - b)
        return s, d
    
    cur_sumset, cur_diffset = get_sets(current_set)
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        choice = random.random()
        
        if choice < 0.5 and len(current_set) < 30:
            # Try adding a new element - prioritize positions that add new differences
            min_pos = 0
            max_pos = max(current_set) * 2 + 10 if current_set else 10
            
            best_candidate = None
            best_candidate_score = current_score + 1e-10
            best_candidate_diff_gain = -1
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                new_val = random.randint(min_pos, max_pos)
                if new_val in current_set:
                    continue
                
                new_set = sorted(current_set + [new_val])
                score = compute_score(new_set)
                
                # Compute diff gain
                new_diffset = set()
                new_sumset = set()
                for a in new_set:
                    for b in new_set:
                        new_diffset.add(a - b)
                        new_sumset.add(a + b)
                
                diff_gain = len(new_diffset) - len(cur_diffset)
                sum_gain = len(new_sumset) - len(cur_sumset)
                
                # Prefer moves that add more differences than sums
                guided_score = score + 0.001 * (diff_gain - sum_gain)
                
                if guided_score > best_candidate_score:
                    best_candidate_score = guided_score
                    best_candidate = new_set
                    best_candidate_diff_gain = diff_gain
            
            if best_candidate is not None:
                current_set = best_candidate
                current_score = compute_score(current_set)
                cur_sumset, cur_diffset = get_sets(current_set)
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
        
        elif choice < 0.8 and len(current_set) > 3:
            # Try removing an element - prefer removing elements that don't contribute many unique differences
            removable = [x for x in current_set if x != 0]
            if removable:
                idx = random.choice(removable)
                new_set = [x for x in current_set if x != idx]
                score = compute_score(new_set)
                
                new_diffset = set()
                new_sumset = set()
                for a in new_set:
                    for b in new_set:
                        new_diffset.add(a - b)
                        new_sumset.add(a + b)
                
                diff_loss = len(cur_diffset) - len(new_diffset)
                sum_loss = len(cur_sumset) - len(new_sumset)
                
                # Prefer removals that lose fewer differences than sums
                guided_score = score + 0.001 * (sum_loss - diff_loss)
                
                if guided_score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    cur_sumset, cur_diffset = get_sets(current_set)
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
        
        else:
            # Try modifying an element - prioritize moves that increase |U-U|
            modifiable = [x for x in current_set if x != 0]
            if modifiable:
                idx = random.choice(modifiable)
                best_mod = None
                best_mod_score = current_score + 1e-10
                
                for delta in [-3, -2, -1, 1, 2, 3, 5]:
                    if time.time() > deadline:
                        break
                    new_val = idx + delta
                    if new_val < 0 or new_val in current_set:
                        continue
                    
                    new_set = sorted([x if x != idx else new_val for x in current_set])
                    score = compute_score(new_set)
                    
                    new_diffset = set()
                    new_sumset = set()
                    for a in new_set:
                        for b in new_set:
                            new_diffset.add(a - b)
                            new_sumset.add(a + b)
                    
                    diff_gain = len(new_diffset) - len(cur_diffset)
                    sum_gain = len(new_sumset) - len(cur_sumset)
                    
                    guided_score = score + 0.001 * (diff_gain - sum_gain)
                    
                    if guided_score > best_mod_score:
                        best_mod_score = guided_score
                        best_mod = new_set
                
                if best_mod is not None:
                    current_set = best_mod
                    current_score = compute_score(current_set)
                    cur_sumset, cur_diffset = get_sets(current_set)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
    
    # Final polish: try small adjustments to the best set found
    if best_set and time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            modifiable = [x for x in best_set if x != 0]
            if not modifiable:
                break
            idx = random.choice(modifiable)
            for delta in [-1, 1, -2, 2]:
                if time.time() > deadline:
                    break
                new_val = idx + delta
                if new_val < 0 or new_val in best_set:
                    continue
                
                new_set = sorted([x if x != idx else new_val for x in best_set])
                score = compute_score(new_set)
                if score > best_score + 1e-10:
                    best_score = score
                    best_set = new_set
                    break
    
    return best_set

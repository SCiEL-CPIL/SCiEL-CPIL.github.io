import time

def score_set(S):
    S_set = set(S)
    good_d = set()
    for d in range(1, 512):
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                good_d.add(d)
                break
    return len(good_d)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_S = []
    best_score = 0
    
    def update_best(S):
        nonlocal best_S, best_score
        s = score_set(S)
        if s > best_score:
            best_score = s
            best_S = S.copy()
            
    def is_satisfied(S, d):
        S_set = set(S)
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                return True
        return False

    def backtrack(S, next_d):
        nonlocal best_S, best_score
        nonlocal start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Update best
        update_best(S)
        
        # Check if next_d is already satisfied
        if is_satisfied(S, next_d):
            backtrack(S, next_d + 1)
            return
            
        # If not satisfied, try to add elements
        if len(S) == 32:
            return
            
        S_set = set(S)
        cand_set = set()
        for x in S:
            cand_set.add(x)
            cand_set.add(x - next_d)
            cand_set.add(x - 2 * next_d)
            
        for i in range(1, 20):
            cand_set.add(i)
            
        valid_cands = []
        max_a = 512 - 2 * next_d
        
        for a in cand_set:
            if a < 1 or a > max_a:
                continue
                
            triple = {a, a + next_d, a + 2 * next_d}
            new_elems = triple - S_set
            
            if len(S) + len(new_elems) > 32:
                continue
                
            score_key = (len(new_elems), sum(new_elems))
            valid_cands.append((score_key, a, new_elems))
            
        valid_cands.sort(key=lambda x: x[0])
        
        for _, a, new_elems in valid_cands[:5]:
            new_S = S.copy()
            new_S.update(new_elems)
            backtrack(new_S, next_d + 1)
            
            if time.time() - start_time > time_limit:
                return

    backtrack(set(), 1)
    
    if not best_S:
        best_S = list(range(1, 33))
        
    return best_S

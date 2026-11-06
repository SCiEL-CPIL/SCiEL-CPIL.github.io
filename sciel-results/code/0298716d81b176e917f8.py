import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, chosen, counts):
        """Check if x can be added to chosen given current sum counts."""
        if counts.get(2*x, 0) >= 2:
            return False
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def run_greedy(noise=0.0, time_limit=None):
        """
        Greedy construction.
        Picks the element that maximizes slack (sums with room to spare).
        """
        chosen = []
        counts = defaultdict(int)
        candidates = set(range(1, N + 1))
        
        if time_limit is None:
            time_limit = deadline
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('-inf')
            
            for x in candidates:
                # Fast validity check and score computation
                valid = True
                slack = 0
                
                c2x = counts.get(2*x, 0)
                if c2x >= 2:
                    continue
                if c2x == 0:
                    slack += 1
                else:  # c2x == 1
                    # This sum will be at capacity after adding x
                    pass
                    
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 0:
                        slack += 1
                    # else c == 1, no slack gained
                    
                if not valid:
                    continue
                
                score = slack + (random.random() * noise if noise > 0 else 0)
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add best_x
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
            
        return chosen

    def improve_set(initial_set, time_limit):
        """
        Try to improve the set by:
        1. Adding elements that fit
        2. Swapping elements to create room for additions
        Uses efficient conflict tracking.
        """
        chosen = list(initial_set)
        n = len(chosen)
        
        # Build counts
        counts = defaultdict(int)
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(n):
            for j in range(i+1, n):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
            
        if not candidates:
            return chosen
            
        # Track which candidates are blocked and why
        # For each candidate, track the number of "conflicts" (sums that would exceed limit)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            # Pass 1: Try to add any valid candidate
            for x in cand_list:
                if time.time() > time_limit:
                    break
                if is_valid_addition(x, chosen, counts):
                    # Add it
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    improved = True
                    break  # Restart to re-evaluate all candidates
            
            if improved:
                continue
                
            # Pass 2: Try swaps to enable future additions
            # For each candidate x, find if there's a y in chosen such that
            # removing y and adding x is valid
            swap_done = False
            random.shuffle(cand_list)
            for x in cand_list[:50]:  # Limit swaps per round for speed
                if time.time() > time_limit:
                    break
                    
                # Quick check: how many conflicts does x have?
                conflicts = 0
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        conflicts += 1
                if counts.get(2*x, 0) >= 2:
                    conflicts += 1
                    
                if conflicts == 0:
                    continue  # Already handled in Pass 1
                    
                # Try to find a y to remove that resolves all conflicts
                # For each conflicting y, check if removing y makes x valid
                chosen_list = list(chosen)
                random.shuffle(chosen_list)
                
                for y in chosen_list:
                    # Check if removing y resolves all conflicts for x
                    # We need to verify that after removing y, x has no conflicts
                    valid_after_swap = True
                    for z in chosen:
                        if z == y:
                            continue
                        s = x + z
                        # Count of s in (chosen - {y})
                        c = counts.get(s, 0)
                        # Subtract contribution from y+z if y != z
                        if y != z:
                            c -= 1
                        if c >= 2:
                            valid_after_swap = False
                            break
                    # Check 2x
                    c2x = counts.get(2*x, 0)
                    # 2x is only formed by x+x, which isn't in chosen-y since x not in chosen
                    # But wait, 2x could be formed by other pairs? No, 2x = a+b where a,b in chosen
                    # and a+b=2x. Since x not in chosen, this is about pairs in chosen summing to 2x
                    # That count doesn't change when we remove y (unless y is involved, but y+y=2y not 2x)
                    # Actually, removing y doesn't affect count of 2x unless y+y=2x, i.e., y=x, impossible
                    if c2x >= 2:
                        valid_after_swap = False
                    
                    if valid_after_swap:
                        # Perform the swap
                        # Remove y
                        counts[2*y] -= 1
                        if counts[2*y] == 0:
                            del counts[2*y]
                        for z in chosen:
                            if z != y:
                                s = y + z
                                counts[s] -= 1
                                if counts[s] == 0:
                                    del counts[s]
                        
                        # Add x
                        counts[2*x] = counts.get(2*x, 0) + 1
                        for z in chosen:
                            if z != y:
                                s = x + z
                                counts[s] = counts.get(s, 0) + 1
                                
                        chosen.remove(y)
                        chosen.append(x)
                        candidates.discard(x)
                        candidates.add(y)
                        
                        improved = True
                        swap_done = True
                        break
                        
                if swap_done:
                    break
                    
        return chosen

    # Main loop
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Vary noise to explore different greedy paths
        noise = random.uniform(0, 3.0)
        baseline = run_greedy(noise=noise, time_limit=min(deadline, time.time() + 8))
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
            
        # Refine with improvement
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Allocate time for improvement
        imp_time = min(remaining * 0.7, 15.0)
        refined = improve_set(best_set, time.time() + imp_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
            
        # If we've found a good set, do a final polish with more restarts
        if best_size >= 380 and remaining > 5:
            # Try a few more quick restarts with lower noise
            for _ in range(3):
                if time.time() > deadline - 2:
                    break
                baseline2 = run_greedy(noise=random.uniform(0, 1.0), 
                                       time_limit=min(deadline, time.time() + 5))
                if len(baseline2) > best_size:
                    best_size = len(baseline2)
                    best_set = baseline2
                    refined2 = improve_set(best_set, time.time() + 5)
                    if len(refined2) > best_size:
                        best_size = len(refined2)
                        best_set = refined2

    return best_set

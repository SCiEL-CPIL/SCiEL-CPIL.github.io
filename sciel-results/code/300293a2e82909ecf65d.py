import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        diffs = set()
        for a in s:
            # a + 2d <= 512 => d <= (512 - a) // 2
            max_d = (512 - a) // 2
            for d in range(1, max_d + 1):
                if a + d in s_set and a + 2 * d in s_set:
                    diffs.add(d)
        return len(diffs)
    
    def local_search(s, max_passes=3):
        """Full local search: try swapping each element with every other possible value."""
        s = list(s)
        s_set = set(s)
        current_score = score_set(s)
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            for i in range(len(s)):
                if time.time() > deadline:
                    break
                old_val = s[i]
                # Try all other values 1..512 not in s_set
                for new_val in range(1, 513):
                    if new_val in s_set:
                        continue
                    
                    # Perform swap
                    s[i] = new_val
                    s_set.remove(old_val)
                    s_set.add(new_val)
                    
                    new_score = score_set(s)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        # Keep the change
                    else:
                        # Revert
                        s[i] = old_val
                        s_set.remove(new_val)
                        s_set.add(old_val)
                        
                    if time.time() > deadline:
                        return s, current_score
        return s, current_score

    def simulated_annealing(s, max_iters=10000):
        """Simulated annealing to optimize the set."""
        s = list(s)
        s_set = set(s)
        current_score = score_set(s)
        best_score = current_score
        best_s = list(s)
        
        temp = 5.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: swap one element
            idx = random.randint(0, len(s) - 1)
            old_val = s[idx]
            new_val = random.randint(1, 512)
            
            if new_val in s_set:
                continue
            
            s[idx] = new_val
            s_set.remove(old_val)
            s_set.add(new_val)
            
            new_score = score_set(s)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_s = list(s)
            else:
                # Revert
                s[idx] = old_val
                s_set.remove(new_val)
                s_set.add(old_val)
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best_s, best_score

    # Generate diverse initial solutions
    candidates = []
    
    # 1. Consecutive range
    candidates.append(list(range(1, 33)))
    
    # 2. Random set
    candidates.append(random.sample(range(1, 513), 32))
    
    # 3. Another random set
    candidates.append(random.sample(range(1, 513), 32))
    
    best_overall = None
    best_overall_score = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # 1. Local Search (Strong)
        polished, score = local_search(cand, max_passes=3)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = polished
        
        # 2. Simulated Annealing (if time permits)
        if time.time() < deadline - 20:
            sa_result, sa_score = simulated_annealing(polished, max_iters=5000)
            if sa_score > best_overall_score:
                best_overall_score = sa_score
                best_overall = sa_result
        
        # 3. Final Local Search on the best so far
        if time.time() < deadline - 5:
            final, final_score = local_search(best_overall, max_passes=3)
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall = final

    if best_overall is None:
        best_overall = list(range(1, 33))
    
    return best_overall

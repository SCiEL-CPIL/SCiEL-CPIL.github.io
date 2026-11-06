import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        diffs = set()
        # Optimization: Only check pairs (a, b) where a < b
        # If a, a+d, a+2d are in S, then d = (b-a) if b=a+d, and a+2d is in S.
        # Alternatively, iterate all a, b in S with a < b. Let d = b - a.
        # Check if a + 2*d is in S.
        # This is O(N^2). N=32, so 1024 checks. Very fast.
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                a = s[i]
                b = s[j]
                d = b - a
                c = b + d
                if c in s_set:
                    diffs.add(d)
        return len(diffs)

    def greedy_construct():
        """
        Greedily build a set of size 32 by adding the element that maximizes the score.
        """
        s = []
        s_set = set()
        current_score = 0
        
        for _ in range(32):
            best_val = None
            best_score = current_score - 1 # Ensure we pick something if score doesn't improve? 
            # Actually, we just want to maximize the final score. 
            # Sometimes adding an element doesn't increase the score immediately but helps later.
            # However, pure greedy on score might get stuck. 
            # Let's add a bit of randomness: pick from top K candidates.
            
            candidates = []
            for v in range(1, 513):
                if v in s_set:
                    continue
                # Try adding v
                s_set.add(v)
                new_s = s + [v]
                new_score = score_set(new_s)
                s_set.remove(v)
                
                candidates.append((new_score, v))
            
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick randomly from the top 5 to introduce diversity
            top_k = min(5, len(candidates))
            _, best_val = random.choice(candidates[:top_k])
            
            s.append(best_val)
            s_set.add(best_val)
            current_score = score_set(s)
            
        return s

    def local_search(s, max_passes=20):
        """
        Swap-based local search.
        """
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
                # Try swapping s[i] with every other value 1..512
                # To speed up, we can just iterate 1..512
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

    best_overall = None
    best_overall_score = 0
    
    # Run multiple greedy constructions
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy construction
        s = greedy_construct()
        
        # Local search polish
        polished, score = local_search(s, max_passes=10)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = polished
            
        # Optional: If we have time, do a second round of local search on the best found so far
        if r == num_restarts - 1 or (r % 5 == 0 and time.time() < deadline - 30):
             # Refine the current best
             refined, refined_score = local_search(best_overall, max_passes=15)
             if refined_score > best_overall_score:
                 best_overall_score = refined_score
                 best_overall = refined

    if best_overall is None:
        # Fallback
        best_overall = list(range(1, 33))
        best_overall_score = score_set(best_overall)
        
    return best_overall

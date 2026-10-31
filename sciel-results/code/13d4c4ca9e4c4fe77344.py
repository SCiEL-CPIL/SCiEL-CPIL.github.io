import numpy as np
import random
from collections import defaultdict

N = 50

def construct():
    best_S = []
    best_score = 0.0
    
    # We want to maximize |S| / max_r_k
    # Greedy strategy: Build S such that max_r_k stays small.
    # We will run multiple random restarts with different initial seeds/orderings.
    
    for restart in range(500):
        S = []
        sum_counts = defaultdict(int)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        # We can force the first element to be 0 to simplify sums?
        # Not necessarily, but 0 is a good anchor.
        # Let's just let the greedy pick.
        
        current_max_count = 0
        
        # Greedy loop
        # We try to add elements that keep the max count low.
        # If we allow max count up to K, we can potentially get larger S.
        # Let's try to keep max count as low as possible at each step.
        
        while True:
            best_candidate = None
            best_new_max = float('inf')
            best_tie_breaker = -1
            
            for i in candidates:
                # Calculate potential new max count if we add i
                # New sums: i+i, and i+j for j in S
                
                # Check self sum
                new_self_count = sum_counts[i + i] + 1
                
                # Check cross sums
                max_cross_new = 0
                for j in S:
                    cnt = sum_counts[i + j] + 1
                    if cnt > max_cross_new:
                        max_cross_new = cnt
                
                # The new max count for this addition would be:
                # max(current_max_count, new_self_count, max_cross_new)
                potential_max = max(current_max_count, new_self_count, max_cross_new)
                
                # We want to minimize potential_max.
                # If tie, we might prefer smaller i or random.
                # To encourage diversity, we can add a small random noise or prefer larger sets?
                # Actually, minimizing the peak is the primary goal.
                
                if potential_max < best_new_max:
                    best_new_max = potential_max
                    best_candidate = i
                    best_tie_breaker = random.random()
                elif potential_max == best_new_max:
                    # Tie breaking: prefer random choice among best candidates
                    if random.random() < 0.5:
                        best_candidate = i
                        best_tie_breaker = random.random()

            if best_candidate is None:
                break
            
            # Add the best candidate
            S.append(best_candidate)
            
            # Update sum_counts
            sum_counts[best_candidate + best_candidate] += 1
            for j in S[:-1]: # S[:-1] is the old S
                sum_counts[best_candidate + j] += 1
            
            # Update current_max_count
            # Instead of scanning all sums, we can track it, but scanning is fine for N=50
            # max(sum_counts.values()) is O(N^2) which is 2500, very fast.
            if sum_counts:
                current_max_count = max(sum_counts.values())
            
            # Remove from candidates
            candidates.remove(best_candidate)
            
            # Optional: Stop if max count becomes too high relative to size?
            # Score = |S| / max_count.
            # If max_count grows too fast, score drops.
            # But we don't know the final size.
            # Let's just run until candidates are empty or max_count > 10 (arbitrary cap)
            if current_max_count > 10:
                break

        # Evaluate this set S
        if not S:
            continue
            
        max_r_k = max(sum_counts.values()) if sum_counts else 1
        score = len(S) / max_r_k
        
        if score > best_score:
            best_score = score
            best_S = S[:]
            
    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)

import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # 1. Initialize with the longest chain (length 13)
    # Chain: 0, 1, 3, 7, ..., (1<<12)-1
    family = []
    cur = 0
    family.append(cur)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    fam_set = set(family)
    
    # scores[s] = number of sets in `family` that are comparable to s
    # Comparability: A subset B or B subset A
    scores = [0] * total_masks
    
    def update_scores_for_addition(mask, delta):
        # Update scores for all masks by adding delta if comparable to mask
        # This is used when we add or remove a mask from the family
        for s in range(total_masks):
            if (s & ~mask) == 0 or (mask & ~s) == 0:
                scores[s] += delta

    # Initial scores
    for f in family:
        update_scores_for_addition(f, 1)
        
    # Note: For s in family, scores[s] includes s itself?
    # My update_scores_for_addition adds 1 to scores[s] if s is comparable to mask.
    # Since s is comparable to s, scores[s] will be at least 1 for s in family.
    # Specifically, if s in family, scores[s] = (number of comparable sets in family) including s.
    # So comparable_others[s] = scores[s] - 1.

    # Greedy addition
    while len(family) < m and time.time() < deadline:
        # Find best candidate not in family
        best_score = -1
        best_mask = -1
        
        for s in range(total_masks):
            if s in fam_set:
                continue
            if scores[s] > best_score:
                best_score = scores[s]
                best_mask = s
                
        if best_mask == -1:
            break
            
        # Add to family
        family.append(best_mask)
        fam_set.add(best_mask)
        
        # Update scores
        update_scores_for_addition(best_mask, 1)
                
    # Local Search (Hill Climbing)
    # We want to maximize total comparable pairs.
    # Total pairs = sum_{s in family} (scores[s] - 1) / 2
    
    improved = True
    # We can use random restarts or just iterate until no improvement.
    # Given the small size, a single long hill climb is likely sufficient, 
    # but we can add some randomization if stuck.
    
    # Precompute comparability for all pairs? 4096x4096 bits = 2MB. 
    # Or just check on the fly. On the fly is (s & ~f) == 0 or ... which is fast.
    
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps
        # To avoid bias, we can shuffle the order of family members or candidates.
        # But sequential is fine for now.
        
        # We iterate over all A in family.
        # For each A, we find the best B not in family to swap with.
        # If best B improves the score, we swap.
        
        # Optimization: 
        # Current contribution of A to total score (excluding self) is scores[A] - 1.
        # If we remove A, the new score for any B not in family is:
        # new_score[B] = scores[B] - (1 if B comparable to A else 0)
        # The gain of swapping A with B is:
        # gain = new_score[B] - (scores[A] - 1)
        # If gain > 0, swap.
        
        for i in range(len(family)):
            if time.time() > deadline:
                break
            A = family[i]
            score_A_others = scores[A] - 1
            
            best_gain = 0
            best_B = -1
            
            for B in range(total_masks):
                if B in fam_set:
                    continue
                # Check if B is comparable to A
                comp = 1 if ((B & ~A) == 0 or (A & ~B) == 0) else 0
                new_score_B = scores[B] - comp
                gain = new_score_B - score_A_others
                
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
                    
            if best_B != -1 and best_gain > 0:
                # Perform swap
                fam_set.remove(A)
                fam_set.add(best_B)
                family[i] = best_B
                
                # Update scores
                # Remove A's contribution
                update_scores_for_addition(A, -1)
                # Add B's contribution
                update_scores_for_addition(best_B, 1)
                
                improved = True
                # Continue scanning or restart? 
                # Continuing is fine, as scores are updated.
                # However, the "best_B" found was based on the state before swap.
                # The next iterations will use updated scores.
                
        # If no improvement in a full pass, we might be in a local optimum.
        # We can try random swaps (simulated annealing style) if time permits,
        # but simple hill climbing often works well for this size.
        
    return _convert(family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result

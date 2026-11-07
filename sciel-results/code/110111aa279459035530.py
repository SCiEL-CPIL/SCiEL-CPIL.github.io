import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Multi-chain initialization: Create multiple chains to allow cross-comparability
    family = []
    fam_set = set()
    
    # Start with empty set
    family.append(0)
    fam_set.add(0)
    
    # Create multiple chains by greedily adding sets that maximize comparability
    # scores[s] = number of sets in family comparable to s
    scores = [0] * total_masks
    
    def update_scores_for_addition(mask, delta):
        for s in range(total_masks):
            if (s & ~mask) == 0 or (mask & ~s) == 0:
                scores[s] += delta
    
    update_scores_for_addition(0, 1)
    
    # Greedily add sets to form multiple chains
    while len(family) < m and time.time() < deadline:
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
            
        family.append(best_mask)
        fam_set.add(best_mask)
        update_scores_for_addition(best_mask, 1)
                
    # Local Search with simulated annealing
    def get_total_score():
        return sum(scores[s] - 1 for s in family) // 2
    
    current_score = get_total_score()
    best_family = family[:]
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    cooling = 0.95
    
    while time.time() < deadline and T > 0.01:
        improved = False
        
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
                comp = 1 if ((B & ~A) == 0 or (A & ~B) == 0) else 0
                new_score_B = scores[B] - comp
                gain = new_score_B - score_A_others
                
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
                    
            if best_B != -1:
                # Accept if gain > 0 or with probability exp(gain/T)
                accept = False
                if best_gain > 0:
                    accept = True
                else:
                    import math
                    if random.random() < math.exp(best_gain / T):
                        accept = True
                
                if accept:
                    fam_set.remove(A)
                    fam_set.add(best_B)
                    family[i] = best_B
                    
                    update_scores_for_addition(A, -1)
                    update_scores_for_addition(best_B, 1)
                    
                    new_score = get_total_score()
                    if new_score > best_score:
                        best_score = new_score
                        best_family = family[:]
                    
                    improved = True
        
        T *= cooling
    
    # Restore best family
    family = best_family
    
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

import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Precompute comparability: comp[a][b] = True if a is subset of b or vice versa
    # For memory efficiency, use a list of sets for each mask
    # Actually, let's precompute for each mask which other masks are comparable
    # This is expensive upfront but saves time in search
    
    # Better approach: precompute subset relationships using bit manipulation
    # For each mask, store list of masks it's comparable to
    # But 4096 * 4096 is too much memory
    
    # Instead, let's optimize the score update by tracking which masks are in family
    # and computing comparability on the fly but more efficiently
    
    # Dense initialization: Start with Boolean lattice B_5 (all subsets of {0,1,2,3,4})
    # That's 32 sets, then add 8 more greedily
    
    family = []
    fam_set = set()
    
    # Add all subsets of {0,1,2,3,4} - this is B_5 with 32 elements
    for mask in range(1 << 5):
        family.append(mask)
        fam_set.add(mask)
    
    # scores[s] = number of sets in family comparable to s
    scores = [0] * total_masks
    
    # Initialize scores: for each set in family, count how many are comparable
    for a in family:
        for b in family:
            if a != b and ((a & ~b) == 0 or (b & ~a) == 0):
                scores[a] += 1
    
    def update_scores_for_addition(mask, delta):
        """Update scores when adding/removing a mask from family"""
        for s in range(total_masks):
            if (s & ~mask) == 0 or (mask & ~s) == 0:
                scores[s] += delta
    
    # Greedily add 8 more sets to maximize comparability
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
    
    # Simulated annealing with improved acceptance
    T = 10.0
    cooling = 0.97  # Slower cooling for more exploration
    
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
    
    # Final greedy polish pass: try to improve by swapping
    for _ in range(50):
        if time.time() > deadline:
            break
        improved_any = False
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
            
            if best_B != -1 and best_gain > 0:
                fam_set.remove(A)
                fam_set.add(best_B)
                family[i] = best_B
                
                update_scores_for_addition(A, -1)
                update_scores_for_addition(best_B, 1)
                
                new_score = get_total_score()
                if new_score > best_score:
                    best_score = new_score
                    best_family = family[:]
                
                improved_any = True
        
        if not improved_any:
            break
    
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

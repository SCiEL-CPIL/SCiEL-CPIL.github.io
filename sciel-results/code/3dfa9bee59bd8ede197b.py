import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55

    # Multi-chain initialization
    family = []
    fam_set = set()
    
    family.append(0)
    fam_set.add(0)
    
    scores = [0] * total_masks
    
    def update_scores_for_addition(mask, delta):
        for s in range(total_masks):
            if (s & ~mask) == 0 or (mask & ~s) == 0:
                scores[s] += delta

    update_scores_for_addition(0, 1)
    
    # Greedy initialization
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
        
    def get_total_score():
        # Sum of scores / 2 - but scores include self? No, scores[s] is count of OTHERS comparable to s.
        # Wait, update_scores_for_addition(mask, 1) adds 1 to scores[s] if s comparable to mask.
        # When we add mask, we call update for mask.
        # Does scores[mask] include mask itself?
        # In update_scores_for_addition, we iterate s. If s=mask, (mask & ~mask)==0 is True. So scores[mask] is incremented.
        # So scores[mask] includes itself.
        # So sum(scores[s] for s in family) counts each pair twice, plus 40 self-pairs.
        # Total comparable pairs = (sum(scores[s]) - m) / 2.
        return (sum(scores[s] for s in family) - m) // 2

    current_score = get_total_score()
    best_family = family[:]
    best_score_val = current_score
    
    # Simulated Annealing with Random Moves
    T = 50.0
    cooling = 0.99
    min_T = 0.01
    
    # Precompute comparability for all pairs? No, too big.
    # Just use bitwise checks.
    
    def is_comparable(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0

    while time.time() < deadline and T > min_T:
        # Pick a random move type: 1-swap or 2-swap
        if random.random() < 0.5:
            # 1-swap
            i = random.randint(0, m - 1)
            A = family[i]
            # Pick random B not in family
            while True:
                B = random.randint(0, total_masks - 1)
                if B not in fam_set:
                    break
            
            # Calculate gain
            # Remove A, Add B
            # Gain = (comp(B, family\{A}) + 0) - (comp(A, family\{A}) + 0)
            # comp(X, family\{A}) = scores[X] - 1 (self) - comp(X, A)
            # Wait, scores[X] includes X itself.
            # scores[A] includes A itself and all comparable in family.
            # So comp(A, family\{A}) = scores[A] - 1.
            # comp(B, family\{A}) = scores[B] - 1 - (1 if is_comparable(A, B) else 0).
            # Actually, if B is not in family, scores[B] does NOT include B itself?
            # No, scores is maintained for ALL masks.
            # When we added A, we updated scores[B] if comparable.
            # So scores[B] is the count of sets in family comparable to B.
            # It does NOT include B itself, because B is not in family.
            # So comp(B, family) = scores[B].
            # comp(B, family\{A}) = scores[B] - (1 if is_comparable(A, B) else 0).
            # comp(A, family\{A}) = scores[A] - 1.
            
            comp_AB = 1 if is_comparable(A, B) else 0
            gain = (scores[B] - comp_AB) - (scores[A] - 1)
            
            if gain > 0 or random.random() < math.exp(gain / T):
                # Accept
                fam_set.remove(A)
                fam_set.add(B)
                family[i] = B
                
                update_scores_for_addition(A, -1)
                update_scores_for_addition(B, 1)
                
                new_score = get_total_score()
                if new_score > best_score_val:
                    best_score_val = new_score
                    best_family = family[:]
        else:
            # 2-swap
            # Pick i, j distinct
            i, j = random.sample(range(m), 2)
            A, B = family[i], family[j]
            
            # Pick C, D distinct, not in family
            while True:
                C = random.randint(0, total_masks - 1)
                D = random.randint(0, total_masks - 1)
                if C not in fam_set and D not in fam_set and C != D:
                    break
            
            # Gain calculation
            # Old: comp(A, R) + comp(B, R) + comp(A, B)
            # R = family \ {A, B}
            # comp(A, R) = scores[A] - 1 - comp(A, B)
            # comp(B, R) = scores[B] - 1 - comp(A, B)
            # Total Old = 2*scores[A] - 2 - 2*comp(A, B) + comp(A, B) = 2*scores[A] - 2 - comp(A, B)
            # Wait, scores[A] is count of comparable in family.
            # So comp(A, family) = scores[A].
            # comp(A, R) = scores[A] - 1 - comp(A, B).
            # Similarly for B.
            # Old Contribution = (scores[A] - 1 - comp_AB) + (scores[B] - 1 - comp_AB) + comp_AB
            #                  = scores[A] + scores[B] - 2 - comp_AB.
            
            # New: comp(C, R) + comp(D, R) + comp(C, D)
            # comp(C, R) = scores[C] - comp(C, A) - comp(C, B)
            # comp(D, R) = scores[D] - comp(D, A) - comp(D, B)
            # New Contribution = scores[C] + scores[D] - comp(C,A) - comp(C,B) - comp(D,A) - comp(D,B) + comp(C,D)
            
            comp_AB = 1 if is_comparable(A, B) else 0
            comp_CA = 1 if is_comparable(C, A) else 0
            comp_CB = 1 if is_comparable(C, B) else 0
            comp_DA = 1 if is_comparable(D, A) else 0
            comp_DB = 1 if is_comparable(D, B) else 0
            comp_CD = 1 if is_comparable(C, D) else 0
            
            old_val = scores[A] + scores[B] - 2 - comp_AB
            new_val = scores[C] + scores[D] - comp_CA - comp_CB - comp_DA - comp_DB + comp_CD
            
            gain = new_val - old_val
            
            if gain > 0 or random.random() < math.exp(gain / T):
                # Accept
                fam_set.remove(A)
                fam_set.remove(B)
                fam_set.add(C)
                fam_set.add(D)
                family[i] = C
                family[j] = D
                
                update_scores_for_addition(A, -1)
                update_scores_for_addition(B, -1)
                update_scores_for_addition(C, 1)
                update_scores_for_addition(D, 1)
                
                new_score = get_total_score()
                if new_score > best_score_val:
                    best_score_val = new_score
                    best_family = family[:]
        
        T *= cooling
        
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

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def solve_n(n):
        """
        Attempts to find a valid coloring for n vertices.
        Returns list of colors (0 for red, 1 for blue) for all triples in lexicographic order.
        Returns None if not found within time limit.
        """
        if n < 1:
            return None
        
        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Precompute incidence lists
        # For each triple, which K4s and K5s does it belong to?
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        # For each K4, list of triple indices
        k4_triples = []
        for k4 in k4s:
            t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k4s[idx].append(len(k4_triples) - 1)
                
        # For each K5, list of triple indices
        k5_triples = []
        for k5 in k5s:
            t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k5s[idx].append(len(k5_triples) - 1)
                
        num_k4s = len(k4s)
        num_k5s = len(k5s)
        
        # Local Search with Restarts
        max_restarts = 20
        restart_time_limit = 4.0  # seconds per restart
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Initialize red counts
            rc_k4 = [0] * num_k4s
            rc_k5 = [0] * num_k5s
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for k4_idx in triple_to_k4s[i]:
                        rc_k4[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        rc_k5[k5_idx] += 1
                        
            # Count initial violations
            violations = 0
            for r in rc_k4:
                if r == 4:
                    violations += 1
            for r in rc_k5:
                if r == 0:
                    violations += 1
                    
            if violations == 0:
                return colors
                
            # Simulated Annealing parameters
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            current_time = time.time()
            steps = 0
            max_steps = 50000
            
            while steps < max_steps and time.time() < current_time + restart_time_limit:
                if time.time() > deadline - 5:
                    break
                    
                # Pick random triple
                t = random.randint(0, num_triples - 1)
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    if old_color == 0: # Red -> Blue
                        # rc_k4 decreases by 1
                        if rc_k4[k4_idx] == 4:
                            delta -= 1 # Violation removed
                    else: # Blue -> Red
                        # rc_k4 increases by 1
                        if rc_k4[k4_idx] == 3:
                            delta += 1 # Violation added
                
                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    if old_color == 0: # Red -> Blue
                        # rc_k5 decreases by 1
                        if rc_k5[k5_idx] == 0:
                            delta -= 1 # Violation removed (was all blue, now not)
                        elif rc_k5[k5_idx] == 1:
                            delta += 1 # Violation added (was 1 red, now 0 red -> all blue)
                    else: # Blue -> Red
                        # rc_k5 increases by 1
                        if rc_k5[k5_idx] == 0:
                            delta -= 1 # Violation removed
                        elif rc_k5[k5_idx] == 1:
                            # No change in violation status? 
                            # Was 1 red (not all blue), becomes 2 red (not all blue).
                            # Wait.
                            # Bad if rc_k5 == 0.
                            # If rc_k5 was 0, it was bad. Becomes 1. Not bad. Delta -1.
                            # If rc_k5 was 1, it was not bad. Becomes 2. Not bad. Delta 0.
                            pass 
                            # Actually, if rc_k5 was 1, it means 1 red, 9 blue. Not all blue.
                            # If rc_k5 becomes 2, 2 red, 8 blue. Not all blue.
                            # So no change.
                        # What if rc_k5 was something else? No change.
                        
                        # Correction on Blue -> Red for K5:
                        # rc_k5 increases by 1.
                        # If rc_k5 was 0 (bad), becomes 1 (not bad). Delta -1.
                        # If rc_k5 was > 0 (not bad), becomes > 1 (not bad). Delta 0.
                        # So only check if rc_k5[k5_idx] == 0.
                        
                        # Let's re-verify Red -> Blue for K5:
                        # rc_k5 decreases by 1.
                        # If rc_k5 was 0 (bad), becomes -1 (impossible). 
                        # Wait, if rc_k5 was 0, it means 0 red, 10 blue. All blue. Bad.
                        # Flipping one blue to red: rc_k5 becomes 1. 1 red, 9 blue. Not all blue. Not bad.
                        # So Delta -1.
                        # If rc_k5 was 1 (1 red, 9 blue). Not bad.
                        # Flipping one blue to red: rc_k5 becomes 2. Not bad.
                        # So Delta 0.
                        # Wait, my previous logic for Red->Blue K5:
                        # "if rc_k5[k5_idx] == 0: delta -= 1" -> Correct.
                        # "elif rc_k5[k5_idx] == 1: delta += 1" -> INCORRECT.
                        # If rc_k5 is 1, it is NOT bad. It becomes 2, still NOT bad.
                        # So Delta 0.
                        
                        # Let's re-verify Blue -> Red for K4:
                        # rc_k4 increases by 1.
                        # Bad if rc_k4 == 4.
                        # If rc_k4 was 3, becomes 4. Bad. Delta +1.
                        # If rc_k4 was 4, becomes 5 (impossible).
                        # So only check if rc_k4[k5_idx] == 3.
                        
                        # Let's re-verify Red -> Blue for K4:
                        # rc_k4 decreases by 1.
                        # If rc_k4 was 4, becomes 3. Not bad. Delta -1.
                        # If rc_k4 was < 4, becomes < 4. Not bad. Delta 0.
                        
                        # So the logic should be:
                        # K4:
                        # R->B: if rc==4, delta -= 1
                        # B->R: if rc==3, delta += 1
                        # K5:
                        # R->B: if rc==0, delta -= 1
                        # B->R: if rc==0, delta -= 1
                        # Wait, B->R for K5:
                        # rc increases.
                        # If rc was 0 (bad), becomes 1 (not bad). Delta -1.
                        # If rc was > 0, becomes > 1. Not bad. Delta 0.
                        # So for K5, both R->B and B->R have delta -1 if rc==0?
                        # R->B: rc decreases. If rc was 0, it was bad. Becomes -1? No, rc counts reds.
                        # If rc was 0, all were blue. Flip one blue to red. rc becomes 1.
                        # So rc goes 0 -> 1.
                        # My code says:
                        # if old_color == 0: (R->B)
                        #   rc decreases.
                        #   if rc == 0: delta -= 1.
                        #   This implies rc was 0 BEFORE flip.
                        #   If rc was 0, it was bad. After flip, rc is -1? No, we are calculating delta based on state BEFORE flip.
                        #   If rc_before == 0, it was bad. After flip, it is not bad. So delta -1.
                        #   Correct.
                        # if old_color == 1: (B->R)
                        #   rc increases.
                        #   if rc == 0: delta -= 1.
                        #   If rc_before == 0, it was bad. After flip, rc is 1. Not bad. So delta -1.
                        #   Correct.
                        
                        # So for K5, if rc_before == 0, delta is always -1 regardless of flip direction?
                        # Yes, because 0 reds means all blue (bad). Any change to red count makes it not all blue.
                        # But wait.
                        # If rc_before == 10 (all red). Not bad for K5 (blue).
                        # Flip R->B: rc becomes 9. Not bad.
                        # Flip B->R: rc becomes 11 (impossible).
                        
                        # So the logic holds.
                        
                # Accept move
                if delta < 0 or (random.random() < pow(2.718, -delta / max(T, 0.001))):
                    # Apply move
                    colors[t] = new_color
                    
                    # Update counts
                    for k4_idx in triple_to_k4s[t]:
                        if old_color == 0:
                            rc_k4[k4_idx] -= 1
                        else:
                            rc_k4[k4_idx] += 1
                            
                    for k5_idx in triple_to_k5s[t]:
                        if old_color == 0:
                            rc_k5[k5_idx] -= 1
                        else:
                            rc_k5[k5_idx] += 1
                            
                    violations += delta
                    
                    if violations == 0:
                        return colors
                        
                    T *= cooling
                    if T < min_T:
                        T = min_T
                else:
                    T *= cooling
                    if T < min_T:
                        T = min_T
                steps += 1
                
        return None

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 15:
            break
        colors = solve_n(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: return seed result
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

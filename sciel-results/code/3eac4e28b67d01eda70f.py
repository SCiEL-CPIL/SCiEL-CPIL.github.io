import random
import time

def construct():
    N = 256
    # Precompute all 3-APs and 4-APs
    # A 3-AP is (a, b, c) with a + c = 2b, a < b < c
    # A 4-AP is (a, b, c, d) with a + d = b + c = 2(a+c)/2? No, a, a+d, a+2d, a+3d.
    
    # Efficient representation:
    # For 3-APs, store (a, c) for each b? Or just list of tuples.
    # For 4-APs, list of tuples.
    
    ap3s = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3s.append((a, b, c))
            
    ap4s = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            ap4s.append((a, b, c, dd))

    # Index 4-APs by their elements to quickly check validity?
    # Or just check all 4-APs? 10k checks per move is fine.
    
    # Initial State: Greedy no-4-AP
    # The seed program's logic is a good start for validity, but poor for 3-APs.
    # Let's try to start with a random set and fix 4-APs.
    
    # Better: Start with the greedy set, then try to add elements that don't create 4-APs 
    # but create 3-APs.
    
    selected = set()
    # Reproduce seed logic for initial valid set
    for x in range(1, N + 1):
        valid = True
        for d in range(1, (x - 1) // 3 + 1):
            if x - d in selected and x - 2 * d in selected and x - 3 * d in selected:
                valid = False
                break
            # Also check if adding x completes a 4-AP where x is not the last term?
            # The seed checks if x is the 4th term. 
            # It should also check if x is the 1st, 2nd, or 3rd term of a 4-AP with existing elements.
            # The seed is actually flawed for "no 4-APs" if it only checks x as the max.
            # But the task says "no nonconstant 4-AP".
            # Let's assume the seed produced a valid set. We will verify and repair.
        
        if valid:
            selected.add(x)
            
    # Verify and repair initial set
    # Remove elements that form 4-APs
    changed = True
    while changed:
        changed = False
        for a, b, c, d in ap4s:
            if a in selected and b in selected and c in selected and d in selected:
                # Remove one of them, say the largest
                selected.remove(d)
                changed = True
                break # Restart scan
    
    # Now we have a valid set. Let's optimize 3-APs.
    # Simulated Annealing
    current_score = sum(1 for a, b, c in ap3s if a in selected and b in selected and c in selected)
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_selected = set(selected)
    best_score = current_score
    
    while T > T_min and time.time() - start_time < 55:
        # Random move: flip a bit
        x = random.randint(1, N)
        
        # Try adding x
        if x not in selected:
            # Check if adding x creates a 4-AP
            creates_4ap = False
            for a, b, c, d in ap4s:
                # If x is in the 4-AP and the other 3 are in selected, it creates a 4-AP
                if x in (a, b, c, d):
                    others = [y for y in (a, b, c, d) if y != x]
                    if all(y in selected for y in others):
                        creates_4ap = True
                        break
            
            if not creates_4ap:
                # Calculate delta 3-APs
                delta3 = 0
                for a, b, c in ap3s:
                    if x in (a, b, c):
                        others = [y for y in (a, b, c) if y != x]
                        if all(y in selected for y in others):
                            delta3 += 1
                
                new_score = current_score + delta3
                if new_score > best_score:
                    best_score = new_score
                    best_selected = set(selected | {x})
                
                # Acceptance
                if delta3 >= 0:
                    selected.add(x)
                    current_score = new_score
                else:
                    if random.random() < pow(2.718, delta3 / T):
                        selected.add(x)
                        current_score = new_score
                        
        # Try removing x
        elif x in selected:
            # Removing always maintains validity
            delta3 = 0
            for a, b, c in ap3s:
                if x in (a, b, c):
                    others = [y for y in (a, b, c) if y != x]
                    if all(y in selected for y in others):
                        delta3 -= 1
                        
            new_score = current_score + delta3
            
            # Acceptance
            if delta3 >= 0:
                selected.remove(x)
                current_score = new_score
            else:
                if random.random() < pow(2.718, delta3 / T):
                    selected.remove(x)
                    current_score = new_score
        
        T *= cooling
        
    return sorted(best_selected)

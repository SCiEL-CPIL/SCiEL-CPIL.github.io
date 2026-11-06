import time
import random

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for 90s limit
    
    N = 256
    # Precompute 3-APs and 4-APs
    # 3-APs: (a, b, c) such that a+b=2c, a<b<c
    # 4-APs: (a, b, c, d) such that a+d=b+c=2b? No, a, a+d, a+2d, a+3d.
    
    # We need to know for each x, which 3-APs and 4-APs it participates in.
    # ap3_inv[x] = list of (a, c) such that (a, x, c) is a 3-AP (x is middle)
    #             + list of (x, c) such that (x, x+d, x+2d) is 3-AP (x is first) -> need (x, x+d, x+2d)
    #             + list of (a, x) such that (a, a+d, x) is 3-AP (x is last) -> need (a, x-d, x)
    
    # Actually, simpler:
    # For each 3-AP (a, b, c), store it in a list.
    # For each 4-AP (a, b, c, d), store it in a list.
    
    # To check if adding x creates a 4-AP:
    # Iterate through all 4-APs that contain x. If the other 3 are in S, it's invalid.
    
    # To calculate gain of adding x:
    # Iterate through all 3-APs that contain x. If the other 2 are in S, gain += 1.
    
    # Precomputation
    ap3s = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            ap3s.append((a, a+d, a+2*d))
            
    ap4s = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap4s.append((a, a+d, a+2*d, a+3*d))
            
    # Map element to APs it is part of
    # elem_to_ap3[x] = list of indices of ap3s containing x
    # elem_to_ap4[x] = list of indices of ap4s containing x
    elem_to_ap3 = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)
        
    S = set()
    
    # Greedy Construction
    candidates = list(range(1, N+1))
    random.shuffle(candidates) # Tie-breaking
    
    while True:
        if time.time() - start_time > timeout - 10:
            break
            
        best_x = -1
        best_gain = -1
        valid_candidates = []
        
        for x in candidates:
            if x in S:
                continue
                
            # Check 4-AP constraint
            valid = True
            for idx in elem_to_ap4[x]:
                a, b, c, d = ap4s[idx]
                # Check if other 3 are in S
                others = [a, b, c, d]
                others.remove(x)
                if all(o in S for o in others):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Gain
            gain = 0
            for idx in elem_to_ap3[x]:
                a, b, c = ap3s[idx]
                others = [a, b, c]
                others.remove(x)
                if all(o in S for o in others):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                valid_candidates = [x]
            elif gain == best_gain:
                valid_candidates.append(x)
                
        if best_x == -1:
            break
            
        # If multiple candidates have same gain, pick one randomly to add diversity
        # Or just pick the first one found (which was best_x)
        # Let's pick randomly from valid_candidates if best_gain > 0, else best_x
        if best_gain > 0 and len(valid_candidates) > 1:
            best_x = random.choice(valid_candidates)
            
        S.add(best_x)
        candidates.remove(best_x)
        
    # Local Search
    # Count 3-APs in current S
    def count_3aps(S):
        count = 0
        for idx in range(len(ap3s)):
            a, b, c = ap3s[idx]
            if a in S and b in S and c in S:
                count += 1
        return count
        
    current_score = count_3aps(S)
    
    # Optimize with swaps
    # To speed up, we can estimate the change in score for a swap (x, y)
    # Delta = Gain(y) - Loss(x) + Interaction
    # But interaction is complex. Let's just do a few rounds of random swaps if time permits.
    
    # A better local search:
    # For each x in S, try removing it.
    # Then for each y not in S, try adding it.
    # This is O(|S| * (N-|S|)) per step.
    # |S| is likely around 50-100. N=256.
    # 100 * 150 = 15000 checks.
    # Each check involves counting 3-APs? No, we can update incrementally.
    
    # Let's try a simple hill climbing:
    # 1. Try removing each x in S. If no 4-AP is formed (actually removing never forms 4-AP), calculate new score.
    #    Wait, removing x might break 3-APs.
    # 2. Try adding each y not in S. If no 4-AP is formed, calculate new score.
    #    We want to find a swap (x, y) that increases score.
    
    # Since N is small, we can afford to be a bit slower.
    # Let's do a few iterations of:
    #   For each x in S:
    #     For each y not in S:
    #       If S' = S - {x} + {y} is valid (no 4-AP):
    #          If Score(S') > Score(S):
    #             S = S'
    #             Score(S) = Score(S')
    #             break inner loop
    #     If improved, break outer loop? Or continue?
    
    # To make it faster, we can pre-calculate the "cost" of removing x and "benefit" of adding y.
    # Cost(x) = number of 3-APs in S that contain x.
    # Benefit(y) = number of 3-APs in S that would be completed by y.
    # Interaction: 3-APs that contain both x and y? If x is removed, those are lost. If y is added, those are gained?
    # Actually, if we swap x for y:
    # New Score = Old Score - Cost(x) + Benefit(y) - Interaction(x, y)
    # Where Interaction(x, y) is the number of 3-APs that contain both x and y.
    # Because those 3-APs are counted in Cost(x) (lost) and Benefit(y) (gained? No, if they contain x and y, and x is removed, they are lost. If y is added, they are not gained because x is gone).
    # Wait.
    # Let A be set of 3-APs in S.
    # Let B be set of 3-APs in S \ {x} \cup {y}.
    # A 3-AP in B either:
    # 1. Does not contain x or y. (In A and B)
    # 2. Contains y but not x. (In B, not in A). This is part of Benefit(y) excluding those with x.
    # 3. Contains x but not y. (In A, not in B). This is part of Cost(x) excluding those with y.
    # 4. Contains both x and y. (In A, not in B, because x is removed).
    
    # So, Delta = (Benefit(y) - Interaction(x, y)) - (Cost(x) - Interaction(x, y)) = Benefit(y) - Cost(x).
    # Wait, is this correct?
    # Let's verify.
    # Score(S) = |A|
    # Score(S') = |B|
    # A \ B = {3-APs containing x, not y}
    # B \ A = {3-APs containing y, not x}
    # |A| - |B| = |A \ B| - |B \ A|
    # |A \ B| = Cost(x) - Interaction(x, y)
    # |B \ A| = Benefit(y) - Interaction(x, y)
    # So |B| - |A| = (Cost(x) - Interaction(x, y)) - (Benefit(y) - Interaction(x, y)) ... wait.
    # |A| - |B| = (Cost(x) - Int) - (Benefit(y) - Int) = Cost(x) - Benefit(y).
    # So |B| - |A| = Benefit(y) - Cost(x).
    # Yes! The interaction terms cancel out.
    # So we just need Cost(x) and Benefit(y).
    
    # Cost(x) for x in S: number of 3-APs in S containing x.
    # Benefit(y) for y not in S: number of 3-APs in S that can be completed by y (i.e., other 2 in S).
    
    # We can maintain Cost and Benefit incrementally?
    # Or just recalculate for each swap candidate?
    # Recalculating Cost(x) for all x in S is O(|S| * deg(x)).
    # deg(x) is small.
    
    # Let's try this local search.
    
    improved = True
    while improved and time.time() - start_time < timeout - 5:
        improved = False
        
        # Calculate Cost for all x in S
        cost = {x: 0 for x in S}
        for idx in range(len(ap3s)):
            a, b, c = ap3s[idx]
            if a in S and b in S and c in S:
                cost[a] += 1
                cost[b] += 1
                cost[c] += 1
                
        # Calculate Benefit for all y not in S
        # This is expensive if done naively for all y.
        # Benefit(y) = count of (a, c) in S such that (a, y, c) is 3-AP, etc.
        # We can iterate over 3-APs and check if 2 are in S and 1 is y.
        
        # Let's just try random swaps to speed up.
        # Pick random x in S, random y not in S.
        # Check validity and score change.
        
        for _ in range(100): # 100 random swaps per iteration
            if time.time() - start_time > timeout - 5:
                break
                
            x = random.choice(list(S))
            y_candidates = [y for y in range(1, N+1) if y not in S]
            if not y_candidates:
                break
            y = random.choice(y_candidates)
            
            # Check 4-AP validity for S' = S - {x} + {y}
            valid = True
            # Removing x cannot create 4-AP.
            # Adding y might.
            for idx in elem_to_ap4[y]:
                a, b, c, d = ap4s[idx]
                others = [a, b, c, d]
                others.remove(y)
                # Check if other 3 are in S'
                # S' = S - {x} + {y}
                # So we need others to be in S, and if x is in others, it's removed.
                # Actually, if x is in others, then x is NOT in S'.
                # So we check: for o in others, o in S' ?
                # o in S' iff (o in S and o != x) or (o == y)
                # But y is not in others.
                # So o in S' iff o in S and o != x.
                
                all_in_sp = True
                for o in others:
                    if o == x:
                        continue # x is removed, so o is not in S'
                    if o not in S:
                        all_in_sp = False
                        break
                if all_in_sp:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Delta
            # We need Cost(x) and Benefit(y)
            # We have cost[x] from above.
            # Need Benefit(y).
            
            # Calculate Benefit(y)
            benefit_y = 0
            for idx in elem_to_ap3[y]:
                a, b, c = ap3s[idx]
                others = [a, b, c]
                others.remove(y)
                # Check if others are in S'
                # o in S' iff o in S and o != x
                all_in_sp = True
                for o in others:
                    if o == x:
                        continue
                    if o not in S:
                        all_in_sp = False
                        break
                if all_in_sp:
                    benefit_y += 1
            
            delta = benefit_y - cost[x]
            
            if delta > 0:
                S.remove(x)
                S.add(y)
                improved = True
                # Update cost? No, we recalculate at start of next while loop.
                break # Restart local search loop to recalculate costs
                
    return sorted(S)

construct()

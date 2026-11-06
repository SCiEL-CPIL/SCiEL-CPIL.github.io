import random
import time
import math

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            e = a + 3 * d
            if e > N:
                break
            ap = (a - 1, a + d - 1, a + 2 * d - 1, e - 1)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                index_to_aps[idx].append(ap_idx)
                
    def count_violations(colors):
        # Returns number of APs with < 3 distinct colors
        violations = 0
        for ap in aps:
            c1 = colors[ap[0]]
            c2 = colors[ap[1]]
            c3 = colors[ap[2]]
            c4 = colors[ap[3]]
            # Check if distinct count < 3
            # If all same: 1
            # If 2 same, 2 same: 2 (e.g., 0,0,1,1)
            # If 3 same, 1 diff: 2 (e.g., 0,0,0,1)
            # If 2 same, 1 same, 1 diff: 3 (e.g., 0,0,1,2) -> OK
            # If 2 same, 2 diff: 3 (e.g., 0,0,1,2) -> OK
            # If all diff: 4 -> OK
            
            # Fast check for < 3 distinct:
            # It fails if the set of colors has size 1 or 2.
            # Size 1: c1==c2==c3==c4
            # Size 2: 
            #   (c1==c2 and c3==c4) and (c1==c3 or c1!=c3) -> actually just check if only 2 values present.
            
            # Simple check:
            # If c1==c2 and c3==c4: 
            #   if c1==c3: 1 color (Bad)
            #   else: 2 colors (Bad)
            #   So if c1==c2 and c3==c4, it's always Bad.
            if c1 == c2:
                if c3 == c4:
                    violations += 1
                else:
                    # c1, c1, c3, c4 with c3!=c4
                    # If c1==c3 or c1==c4, then only 2 colors.
                    if c1 == c3 or c1 == c4:
                        violations += 1
            elif c1 == c3:
                # c1, c2, c1, c4
                if c2 == c4:
                    violations += 1
            elif c1 == c4:
                # c1, c2, c3, c1
                if c2 == c3:
                    violations += 1
            else:
                # c1 distinct from c2, c3, c4
                if c2 == c3:
                    if c2 == c4:
                        violations += 1
                elif c2 == c4:
                    if c1 == c3: # c1!=c2, c1!=c4. c2==c4.
                        # Colors: c1, c2, c3, c2.
                        # If c1==c3, colors are {c1, c2}. Bad.
                        violations += 1
                elif c3 == c4:
                    if c2 == c3:
                        violations += 1
        return violations

    def get_local_violations(idx, colors):
        # Count violations in APs involving idx
        count = 0
        for ap_idx in index_to_aps[idx]:
            ap = aps[ap_idx]
            c1 = colors[ap[0]]
            c2 = colors[ap[1]]
            c3 = colors[ap[2]]
            c4 = colors[ap[3]]
            if c1 == c2:
                if c3 == c4:
                    count += 1
                else:
                    if c1 == c3 or c1 == c4:
                        count += 1
            elif c1 == c3:
                if c2 == c4:
                    count += 1
            elif c1 == c4:
                if c2 == c3:
                    count += 1
            else:
                if c2 == c3:
                    if c2 == c4:
                        count += 1
                elif c2 == c4:
                    if c1 == c3:
                        count += 1
                elif c3 == c4:
                    if c2 == c3:
                        count += 1
        return count

    def simulated_annealing(K, time_limit):
        start_time = time.time()
        # Initialize random
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_energy = count_violations(colors)
        
        if current_energy == 0:
            return colors

        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-5
        
        # To speed up, we can pre-compute the APs for each index
        # and just re-evaluate those.
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
            
            # Pick random index
            i = random.randint(0, N - 1)
            old_color = colors[i]
            
            # Pick new color
            if K == 1:
                new_color = 0
            else:
                new_color = random.randint(0, K - 1)
                while new_color == old_color:
                    new_color = random.randint(0, K - 1)
            
            # Calculate delta energy
            # Energy is sum of violations over all APs.
            # Only APs involving i change.
            
            old_local = 0
            new_local = 0
            
            for ap_idx in index_to_aps[i]:
                ap = aps[ap_idx]
                # Old state
                c1 = colors[ap[0]]
                c2 = colors[ap[1]]
                c3 = colors[ap[2]]
                c4 = colors[ap[3]]
                
                # Check old violation
                is_old_bad = False
                if c1 == c2:
                    if c3 == c4:
                        is_old_bad = True
                    else:
                        if c1 == c3 or c1 == c4:
                            is_old_bad = True
                elif c1 == c3:
                    if c2 == c4:
                        is_old_bad = True
                elif c1 == c4:
                    if c2 == c3:
                        is_old_bad = True
                else:
                    if c2 == c3:
                        if c2 == c4:
                            is_old_bad = True
                    elif c2 == c4:
                        if c1 == c3:
                            is_old_bad = True
                    elif c3 == c4:
                        if c2 == c3:
                            is_old_bad = True
                if is_old_bad:
                    old_local += 1
                    
                # New state
                # Temporarily set colors[i] to new_color? 
                # Or just calculate with new_color at position i.
                # We know which position i is in the AP.
                # ap = (i0, i1, i2, i3)
                
                # Re-construct colors for this AP with new_color at i
                # Since we are in a loop, it's faster to just swap in the array, check, swap out.
                # But we are calculating delta.
                
                colors[i] = new_color
                
                nc1 = colors[ap[0]]
                nc2 = colors[ap[1]]
                nc3 = colors[ap[2]]
                nc4 = colors[ap[3]]
                
                is_new_bad = False
                if nc1 == nc2:
                    if nc3 == nc4:
                        is_new_bad = True
                    else:
                        if nc1 == nc3 or nc1 == nc4:
                            is_new_bad = True
                elif nc1 == nc3:
                    if nc2 == nc4:
                        is_new_bad = True
                elif nc1 == nc4:
                    if nc2 == nc3:
                        is_new_bad = True
                else:
                    if nc2 == nc3:
                        if nc2 == nc4:
                            is_new_bad = True
                    elif nc2 == nc4:
                        if nc1 == nc3:
                            is_new_bad = True
                    elif nc3 == nc4:
                        if nc2 == nc3:
                            is_new_bad = True
                if is_new_bad:
                    new_local += 1
                    
                colors[i] = old_color # Revert for next AP in loop? 
                # Actually, we can just leave it reverted for each AP check 
                # to avoid confusion, or better, calculate new_local 
                # by just substituting the value.
                
            # Wait, the loop above reverts colors[i] for each AP.
            # This is correct for calculating the sum of local violations.
            # But we need to apply the change to calculate the total delta.
            # Delta = new_local - old_local.
            
            delta = new_local - old_local
            
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                colors[i] = new_color
                current_energy += delta
                if current_energy == 0:
                    return colors
            
            T *= cooling_rate
            
        return None

    # Try increasing K
    # Minimum K is 3.
    # Allocate time based on remaining time.
    total_time_limit = 85.0 # 5s safety margin
    start_time = time.time()
    
    for K in range(3, 11):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time to this K
        # More time for smaller K as it's more constrained? 
        # Or equal time?
        # Let's give 10s per K attempt.
        time_for_k = min(10.0, remaining)
        
        sol = simulated_annealing(K, time_for_k)
        if sol:
            return sol
            
    # Fallback: Greedy with 128 colors (always valid)
    # This ensures we return *something* valid, even if score is low.
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors

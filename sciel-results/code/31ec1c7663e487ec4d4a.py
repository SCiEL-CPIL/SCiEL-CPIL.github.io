import math

def construct():
    # We want to maximize sum(1/m).
    # We process moduli in increasing order of m (decreasing 1/m).
    # For each m, we choose the residue r that allows the most (or heaviest) future moduli to be added.
    
    selected = []  # list of (m, r)
    
    # Precompute gcds for speed? Or just use math.gcd.
    
    for m in range(20, 201):
        best_r = None
        best_score = -1
        best_future_moduli = [] # To track which ones are added for tie-breaking or just score
        
        for r in range(m):
            # Check validity against current selected
            valid = True
            for (m2, r2) in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If valid, simulate adding (m, r) and see how many future moduli we can add
            # We use a simple greedy for the future simulation to keep it fast
            temp_selected = selected + [(m, r)]
            future_sum = 0.0
            future_count = 0
            
            # Simulate future moduli m' > m
            for m_prime in range(m + 1, 201):
                found_r_prime = False
                for r_prime in range(m_prime):
                    valid_prime = True
                    for (m3, r3) in temp_selected:
                        g3 = math.gcd(m_prime, m3)
                        if r_prime % g3 == r3 % g3:
                            valid_prime = False
                            break
                    if valid_prime:
                        found_r_prime = True
                        # We add it to temp_selected to block others in simulation
                        temp_selected.append((m_prime, r_prime))
                        future_sum += 1.0 / m_prime
                        future_count += 1
                        break
                
                # Note: In the simulation, we only add if found.
                # This simulation is a greedy choice for the future.
                
            # We want to maximize the total score contribution: 1/m + future_sum
            # Since 1/m is constant for this m, we maximize future_sum.
            # If future_sum is equal, we might prefer the one with more count? 
            # Or just pick the first one?
            
            if future_sum > best_score:
                best_score = future_sum
                best_r = r
                # Store the simulated selection to avoid re-computing? 
                # No, we just need best_r. The actual selection will be updated.
        
        if best_r is not None:
            selected.append((m, best_r))
        # If best_r is None, no residue works for m given current selections.
        # This shouldn't happen often if we start empty, but might happen later.
        
    return [[m, r] for m, r in selected]

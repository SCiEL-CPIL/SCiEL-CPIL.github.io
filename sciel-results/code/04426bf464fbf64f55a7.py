import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Beam search to find a valid set of disjoint residue classes
    # State: list of (m, r)
    beam = [[]]
    beam_width = 20
    
    # We consider moduli in increasing order to prioritize high weights (1/m)
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
            
        new_beam = []
        
        for state in beam:
            # Find all valid residues r for modulus m given the current state
            valid_rs = []
            # Optimization: If state is empty, all r are valid
            if not state:
                valid_rs = list(range(m))
            else:
                for r in range(m):
                    ok = True
                    for mi, ri in state:
                        g = math.gcd(m, mi)
                        if r % g == ri % g:
                            ok = False
                            break
                    if ok:
                        valid_rs.append(r)
            
            if valid_rs:
                # We found valid residues. We should include m.
                # To maintain diversity in the beam, we don't want to add all m valid residues
                # as separate branches, as that explodes the beam size.
                # We select a diverse subset.
                # Strategy: Pick residues that are spaced apart.
                if len(valid_rs) <= 3:
                    selected_rs = valid_rs
                else:
                    # Take evenly spaced samples
                    step = len(valid_rs) // 3
                    selected_rs = [valid_rs[i] for i in range(0, len(valid_rs), step)]
                    # Ensure we don't miss the last one if needed, but 3 is enough
                    if selected_rs[-1] != valid_rs[-1]:
                        selected_rs.append(valid_rs[-1])
                    # Cap at 3 to keep beam size controlled
                    selected_rs = selected_rs[:3]
                
                for r in selected_rs:
                    new_beam.append(state + [(m, r)])
            else:
                # No valid residue, must skip m
                new_beam.append(state)
        
        # Prune the new beam
        # Prioritize states with more moduli (higher sum potential)
        # If counts are equal, we keep them as they are.
        new_beam.sort(key=lambda s: len(s), reverse=True)
        beam = new_beam[:beam_width]
        
    # Final step: Select the best state from the beam
    # Calculate exact score for each candidate in the beam
    best_score = -1
    best_state = []
    
    for state in beam:
        # Calculate sum of 1/m
        score = 0.0
        for m, r in state:
            score += 1.0 / m
        
        if score > best_score:
            best_score = score
            best_state = state
    
    # If no state was found (unlikely), return empty or a default
    if not best_state:
        # Fallback: just return the first valid simple configuration
        # Multiples of 20
        return [[20 * j, j - 1] for j in range(1, 11)]
        
    return [[m, r] for m, r in best_state]

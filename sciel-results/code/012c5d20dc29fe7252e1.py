import math
import random

def construct():
    # Moduli in range [20, 200]
    moduli = list(range(20, 201))
    
    # Sort by weight 1/m descending
    moduli.sort(key=lambda m: -1.0/m)
    
    # Beam search parameters
    beam_width = 50
    
    # Each state is (selected_moduli, selected_residues, score)
    # We process moduli one by one.
    # To save memory and time, we don't store the full history in the beam, 
    # but we need to check compatibility with all previous selections.
    
    # Initial state
    # selected: list of (m, r)
    beam = [([], [], 0.0)]
    
    for m in moduli:
        new_beam = []
        for selected_mods, selected_res, score in beam:
            # Option 1: Skip m
            new_beam.append((selected_mods, selected_res, score))
            
            # Option 2: Include m
            # Find valid residues
            valid_residues = []
            for r in range(m):
                valid = True
                for m2, r2 in zip(selected_mods, selected_res):
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_residues.append(r)
            
            if valid_residues:
                # To keep beam size manageable, we don't add ALL valid residues.
                # We add a sample or the "best" ones.
                # Heuristic: Residues that are "central" or random?
                # Since the score contribution of m is fixed (1/m), the choice of r 
                # only affects future compatibility.
                # A good heuristic is to pick residues that are distinct modulo 
                # the gcds with future moduli? Hard to know future moduli easily 
                # without looking ahead.
                
                # For now, let's add up to 3 distinct valid residues to the beam expansions
                # to maintain diversity.
                # If there are few, add all.
                sample_size = min(len(valid_residues), 3)
                # Use deterministic sampling for reproducibility or random?
                # Random is better for escape local minima, but we need to be fast.
                # Let's pick the first, middle, and last valid residue if possible.
                indices = list(range(len(valid_residues)))
                if len(indices) > 3:
                    chosen_indices = [0, len(indices)//2, len(indices)-1]
                else:
                    chosen_indices = indices
                    
                for idx in chosen_indices:
                    r = valid_residues[idx]
                    new_selected_mods = selected_mods + [m]
                    new_selected_res = selected_res + [r]
                    new_score = score + 1.0/m
                    new_beam.append((new_selected_mods, new_selected_res, new_score))
        
        # Prune beam: keep top beam_width states by score
        # If scores are tied, we might want to keep diverse states.
        # Simple sort by score desc.
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
        
        # Early termination if beam is empty (shouldn't happen)
        if not beam:
            break
            
    # Return the best state
    best_state = beam[0]
    selected_mods, selected_res, _ = best_state
    result = [[m, r] for m, r in zip(selected_mods, selected_res)]
    return result

import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Beam Search parameters
    beam_width = 50
    N = 512
    max_a = N - 1
    
    # Each beam is a tuple: (score, frozenset_of_chosen_a, list_of_chosen_a)
    # We use list for fast append, and maintain gcd compatibility.
    # To speed up gcd checks, we can store the set of prime factors of the product?
    # No, just checking gcd(a, b) for b in chosen is fine.
    
    # Initialize beam with one empty state
    # State: [score, chosen_list]
    # We sort beams by score descending at each step.
    
    beams = [[0.0, []]]
    
    # Pre-compute weights
    weights = [1.0 / (N - a) for a in range(max_a, 0, -1)]
    # weights[i] corresponds to a = max_a - i
    
    # Iterate a from max_a down to 1
    for a in range(max_a, 0, -1):
        w = 1.0 / (N - a)
        new_beams = []
        
        for score, chosen in beams:
            # Option 1: Skip a
            new_beams.append((score, chosen))
            
            # Option 2: Include a (if coprime to all in chosen)
            if all(gcd(a, b) == 1 for b in chosen):
                new_beams.append((score + w, chosen + [a]))
        
        # Prune to beam_width
        # Sort by score descending
        # To avoid copying lists, we can sort indices or just sort the list of tuples
        # Note: chosen lists are shared references if we don't copy. 
        # In Option 1, we append the same list object. 
        # In Option 2, we create a new list.
        # This is okay for beam search as long as we don't modify the old lists.
        
        # Sort by score
        new_beams.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top beam_width
        # We need to be careful: if we keep multiple beams that share the same 'chosen' list object,
        # and later we append to it, it will affect all.
        # So we must ensure that when we create a new beam, we copy the list if we plan to modify it?
        # Actually, in the next iteration, we will create new lists from these.
        # The 'chosen' in new_beams are either the old list (skip) or a new list (include).
        # If we keep a 'skip' beam, it points to the old list.
        # If we keep an 'include' beam, it points to a new list.
        # In the next step, we iterate over these beams.
        # For 'skip', we pass the list again.
        # For 'include', we create a new list from it.
        # So the old lists are never modified in-place. They are immutable in this context.
        
        beams = new_beams[:beam_width]
        
        # Time check
        if time.time() - start_time > deadline - safety_margin:
            break
    
    # Return the best beam's chosen list
    best_beam = max(beams, key=lambda x: x[0])
    return best_beam[1]

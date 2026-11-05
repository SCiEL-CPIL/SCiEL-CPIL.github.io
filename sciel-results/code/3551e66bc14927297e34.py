import math
import time

def construct():
    deadline = time.time() + 110  # Safety margin
    
    moduli = list(range(20, 201))
    # Sort by weight 1/m descending -> m ascending
    moduli.sort()
    
    # Beam Search
    # State: (list_of_pairs, score)
    # We store list_of_pairs as a tuple of (m, r) to be hashable? No, we just need to iterate.
    # To save memory and time, we can store just the residues for the moduli seen so far?
    # No, because we need to check against all previous.
    # Since we process in order, we can store the list of (m, r).
    
    beam_width = 1000
    current_beam = [([], 0.0)]
    
    for m in moduli:
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for pairs, score in current_beam:
            # Option 1: Skip m
            # We add this to new_beam. 
            # Note: Multiple paths can lead to the same set of pairs? 
            # Yes, but score is determined by the set of m's. 
            # If we skip, the set of m's is the same.
            # So we might have duplicates in new_beam. 
            # We should deduplicate based on the set of moduli included? 
            # Or just let them exist and sort by score. 
            # Since score is sum of 1/m, identical sets have identical scores.
            # We can keep only one copy of a specific set of moduli+residues.
            # But residues matter for future constraints.
            # So (pairs, score) is the state.
            
            # To avoid explosion, we can just append and let sorting handle it.
            # But 1000 states * 200 residues = 200k new states per step.
            # 181 steps * 200k = 36M states. This might be too much memory/time in Python if we create lists.
            
            # Optimization: 
            # Instead of storing full lists, we can store the residues in a fixed-size array?
            # Or just accept the list copy cost. 
            # List copy of size ~100 is fast.
            
            # Add skip state
            new_beam.append((pairs, score))
            
            # Add include states
            # We need to check compatibility of r with all p in pairs
            # pairs is a list of (m_i, r_i)
            
            # Pre-compute gcds for this m against all m_i in pairs?
            # The set of m_i in pairs is the same for all states in the current beam?
            # NO. Different states in the beam have different sets of moduli included.
            # So we must check against the specific pairs in this state.
            
            for r in range(m):
                valid = True
                for m_i, r_i in pairs:
                    g = math.gcd(m, m_i)
                    if r % g == r_i % g:
                        valid = False
                        break
                if valid:
                    new_pairs = pairs + [(m, r)]
                    new_beam.append((new_pairs, score + 1.0/m))
        
        # Sort by score descending
        new_beam.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top beam_width
        current_beam = new_beam[:beam_width]
        
        # Optional: Prune identical sets of moduli?
        # If two states have the same set of moduli but different residues, they have the same score.
        # We should keep the one that is "best" for future? 
        # Hard to define "best". 
        # But keeping multiple diverse residue assignments for the same moduli set is good for escaping local optima.
        # So we keep them.
        
    # Find the best score in the final beam
    best_pairs = []
    best_score = 0.0
    for pairs, score in current_beam:
        if score > best_score:
            best_score = score
            best_pairs = pairs
            
    return [[m, r] for m, r in best_pairs]

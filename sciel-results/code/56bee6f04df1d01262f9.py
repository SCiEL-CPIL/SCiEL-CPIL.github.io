import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy: Find the best K such that we can pick max( floor(128/K), K ) moduli
    # Actually, we can pick min( floor(128/K), K ) if we use the simple distinct residue logic.
    # But we can potentially pick more if we use the structure of gcds.
    # For now, return the best single-factor set found.
    
    best_count = 0
    best_moduli = []
    best_residues = []
    
    for K in range(2, 129):
        candidates = []
        for m in range(K, 129, K):
            candidates.append(m)
        
        # We can pick at most K residues distinct mod K.
        # We have len(candidates) candidates.
        # We can pick min(len(candidates), K) moduli.
        # And we can always assign distinct residues 0..N-1 mod K.
        n = min(len(candidates), K)
        
        if n > best_count:
            best_count = n
            best_moduli = candidates[:n]
            best_residues = list(range(n))
            
    # Construct the result
    result = []
    for r, m in zip(best_residues, best_moduli):
        result.append([r, m])
        
    return result

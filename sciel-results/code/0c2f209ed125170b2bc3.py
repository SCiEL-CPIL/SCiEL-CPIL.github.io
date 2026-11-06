import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Use affine geometry AG(4,2) construction:
    # Points are vectors in GF(2)^4, represented as integers 0..15
    # Lines (triples) are {a, b, a^b} for distinct a, b in GF(2)^4
    # This gives a linear triple system where any two triples share at most one point.
    
    # Generate all lines in AG(4,2)
    triples = []
    seen = set()
    for a in range(16):
        for b in range(a + 1, 16):
            c = a ^ b
            if c == a or c == b:
                continue
            triple = tuple(sorted([a, b, c]))
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)
    
    # The full affine geometry has C(16,2)/3 = 120/3 = 40 lines
    # Each point appears in (16-1)/2 = 7.5... wait, let me recalculate.
    # In AG(4,2), each point is on (16-1)/(3-1) = 15/2 = 7.5 lines? That's not integer.
    # Actually, in AG(n,2), each point is on (2^n - 1)/(2-1) = 2^n - 1 lines? No.
    # Let me just use the generated triples and compute alpha.
    
    def get_alpha(triple_list):
        """
        Compute the independence number: largest set of points containing no entire triple.
        """
        if not triple_list:
            return 16
        
        triple_masks = []
        for t in triple_list:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check from largest size down
        for k in range(16, 0, -1):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in triple_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1
    
    # Start with the full affine geometry
    current_system = triples[:]
    current_alpha = get_alpha(current_system)
    
    best_alpha = current_alpha
    best_system = current_system[:]
    
    # Try to improve by removing some triples to reduce alpha
    # Actually, removing triples can only increase or maintain alpha.
    # So the full system should have the minimum alpha for this construction.
    
    # Let's verify: the full AG(4,2) has 40 triples.
    # What is its independence number?
    
    # If alpha is already small (e.g., 4 or 5), we're done.
    # If not, we can try local search to find a better subset.
    
    if current_alpha <= 4:
        return [list(t) for t in best_system]
    
    # Local search: try removing triples and see if we can get a better configuration
    # Actually, we want to MINIMIZE alpha, so we want MORE triples, not fewer.
    # The full system is maximal, so alpha should be minimized.
    
    # Let's just return the full affine geometry
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)

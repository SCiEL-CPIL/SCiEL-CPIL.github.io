import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for safety
    
    def try_n(n):
        # Precompute triple indices
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quad indices (4 triples per quad)
        quads = list(combinations(range(n), 4))
        quad_indices = np.array([[triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], 
                                  triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]] 
                                 for a,b,c,d in quads], dtype=np.int32)
        
        # Precompute quint indices (10 triples per quint)
        quints = list(combinations(range(n), 5))
        quint_indices = np.array([[triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], 
                                   triple_to_idx[(a,b,e)], triple_to_idx[(a,c,d)],
                                   triple_to_idx[(a,c,e)], triple_to_idx[(a,d,e)],
                                   triple_to_idx[(b,c,d)], triple_to_idx[(b,c,e)],
                                   triple_to_idx[(b,d,e)], triple_to_idx[(c,d,e)]] 
                                  for a,b,c,d,e in quints], dtype=np.int32)
        
        # Number of triples
        num_triples = len(triples)
        
        # Function to check violations and return indices of violating triples
        def get_violating_triples(colors):
            # colors is a numpy array of 0s and 1s
            # Red violation: quad where all 4 triples are 0
            if len(quad_indices) > 0:
                quad_colors = colors[quad_indices]
                red_violations = np.all(quad_colors == 0, axis=1)
                if np.any(red_violations):
                    violating_quads = quad_indices[red_violations]
                    # Flatten to get individual triple indices
                    red_violating_triples = set(violating_quads.flatten().tolist())
                else:
                    red_violating_triples = set()
            else:
                red_violating_triples = set()
                
            # Blue violation: quint where all 10 triples are 1
            if len(quint_indices) > 0:
                quint_colors = colors[quint_indices]
                blue_violations = np.all(quint_colors == 1, axis=1)
                if np.any(blue_violations):
                    violating_quints = quint_indices[blue_violations]
                    blue_violating_triples = set(violating_quints.flatten().tolist())
                else:
                    blue_violating_triples = set()
            else:
                blue_violating_triples = set()
                
            all_violating = red_violating_triples.union(blue_violating_triples)
            return all_violating

        # Local search loop
        max_iterations = 2000
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                return None
                
            # Initialization strategies
            if attempt == 0:
                colors = np.random.randint(0, 2, num_triples)
            elif attempt == 1:
                # Modular init
                colors = np.array([(a+b+c) % 2 for a,b,c in triples], dtype=np.int8)
            elif attempt == 2:
                # Another modular init
                colors = np.array([(a*b + b*c + c*a) % 2 for a,b,c in triples], dtype=np.int8)
            else:
                # Random with slight bias towards 0 or 1? 
                # R(4,5) suggests blue is more constrained (K5 vs K4), so maybe fewer blues?
                # Actually, random is usually best for local search starts.
                colors = np.random.randint(0, 2, num_triples)
            
            current_colors = colors.copy()
            
            for it in range(max_iterations):
                if time.time() - start_time > time_limit:
                    return None
                    
                violating_triples = get_violating_triples(current_colors)
                
                if not violating_triples:
                    return current_colors.tolist()
                
                # Pick a random violating triple to flip
                flip_idx = random.choice(list(violating_triples))
                current_colors[flip_idx] = 1 - current_colors[flip_idx]
                
        return None

    # Try n=34 first, then decrease
    for n in [34, 33, 32, 30, 25]:
        if time.time() - start_time > time_limit:
            break
            
        colors = try_n(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a small valid n (e.g., n=5 is always valid? 
    # For n=5, no K5 blue? Wait, if all 10 triples are blue, that's a violation.
    # So we must ensure no K5 blue.
    # For n=5, if we color at least one triple red, no blue K5.
    # No red K4 is possible with n=5? Yes, we can pick 4 vertices.
    # We need to ensure no 4 vertices have all 4 triples red.
    # Simple fallback: n=5, colors all 0 except one 1.
    # Triples of 5: (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4)
    # If we set (0,1,2) to 1 (blue), and rest 0 (red).
    # Check Red K4: Any 4 vertices?
    # {0,1,2,3}: Triples (0,1,2)=1, (0,1,3)=0, (0,2,3)=0, (1,2,3)=0. Not all red.
    # {0,1,2,4}: (0,1,2)=1... Not all red.
    # {0,1,3,4}: (0,1,3)=0, (0,1,4)=0, (0,3,4)=0, (1,3,4)=0. ALL RED. VIOLATION.
    # So n=5 requires care.
    
    # Let's just return n=4.
    # n=4: 4 triples.
    # No red K4? The only K4 is {0,1,2,3}. If all 4 are red, violation.
    # So make one blue.
    # No blue K5? Impossible with n=4.
    # So n=4, colors [0,0,0,1] is valid.
    
    return {'n': 4, 'colors': [0, 0, 0, 1]}

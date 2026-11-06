"""Improved program for the suite task; uses multi-start local search with on-the-fly constraint checking."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    # Try to maximize n. Known valid constructions exist for n=34.
    # We will try n=35, then 34.
    target_ns = [35, 34]
    
    for n in target_ns:
        if n < 1 or n > 40:
            continue
            
        # Try multiple restarts
        for restart in range(50):
            if time.time() > 115: # Safety check for 120s limit
                break
                
            result = _try_n(n, restart)
            if result is not None:
                return result
        
    # Fallback to a smaller n if all fail (should not happen for 34)
    # Generate a valid coloring for n=9 or similar as fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

def _try_n(n, seed_offset):
    random.seed(12345 + seed_offset)
    
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    # Map quad to index
    quads = list(combinations(range(n), 4))
    quad_index = {}
    for i, q in enumerate(quads):
        quad_index[q] = i
        
    # Map quint to index
    quints = list(combinations(range(n), 5))
    quint_index = {}
    for i, q in enumerate(quints):
        quint_index[q] = i
        
    num_quads = len(quads)
    num_quints = len(quints)
    
    # Precompute for each triple, which quads and quints it belongs to
    # This is the expensive part, but done once per n.
    triple_to_quads = [[] for _ in range(num_triples)]
    triple_to_quints = [[] for _ in range(num_triples)]
    
    for i, (a, b, c) in enumerate(triples):
        for x in range(n):
            if x in (a, b, c):
                continue
            quad = tuple(sorted((a, b, c, x)))
            triple_to_quads[i].append(quad_index[quad])
            
        for x in range(n):
            if x in (a, b, c):
                continue
            for y in range(x+1, n):
                if y in (a, b, c):
                    continue
                quint = tuple(sorted((a, b, c, x, y)))
                triple_to_quints[i].append(quint_index[quint])
                
    # Initialize colors randomly
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Initialize counts
    red_count = [0] * num_quads
    blue_count = [0] * num_quints
    
    for i, c in enumerate(colors):
        if c == 0: # Red
            for qi in triple_to_quads[i]:
                red_count[qi] += 1
        else: # Blue
            for qi in triple_to_quints[i]:
                blue_count[qi] += 1
                
    # Local Search: Hill Climbing with Random Restarts
    # We look for violations and flip edges to resolve them.
    
    max_iters = 20000
    for it in range(max_iters):
        # Check for Red Violation (4-sets with 4 red edges)
        # We can sample or scan. Scanning is O(num_quads) which is ~50k for n=34.
        # 20k iters * 50k = 1e9 ops, might be too slow in Python.
        # Optimization: Maintain a list of "active" violating quads/quints?
        # Or, just pick a random quad/quint and check?
        # Better: Maintain a set of violating indices.
        
        # Let's use a simpler approach: Pick a random edge, flip it, and see if it improves "total violations".
        # Total violations = sum(1 for rc in red_count if rc==4) + sum(1 for bc in blue_count if bc==10)
        
        # To make it faster, we only check violations for the quads/quints affected by the flip.
        
        def get_violation_count():
            v = 0
            for rc in red_count:
                if rc == 4: v += 1
            for bc in blue_count:
                if bc == 10: v += 1
            return v
            
        current_violations = get_violation_count()
        
        if current_violations == 0:
            # Verify
            if all(rc < 4 for rc in red_count) and all(bc < 10 for bc in blue_count):
                return {'n': n, 'colors': colors}
                
        # Try to flip an edge to reduce violations
        # Pick a random edge
        i = random.randint(0, num_triples - 1)
        
        old_c = colors[i]
        new_c = 1 - old_c
        
        # Calculate change in violations
        delta_v = 0
        
        # Update red_count for quads
        for qi in triple_to_quads[i]:
            old_rc = red_count[qi]
            if old_c == 0: # was red, now blue -> red_count decreases
                if old_rc == 4: delta_v -= 1 # violation resolved
                new_rc = old_rc - 1
                if new_rc == 4: delta_v += 1 # violation created (impossible if old was 4 and we decremented)
                # Actually:
                # if old_c == 0 (red), red_count[qi] decreases by 1.
                # if old_rc == 4, it becomes 3 (good, -1 violation).
                # if old_rc == 3, it becomes 2 (no change).
                # if old_rc == 5, it becomes 4 (bad, +1 violation).
                
                if old_rc == 4:
                    delta_v -= 1
                elif old_rc == 5:
                    delta_v += 1
                red_count[qi] = new_rc
            else: # was blue, now red -> red_count increases
                if old_rc == 3:
                    delta_v += 1 # becomes 4
                # if old_rc == 4, becomes 5 (no change in violation count, still >=4? No, violation is exactly 4? 
                # "No 4 vertices may have all four triples red". This means red_count == 4 is bad.
                # red_count == 5 is also bad? No, if 5 vertices are all red, it contains 5 choose 3 = 10 red triples?
                # The constraint is on 4-sets. A 4-set has exactly 4 triples.
                # So red_count[qi] can only be 0, 1, 2, 3, 4.
                # So old_rc == 4 -> new_rc == 5 is impossible because a 4-set only has 4 triples.
                # Wait, red_count[qi] is the number of red triples in the 4-set. Max is 4.
                # So if old_rc == 4, we can't increase it.
                # So if old_c == 1 (blue), and we flip to red, red_count increases.
                # If old_rc == 3, new_rc == 4 (violation +1).
                # If old_rc < 3, no violation.
                
                if old_rc == 3:
                    delta_v += 1
                red_count[qi] = old_rc + 1
                
        # Update blue_count for quints
        for qi in triple_to_quints[i]:
            old_bc = blue_count[qi]
            if old_c == 1: # was blue, now red -> blue_count decreases
                if old_bc == 10:
                    delta_v -= 1
                elif old_bc == 11:
                    delta_v += 1
                blue_count[qi] = old_bc - 1
            else: # was red, now blue -> blue_count increases
                if old_bc == 9:
                    delta_v += 1
                blue_count[qi] = old_bc + 1
                
        colors[i] = new_c
        
        # If delta_v < 0, we improved.
        # If delta_v > 0, we might accept with some probability (Simulated Annealing) or reject.
        # For simplicity, let's use Hill Climbing: accept if delta_v <= 0.
        # If we get stuck, we can add random kicks.
        
        if delta_v > 0:
            # Reject: revert
            colors[i] = old_c
            # Revert counts
            for qi in triple_to_quads[i]:
                if old_c == 0:
                    red_count[qi] += 1
                else:
                    red_count[qi] -= 1
            for qi in triple_to_quints[i]:
                if old_c == 1:
                    blue_count[qi] += 1
                else:
                    blue_count[qi] -= 1
            
            # If we reject too often, we might be stuck in local minimum.
            # We can just restart.
            if it > 1000:
                return None # Force restart
            
        # Periodic check for zero violations
        if it % 100 == 0:
            if get_violation_count() == 0:
                return {'n': n, 'colors': colors}
                
    return None

# EVOLVE-BLOCK-END

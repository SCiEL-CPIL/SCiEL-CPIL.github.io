import time
import sys
from math import ceil

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin vs 90s wall-clock

    best_result = None
    best_span = MAXV + 1

    # Strategy: For each divisor, start at its smallest multiple >= anchor.
    # Anchor is a candidate minimum. We try anchors that are multiples of
    # each divisor, then resolve conflicts by incrementing.
    
    # Key structural change: Instead of scanning all L in [1, max_L],
    # we only consider anchors that are "interesting" - multiples of divisors
    # or just below them. This dramatically reduces the search space.
    
    # Phase 1: Try small spans by targeting specific anchor points
    # For each divisor d, the multiples near L are at ceil(L/d)*d.
    # The "interesting" L values are those where some divisor's multiple
    # pattern changes, i.e., L is near a multiple of some d.
    
    # We'll try anchors L from 1 up, but jump smartly.
    # For a given L, compute initial assignment: for each divisor d_i,
    # rep_i = smallest multiple of d_i >= L.
    # If all distinct, we have a valid solution with span = max(reps) - min(reps).
    # If not, we can try to resolve by incrementing conflicting reps.
    
    # More efficient: iterate L, but skip ranges where no new multiple appears.
    # Between consecutive multiples of the LCM of all divisors, the pattern
    # of available multiples repeats. But LCM is huge, so instead:
    
    # For each L, the set of candidates for divisor d is {k*d : k*d in [L, L+S]}
    # for some S. The structure changes only when L passes a multiple of some d.
    
    # Structural approach: Use local search.
    # 1. Start with L=1, assign each divisor its smallest multiple >= L.
    # 2. If conflicts exist, increment the conflicting one to its next multiple.
    # 3. Track the resulting span.
    # 4. Move L forward to the next "interesting" point (next multiple of any divisor).
    
    # Even better: For each possible minimum value m (which must be a multiple
    # of at least one divisor), try to assign all divisors to distinct multiples
    # in [m, m+S] for minimal S.
    
    # Let's try: for each candidate minimum m (multiples of divisors up to some bound),
    # greedily assign each divisor to its smallest unused multiple >= m.
    # This gives a valid assignment if all 10 can be assigned.
    # The span is max(assigned) - m.
    
    # To find good m values: m should be a multiple of at least one divisor.
    # We'll iterate m from 1 upward, but jump to next multiple of any divisor.
    
    # Precompute: for each divisor d, the multiples are d, 2d, 3d, ...
    # A "change point" is any multiple of any divisor.
    
    # For efficiency, we'll scan m from 1 to some limit, but only at points
    # where at least one divisor has a multiple at or near m.
    
    # Actually, the simplest effective approach:
    # For m in range(1, limit):
    #   For each divisor d, find smallest multiple of d >= m that's not used.
    #   If all 10 assigned, record span.
    #   But this is O(limit * N * avg_candidates) which could be slow.
    
    # Optimization: The assignment for m is deterministic (greedy smallest unused).
    # As m increases by 1, most assignments stay the same; only divisors whose
    # current multiple equals m might need reassignment.
    
    # Let me implement a smarter incremental approach:
    
    # For a given m, assign[d] = smallest multiple of d >= m that is not assigned to any other divisor.
    # We process divisors in order of their current assignment (ascending) to minimize span.
    
    # Actually, let's just do a direct search but with early termination and
    # focusing on promising m values.
    
    # Key insight from findings: "Targeted search around multiples of each divisor
    # with conflict resolution via incrementing finds tighter windows than uniform scan"
    
    # Implementation:
    # 1. Generate candidate anchor points: all multiples of each divisor up to some bound.
    # 2. For each anchor m (sorted), try to build a valid assignment.
    # 3. Use greedy with backtracking for conflict resolution.
    
    # To keep it fast, limit the number of anchors and the search depth.
    
    # Generate candidate anchors: multiples of each divisor in [1, 2000]
    # This gives us many "interesting" starting points.
    
    anchor_set = set()
    for d in DIVISORS:
        k = 1
        while k * d <= 2000:
            anchor_set.add(k * d)
            k += 1
    # Also add 1 and some small values
    for i in range(1, 50):
        anchor_set.add(i)
    
    anchors = sorted(anchor_set)
    
    # For each anchor, try to find a valid assignment
    for m in anchors:
        if time.time() > deadline:
            break
            
        # Greedy assignment: for each divisor, pick smallest multiple >= m not used
        # Process divisors in order of their smallest multiple >= m (ascending)
        # to keep values tight.
        
        # First pass: compute smallest multiple >= m for each divisor
        initial = []
        for d in DIVISORS:
            k = ceil(m / d)
            val = k * d
            if val > MAXV:
                initial = None
                break
            initial.append((val, d))
        
        if initial is None:
            continue
        
        # Sort by value to process smallest first
        initial.sort()
        
        used = set()
        assignment = {}
        valid = True
        
        for val, d in initial:
            # Find smallest multiple of d >= val not in used
            v = val
            while v in used and v <= MAXV:
                v += d
            if v > MAXV:
                valid = False
                break
            used.add(v)
            assignment[d] = v
        
        if not valid:
            continue
        
        # Check all assigned
        if len(assignment) != N:
            continue
        
        reps = [assignment[d] for d in DIVISORS]
        span = max(reps) - min(reps)
        
        if span < best_span:
            best_span = span
            best_result = reps[:]
            
            # Early termination if we find span=1 (impossible due to distinctness)
            # or span=2 (very good)
            if span <= 2:
                break
    
    # Phase 2: If we have a result, try to improve with local search
    if best_result is not None and time.time() < deadline - 5:
        # Try to refine by trying nearby anchors
        current_min = min(best_result)
        # Try anchors in [current_min - 20, current_min + 20]
        for dm in range(-20, 21):
            if time.time() > deadline:
                break
            m = current_min + dm
            if m < 1 or m > MAXV:
                continue
            
            initial = []
            for d in DIVISORS:
                k = ceil(m / d)
                val = k * d
                if val > MAXV:
                    initial = None
                    break
                initial.append((val, d))
            
            if initial is None:
                continue
            
            initial.sort()
            
            used = set()
            assignment = {}
            valid = True
            
            for val, d in initial:
                v = val
                while v in used and v <= MAXV:
                    v += d
                if v > MAXV:
                    valid = False
                    break
                used.add(v)
                assignment[d] = v
            
            if not valid or len(assignment) != N:
                continue
            
            reps = [assignment[d] for d in DIVISORS]
            span = max(reps) - min(reps)
            
            if span < best_span:
                best_span = span
                best_result = reps[:]
    
    # Phase 3: Further refinement with more systematic search if time permits
    if best_result is not None and time.time() < deadline - 10:
        # Try a wider range of anchors around the best minimum
        current_min = min(best_result)
        for m in range(max(1, current_min - 100), min(MAXV, current_min + 100) + 1):
            if time.time() > deadline:
                break
            
            initial = []
            for d in DIVISORS:
                k = ceil(m / d)
                val = k * d
                if val > MAXV:
                    initial = None
                    break
                initial.append((val, d))
            
            if initial is None:
                continue
            
            initial.sort()
            
            used = set()
            assignment = {}
            valid = True
            
            for val, d in initial:
                v = val
                while v in used and v <= MAXV:
                    v += d
                if v > MAXV:
                    valid = False
                    break
                used.add(v)
                assignment[d] = v
            
            if not valid or len(assignment) != N:
                continue
            
            reps = [assignment[d] for d in DIVISORS]
            span = max(reps) - min(reps)
            
            if span < best_span:
                best_span = span
                best_result = reps[:]
    
    if best_result is None:
        # Fallback: return divisors themselves
        return DIVISORS[:]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]

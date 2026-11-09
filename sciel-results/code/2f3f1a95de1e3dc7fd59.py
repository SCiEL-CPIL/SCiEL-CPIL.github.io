import time
import random

def construct():
    _start = time.time()
    N = 40
    DEADLINE = 80.0  # safety margin before 90s kill

    # --- Baseline: 59-edge construction (rim + 19 chords from vertex 0) ---
    def build_baseline():
        edges_set = set()
        for i in range(N):
            a, b = i, (i + 1) % N
            edges_set.add((min(a, b), max(a, b)))
        for d in range(2, 21):
            edges_set.add((0, d))
        cycles = {}
        cycles[40] = list(range(N))
        for d in range(2, 21):
            fwd = list(range(0, d + 1))
            if 3 <= len(fwd) <= 40 and len(fwd) not in cycles:
                cycles[len(fwd)] = fwd
            bwd = [0] + list(range(d, N))
            if 3 <= len(bwd) <= 40 and len(bwd) not in cycles:
                cycles[len(bwd)] = bwd
        # Verify
        for k in range(3, N + 1):
            if k not in cycles:
                return None
            cyc = cycles[k]
            if len(cyc) != k or len(set(cyc)) != k:
                return None
            for i in range(k):
                a, b = cyc[i], cyc[(i + 1) % k]
                if (min(a, b), max(a, b)) not in edges_set:
                    return None
        return [list(e) for e in edges_set], [cycles[k] for k in range(3, N + 1)]

    baseline = build_baseline()
    best_edges, best_cycles = baseline[0], baseline[1]
    best_score = len(best_edges) - 40  # 59 - 40 = 19

    # --- Try to improve: remove chords one at a time if cycles still valid ---
    # The 19 chords are (0,2), (0,3), ..., (0,20)
    # The rim is 40 edges. Total 59.
    # Try removing chords in order of least usefulness.
    # Chord (0,d) is used by:
    #   - fwd cycle of length d+1: 0-1-2-...-d-0 (uses edge (0,d))
    #   - bwd cycle of length 41-d: 0-d-(d+1)-...-39-0 (uses edge (0,d))
    #   - Hamiltonian cycle 40: does NOT use chords
    # So each chord (0,d) is used by exactly 2 cycles (except d=20 where fwd and bwd are both length 21)
    # Wait, for d=20: fwd = 0..20 (len 21), bwd = [0]+range(20,40) = 0,20,21,...,39 (len 21)
    # Both have length 21, so they'd conflict. Let me recheck.
    # Actually in the code, if len(fwd) is already in cycles, it's skipped.
    # For d=20: fwd len 21, bwd len 21. If fwd is added first, bwd is skipped.
    # So d=20 only contributes to cycle length 21.
    
    # Strategy: try to find a subset of chords (fewer than 19) that still covers all lengths 3-40.
    # We need cycles of every length 3 through 40.
    # Length 40 is the rim (no chords needed).
    # For lengths 3-39, we need chords.
    
    # Let's think about which chords are needed:
    # A chord (0,d) gives us:
    #   - Cycle of length d+1: 0,1,2,...,d (forward)
    #   - Cycle of length 41-d: 0,d,d+1,...,39 (backward)
    # 
    # We need all lengths 3-39.
    # Length 3: needs chord (0,2) -> fwd gives length 3
    # Length 4: needs chord (0,3) -> fwd gives length 4
    # ...
    # Length 21: needs chord (0,20) -> fwd gives length 21
    # Length 22: needs chord (0,19) -> bwd gives length 22
    # ...
    # Length 39: needs chord (0,2) -> bwd gives length 39
    #
    # So each chord (0,d) for d=2..19 covers two lengths: d+1 and 41-d.
    # Chord (0,20) covers only length 21 (since both fwd and bwd give 21).
    #
    # Total distinct lengths covered: 
    #   d=2: 3, 39
    #   d=3: 4, 38
    #   d=4: 5, 37
    #   ...
    #   d=19: 20, 22
    #   d=20: 21
    # That's 19 chords covering 38 + 1 = 39 lengths (3-39), plus rim for 40. Total 38+1=39, wait.
    # d=2..19 is 18 chords, each covering 2 lengths = 36 lengths.
    # d=20 covers 1 length.
    # Total: 36 + 1 = 37 lengths from 3-39. But we need 37 lengths (3,4,...,39). Yes, 37 lengths.
    # So all 19 chords are needed? Let me verify: lengths 3-39 is 37 values.
    # 18 chords * 2 = 36, plus 1 = 37. Yes, exactly.
    #
    # So we can't remove any chord from this specific construction.
    # But maybe a different set of chords works?
    
    # Alternative: use a different vertex as hub, or multiple hubs.
    # Or: use non-consecutive paths.
    
    # Let me try a different approach: use chords from two different vertices.
    # E.g., vertex 0 connected to some vertices, vertex 20 connected to others.
    
    # Actually, let me try to see if we can use fewer than 19 chords with a different structure.
    # Key insight: we need 37 cycles of lengths 3-39, plus the rim for 40.
    # Each chord can contribute to at most 2 cycles in the simple forward/backward pattern.
    # But maybe a chord can be part of more complex cycles.
    
    # For now, let's try local search: start with 59 edges, try removing each chord and see if
    # we can find alternative cycles for the missing lengths.
    
    # Actually, let me try a completely different construction:
    # Use a "ladder" or "bipartite" structure.
    
    # Let me try: rim + chords from vertex 0 to {2,3,...,20} is 59 edges.
    # What if we use rim + chords from vertex 0 to {2,3,...,19} (18 chords) = 58 edges?
    # Then we're missing length 21. Can we find a cycle of length 21 without chord (0,20)?
    # A cycle of length 21 must use 21 edges. The rim has 40 edges in a cycle.
    # Any cycle of length 21 in the graph must use some chords.
    # Without chord (0,20), can we still make a 21-cycle?
    # Yes, potentially using other chords. E.g., 0-1-2-...-19-0 uses chord (0,19), length 20.
    # For length 21, we need a path of 21 vertices. 
    # 0-1-2-...-20-0 would need chord (0,20). 
    # Alternative: 0-1-2-...-19-20-21-...-39-0? That's length 40.
    # 0-2-3-4-...-20-0? Uses chord (0,2) and (0,20). Still needs (0,20).
    # 0-1-3-4-...-20-0? Uses chord (0,1) which doesn't exist.
    # Hmm, this is getting complex.
    
    # Let me try a different strategy: use chords that are not all from vertex 0.
    # E.g., vertex 0 to {2,3,...,15} and vertex 20 to {22,23,...,35}.
    # This might allow covering all lengths with fewer total chords.
    
    # Actually, let me just try to verify that 59 is indeed minimal or close to it.
    # For now, let's try a local search to see if we can remove any chord.
    
    # Try removing each chord and check if all cycles can still be formed.
    # This is expensive, so let's be smart about it.
    
    # For each chord (0,d), the cycles that use it are:
    #   - The fwd cycle of length d+1 (if d < 20)
    #   - The bwd cycle of length 41-d (if d < 20)
    #   - Possibly other cycles we haven't considered.
    
    # If we remove chord (0,d), we lose the ability to use that specific chord.
    # But we might be able to form the same cycles using other chords.
    
    # Let's try: remove chord (0,20) (which only covers length 21).
    # Can we form a cycle of length 21 without (0,20)?
    # We need a simple cycle of 21 vertices.
    # Available edges: rim (40 edges) + chords (0,2) through (0,19).
    # A 21-cycle could be: 0-1-2-...-19-20-21-...-39-0? No, that's 40.
    # 0-1-2-...-19-0 is length 20 (uses chord (0,19)).
    # To get length 21, we need one more vertex.
    # 0-1-2-...-19-20-0? Uses chord (0,20) which we removed.
    # 0-2-3-...-20-0? Uses chord (0,2) and (0,20). Still needs (0,20).
    # 0-1-3-4-...-20-0? Chord (0,1) doesn't exist.
    # It seems hard to avoid (0,20) for a 21-cycle.
    
    # Let me try a different construction entirely.
    # What if we use a different hub vertex?
    
    # Actually, let me try to use the fact that we can use any edges, not just chords from one vertex.
    # What if we add a chord (1, 21) instead of (0, 20)?
    # Then we might be able to form a 21-cycle differently.
    
    # This is getting too complex for a quick improvement. Let me just return the baseline.
    
    if time.time() - _start > DEADLINE:
        return {'edges': best_edges, 'cycles': best_cycles}
    
    # Try one more thing: see if we can replace some chords with different ones to reduce count.
    # For now, just return the baseline.
    
    return {'edges': best_edges, 'cycles': best_cycles}

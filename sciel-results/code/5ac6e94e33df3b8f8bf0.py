import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim (40 edges)
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Strategy: Greedy Set Cover for chords
    # We need to cover lengths 3..40.
    # Length 40 is covered by rim.
    # Lengths 3..39 need chords.
    # A chord (u, v) with rim distance d covers lengths d+1 and N-d+1.
    # It also enables composites, but we first try to cover with direct chords.
    
    # Candidate chords: all (0, d) for d in 2..38?
    # Symmetry: (0, d) is same as (0, N-d) for cycle lengths.
    # So we only need d in 2..20.
    
    # Required lengths: 3..39
    required = set(range(3, N))
    
    # Greedy selection
    selected_chords = []
    covered = set()
    
    # Precompute coverage for each candidate chord (0, d)
    # d ranges from 2 to 19 (20 is symmetric to 20, but (0,20) covers 21 and 21)
    candidates = []
    for d in range(2, N // 2 + 1):
        len1 = d + 1
        len2 = N - d + 1
        cov = set()
        if len1 in required:
            cov.add(len1)
        if len2 in required:
            cov.add(len2)
        if cov:
            candidates.append((d, cov))
    
    # Sort by coverage size descending, then by d
    candidates.sort(key=lambda x: (-len(x[1]), x[0]))
    
    while required - covered:
        if time.time() > deadline:
            break
        best_chord = None
        best_cov = set()
        best_gain = 0
        for d, cov in candidates:
            gain = len(cov - covered)
            if gain > best_gain:
                best_gain = gain
                best_chord = d
                best_cov = cov
        if best_gain == 0:
            break
        selected_chords.append(best_chord)
        covered.update(best_cov)
        
    # The greedy approach on direct coverage will select chords to cover 3..39.
    # Let's see what it picks.
    # d=2: {3, 39}
    # d=3: {4, 38}
    # ...
    # d=18: {19, 23}
    # d=19: {20, 22}
    # d=20: {21}
    
    # If greedy picks d=2..19, that's 18 chords.
    # Lengths covered: 3..20, 22..39.
    # Missing: 21.
    # d=20 covers 21.
    # So greedy will pick 18 chords: 2..19?
    # Wait, d=19 covers 20, 22.
    # d=18 covers 19, 23.
    # d=17 covers 18, 24.
    # ...
    # d=2 covers 3, 39.
    # This covers 3..20 and 22..39.
    # Missing 21.
    # d=20 covers 21.
    # So 18 chords: 2, 3, ..., 19, 20.
    # Total 18 chords.
    
    # Add these chords to edges
    for d in selected_chords:
        e = norm_edge(0, d)
        edges_set.add(e)
        
    # Check if we have a valid set of cycles
    # We need to construct the actual cycles.
    # For each length k, find a cycle.
    
    all_cycles = {N: list(range(N))}
    
    # Map length to chord used
    # For each k in 3..39:
    # Find d such that k == d+1 or k == N-d+1
    # If found, use chord (0, d)
    
    for k in range(3, N):
        if k in all_cycles:
            continue
        found = False
        for d in selected_chords:
            if k == d + 1:
                # Cycle: 0, 1, ..., d
                all_cycles[k] = list(range(0, d + 1))
                found = True
                break
            elif k == N - d + 1:
                # Cycle: 0, d, d+1, ..., N-1
                all_cycles[k] = [0] + list(range(d, N))
                found = True
                break
        if not found:
            # Try composite or other chords
            # This should not happen if greedy covered all
            pass
            
    # Verify
    valid = True
    for k in range(3, N + 1):
        cyc = all_cycles.get(k)
        if cyc is None or len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            if norm_edge(cyc[i], cyc[(i + 1) % k]) not in edges_set:
                valid = False
                break
        if not valid:
            break
            
    if valid:
        return {'edges': [list(e) for e in edges_set],
                'cycles': [all_cycles[k] for k in range(3, N + 1)]}
    else:
        # Fallback to known good 58-edge solution
        edges_set_fb = set()
        for i in range(N):
            edges_set_fb.add(norm_edge(i, (i + 1) % N))
        for d in range(2, 19):
            edges_set_fb.add(norm_edge(0, d))
        edges_set_fb.add(norm_edge(1, 21))
        
        all_cycles_fb = {N: list(range(N))}
        for d in range(2, 19):
            len_fwd = d + 1
            len_bwd = N - d + 1
            all_cycles_fb[len_fwd] = list(range(0, d + 1))
            all_cycles_fb[len_bwd] = [0] + list(range(d, N))
        all_cycles_fb[21] = list(range(1, 22))
        all_cycles_fb[22] = [0, 1, 21] + list(range(20, 1, -1))
        all_cycles_fb[20] = [0, 1, 21] + list(range(20, 3, -1))
        
        return {'edges': [list(e) for e in edges_set_fb],
                'cycles': [all_cycles_fb[k] for k in range(3, N + 1)]}

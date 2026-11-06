import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    
    # Build a sparse-like representation for fast delta computation
    # For a given bit flip at position i, the change in objective depends on 
    # how many 4-APs contain i and the current state of the other 3 bits in those APs.
    
    # Precompute for each position i, the list of (ap_index, other_positions)
    # Actually, we can compute the delta for flipping bit i as:
    # delta = sum over all APs containing i of (new_contribution - old_contribution)
    # A 4-AP (i1, i2, i3, i4) is monochromatic if all 4 bits are equal.
    # If we flip bit at position p in the AP, the AP becomes non-monochromatic if 
    # before the flip all 4 were equal, and after the flip they are not (which is always true if we flip one bit in a monochromatic AP).
    # Conversely, if before the flip the AP was not monochromatic, flipping one bit might make it monochromatic only if the other 3 bits are equal to the new value.
    
    # Let's define:
    # For each position p, and for each AP containing p, let the other 3 positions be q1, q2, q3.
    # Current bit at p is x_p. Other bits are x_q1, x_q2, x_q3.
    # If x_q1 == x_q2 == x_q3 == x_p: AP is mono. After flip, not mono. Delta = -1.
    # If x_q1 == x_q2 == x_q3 != x_p: AP is not mono. After flip, x_p becomes equal to others, so mono. Delta = +1.
    # Otherwise: AP is not mono, and after flip it's still not mono (since the 3 others are not all equal, or if they are, they are not equal to the new x_p... wait).
    # Actually, if x_q1 == x_q2 == x_q3 = v, and x_p != v, then flipping x_p to v makes it mono. Delta = +1.
    # If x_q1 == x_q2 == x_q3 = v, and x_p == v, then flipping x_p to 1-v makes it not mono. Delta = -1.
    # If x_q1, x_q2, x_q3 are not all equal, then no matter what x_p is, the AP is not mono, and flipping x_p keeps it not mono. Delta = 0.
    
    # So for each position p, we can precompute:
    # count_match[p]: number of APs containing p where the other 3 bits are all equal to x_p.
    # count_mismatch[p]: number of APs containing p where the other 3 bits are all equal to 1-x_p.
    # Then delta = -count_match[p] + count_mismatch[p].
    
    # But this count changes as we flip bits. So we need to update these counts efficiently.
    # Alternatively, we can just recompute the delta on the fly for each candidate flip, but that might be slow.
    
    # Let's try a simpler approach: for each position p, precompute the list of APs containing p.
    # Then for a given coloring, the delta for flipping p is:
    #   sum over APs containing p of:
    #     -1 if all 4 bits in AP are equal (before flip)
    #     +1 if the other 3 bits are equal to 1-x_p (before flip)
    #     0 otherwise
    
    # We can compute this in O(number of APs containing p) per flip.
    # Average number of APs per position is num_aps / N. num_aps is roughly N^2/6 for 4-APs? Let's estimate.
    # For d=1, a can be 1..253, so 253 APs. For d=2, 250, etc. Sum_{d=1}^{85} (256 - 3d) = 85*256 - 3*(85*86/2) = 21760 - 3*3655 = 21760 - 10965 = 10795.
    # So num_aps ~ 10800. Average per position ~ 10800/256 ~ 42.
    # So each flip evaluation is O(42). A full pass over 256 positions is O(256*42) ~ 10752 operations.
    # This is very fast. We can do many passes.
    
    # Let's implement coordinate descent:
    # Start with a random coloring or a simple heuristic.
    # Then repeatedly scan all positions, and flip the one with the most negative delta (or any negative delta).
    # Stop when no more improvements or time limit.
    
    # Initialize coloring with alternating blocks of 2 (like the seed) or random.
    # Let's try a few initializations.
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for (i1, i2, i3, i4) in aps:
            if coloring[i1] == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count
    
    def delta_for_flip(coloring, p):
        """Compute the change in objective if we flip bit at position p."""
        # Count how many APs containing p are currently mono (will become non-mono)
        # and how many have the other 3 bits equal to 1-coloring[p] (will become mono)
        dec = 0
        inc = 0
        for (i1, i2, i3, i4) in aps:
            # Check if p is in this AP
            # We need a faster way to find APs containing p. Let's precompute.
            pass
        # This is too slow. Let's precompute the list of APs for each position.
        return 0  # Placeholder
    
    # Precompute for each position p, the list of APs containing p.
    aps_by_pos = [[] for _ in range(N)]
    for idx, (i1, i2, i3, i4) in enumerate(aps):
        aps_by_pos[i1].append((idx, i2, i3, i4))
        aps_by_pos[i2].append((idx, i1, i3, i4))
        aps_by_pos[i3].append((idx, i1, i2, i4))
        aps_by_pos[i4].append((idx, i1, i2, i3))
    
    def delta_for_flip(coloring, p):
        """Compute the change in objective if we flip bit at position p."""
        dec = 0
        inc = 0
        for (idx, q1, q2, q3) in aps_by_pos[p]:
            if coloring[q1] == coloring[q2] == coloring[q3]:
                if coloring[q1] == coloring[p]:
                    dec += 1  # Was mono, will become non-mono
                else:
                    inc += 1  # Was non-mono, will become mono
        return -dec + inc
    
    # Coordinate descent
    best_coloring = None
    best_score = float('inf')
    
    # Try a few initializations
    initializations = []
    # 1. Alternating blocks of 2
    initializations.append([(i // 2) % 2 for i in range(N)])
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializations.append(rng.integers(0, 2, N).tolist())
    # 3. Thue-Morse like: (i & (i >> 1)).bit_count() % 2
    initializations.append([(i & (i >> 1)).bit_count() % 2 for i in range(N)])
    
    for init in initializations:
        coloring = init[:]
        score = count_mono(coloring)
        if score < best_score:
            best_score = score
            best_coloring = coloring[:]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_pos = -1
            for p in range(N):
                d = delta_for_flip(coloring, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            if best_pos != -1 and best_delta < 0:
                coloring[best_pos] = 1 - coloring[best_pos]
                score += best_delta
                improved = True
                if score < best_score:
                    best_score = score
                    best_coloring = coloring[:]
    
    return best_coloring

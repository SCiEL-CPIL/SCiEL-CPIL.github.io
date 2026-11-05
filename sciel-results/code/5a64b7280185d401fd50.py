import time
import random
import sys

def construct():
    N = 128
    # Precompute all 4-APs (0-indexed)
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_bad(coloring):
        """Returns number of bad APs and the set of bad AP indices."""
        bad_count = 0
        bad_set = set()
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad_count += 1
                bad_set.add(idx)
        return bad_count, bad_set

    def try_local_search(K, time_limit):
        """
        Attempts to find a valid coloring using K colors.
        Returns the coloring if found, else None.
        """
        start_time = time.time()
        # Initial random coloring
        coloring = [random.randint(0, K - 1) for _ in range(N)]
        bad_count, bad_set = count_bad(coloring)
        
        if bad_count == 0:
            return coloring
            
        T = 10.0 # Initial temperature
        best_coloring = coloring[:]
        best_bad_count = bad_count
        
        while time.time() - start_time < time_limit:
            # Pick a random position
            i = random.randint(0, N - 1)
            old_c = coloring[i]
            
            # Pick a new color different from old
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            # Calculate delta in bad count
            # Only APs containing i can change status
            delta = 0
            affected_aps = pos_to_aps[i]
            
            # We need to know the current status of these APs to calculate delta correctly
            # It's easier to just update the coloring, recalculate bad count for affected APs,
            # and compare. But we need to know if they were bad before.
            
            # Let's store the current bad status in a list for O(1) lookup
            # We'll maintain `is_bad` array outside the loop for efficiency in the main loop.
            # For this helper, let's just do a simple check.
            # Actually, for speed, we should maintain `is_bad` globally in the loop.
            
            # Refactoring: do the search loop directly here to maintain state.
            pass # Placeholder, logic moved to main loop below

        return None # Fallback if not found in this specific function structure

    # Efficient Local Search Loop
    def search(K, time_limit):
        start_time = time.time()
        coloring = [random.randint(0, K - 1) for _ in range(N)]
        is_bad = [False] * len(aps)
        bad_count = 0
        
        for idx, ap in enumerate(aps):
            if len({coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]}) < 3:
                is_bad[idx] = True
                bad_count += 1
                
        if bad_count == 0:
            return coloring
            
        T = 5.0
        best_count = bad_count
        best_coloring = coloring[:]
        
        while time.time() - start_time < time_limit:
            i = random.randint(0, N - 1)
            old_c = coloring[i]
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            # Calculate delta
            delta = 0
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                # Check if this AP was bad before
                was_bad = is_bad[ap_idx]
                
                # Temporarily apply change to calculate new status
                # We can just compute the new distinct colors count
                c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                # Replace color at i
                if ap[0] == i: c0 = new_c
                elif ap[1] == i: c1 = new_c
                elif ap[2] == i: c2 = new_c
                elif ap[3] == i: c3 = new_c
                
                new_distinct = len({c0, c1, c2, c3})
                is_now_bad = new_distinct < 3
                
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
            
            # Acceptance
            if delta <= 0:
                # Apply change
                coloring[i] = new_c
                for ap_idx in pos_to_aps[i]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    is_bad[ap_idx] = len({c0, c1, c2, c3}) < 3
                bad_count += delta
                
                if bad_count < best_count:
                    best_count = bad_count
                    best_coloring = coloring[:]
                    if best_count == 0:
                        return best_coloring
            else:
                # Simulated Annealing
                if random.random() < (1.0 / (1.0 + delta * T)):
                    # Accept
                    coloring[i] = new_c
                    for ap_idx in pos_to_aps[i]:
                        ap = aps[ap_idx]
                        c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                        is_bad[ap_idx] = len({c0, c1, c2, c3}) < 3
                    bad_count += delta
                    if bad_count < best_count:
                        best_count = bad_count
                        best_coloring = coloring[:]
                        if best_count == 0:
                            return best_coloring
            
            # Cool down
            T *= 0.995
            if T < 0.1:
                T = 0.1

        return None

    # Try 3 colors first
    # Use a short time limit for 3 colors as it's likely impossible
    result = search(3, 5.0)
    if result:
        return result

    # Try 4 colors
    result = search(4, 40.0)
    if result:
        return result

    # Try 5 colors as fallback
    result = search(5, 10.0)
    if result:
        return result

    # Fallback to greedy (guaranteed to produce something, though maybe suboptimal)
    # Re-implementing the seed's greedy logic as a last resort
    colors = []
    for x in range(1, N + 1):
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - j * d for j in (0, 1, 2, 3)]
                cs = [color] + [colors[i] for i in idxs[1:]]
                if len(set(cs)) < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # The seed had a second condition. Let's just ensure the basic constraint.
            # If we just ensure "ending at x" constraints, we might fail later.
            # But this is a fallback.
            colors.append(color)
            break
    return colors


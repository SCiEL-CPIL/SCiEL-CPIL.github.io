"""Combined approach: Randomized repair and local search for 4-AP colorings."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    N = 128

    # Precompute all non-constant 4-term APs
    ap_list = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b, c, e = a + d, a + 2 * d, a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))

    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)

    def count_violations(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        count = 0
        for a, b, c, e in ap_list:
            ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
            # Check if distinct count < 3
            if ca == cb:
                if cc == ce:
                    count += 1
                else:
                    if ca == cc or ca == ce:
                        count += 1
            else:
                if cc == ce:
                    if cc == ca or cc == cb:
                        count += 1
                else:
                    if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                        count += 1
        return count

    def local_search(K, time_limit):
        """
        Attempts to find a valid coloring with K colors using local search.
        Returns the valid coloring if found, else None.
        """
        start_time = time.time()
        # Initialize with random colors
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors

        # Simulated Annealing / Hill Climbing
        # We will try to minimize violations.
        # If we get stuck, we might restart or increase K.
        
        # To speed up delta calculation:
        # When changing color of pos i from old_c to new_c:
        # Only APs containing i are affected.
        
        def get_violation_delta(pos, new_color):
            """Calculate change in violations if colors[pos] becomes new_color."""
            old_color = colors[pos]
            if old_color == new_color:
                return 0
            
            delta = 0
            for ap_idx in pos_aps[pos]:
                a, b, c, e = ap_list[ap_idx]
                ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
                
                # Check if this AP is currently a violation
                is_viol_now = False
                if ca == cb:
                    if cc == ce:
                        is_viol_now = True
                    else:
                        if ca == cc or ca == ce:
                            is_viol_now = True
                else:
                    if cc == ce:
                        if cc == ca or cc == cb:
                            is_viol_now = True
                    else:
                        if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                            is_viol_now = True
                
                # Update the specific position's color for the check
                if a == pos:
                    ca = new_color
                elif b == pos:
                    cb = new_color
                elif c == pos:
                    cc = new_color
                elif e == pos:
                    ce = new_color
                
                # Check if this AP is a violation with the new color
                is_viol_new = False
                if ca == cb:
                    if cc == ce:
                        is_viol_new = True
                    else:
                        if ca == cc or ca == ce:
                            is_viol_new = True
                else:
                    if cc == ce:
                        if cc == ca or cc == cb:
                            is_viol_new = True
                    else:
                        if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                            is_viol_new = True
                
                if is_viol_now and not is_viol_new:
                    delta -= 1
                elif not is_viol_now and is_viol_new:
                    delta += 1
            
            return delta

        # Main Loop
        # We use a simple best-improvement hill climbing with random restarts if stuck.
        # Given the small N, we can do many iterations.
        
        max_iter_per_restart = 5000
        num_restarts = 10
        
        for r in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
            
            # Re-initialize if not first restart
            if r > 0:
                colors = [random.randint(0, K - 1) for _ in range(N)]
                current_violations = count_violations(colors)
                if current_violations == 0:
                    return colors
            
            for i in range(max_iter_per_restart):
                if time.time() > start_time + time_limit:
                    break
                
                if current_violations == 0:
                    return colors
                
                # Pick a random position
                pos = random.randint(0, N - 1)
                
                # Find the best new color for this position
                best_delta = 0
                best_color = colors[pos]
                
                for new_c in range(K):
                    if new_c == colors[pos]:
                        continue
                    d = get_violation_delta(pos, new_c)
                    if d < best_delta:
                        best_delta = d
                        best_color = new_c
                
                # If we found an improvement, take it
                if best_delta < 0:
                    colors[pos] = best_color
                    current_violations += best_delta
                else:
                    # If no improvement found at this pos, we could try a random move to escape local minima
                    # But for now, let's just move to next pos.
                    # To avoid getting stuck in a plateau, we might accept equal moves or random moves.
                    # Let's try a random move with small probability if stuck for a while.
                    # Simpler: just continue. The random selection of pos will eventually find a move.
                    pass

        return None

    # Try to find a solution with minimum colors
    # Start with 3 colors
    for K in [3, 4, 5, 6]:
        # Allocate time based on K
        # We have ~85s total.
        # Give 20s to K=3, 10s to K=4, 5s to K=5, 5s to K=6.
        if K == 3:
            t_limit = 20
        elif K == 4:
            t_limit = 15
        elif K == 5:
            t_limit = 10
        else:
            t_limit = 5
            
        if time.time() > deadline - t_limit:
            continue
            
        result = local_search(K, t_limit)
        if result is not None:
            # Validate just in case
            if count_violations(result) == 0:
                # Relabel to 0..K-1
                mapping = {}
                new_colors = []
                next_id = 0
                for c in result:
                    if c not in mapping:
                        mapping[c] = next_id
                        next_id += 1
                    new_colors.append(mapping[c])
                return new_colors

    # Fallback: If all local searches failed, we must return *something* valid.
    # A safe fallback is a random coloring with many colors, then repair.
    # Or use a known pattern.
    # Let's try a greedy construction with K=10 which is very likely to succeed.
    # Greedy: for i in 0..N-1, pick smallest color c such that no AP ending at i is violated.
    
    colors = [0] * N
    for i in range(N):
        found = False
        for c in range(10):
            colors[i] = c
            ok = True
            for ap_idx in pos_aps[i]:
                a, b, cc, e = ap_list[ap_idx]
                # Only check if all are <= i
                if max(a, b, cc, e) == i:
                    ca, cb, ccc, ce = colors[a], colors[b], colors[cc], colors[e]
                    # Check validity
                    if ca == cb:
                        if ccc == ce:
                            ok = False
                            break
                        else:
                            if ca == ccc or ca == ce:
                                ok = False
                                break
                    else:
                        if ccc == ce:
                            if ccc == ca or ccc == cb:
                                ok = False
                                break
                        else:
                            if (ca == ccc and cb == ce) or (ca == ce and cb == ccc):
                                ok = False
                                break
            if ok:
                found = True
                break
        if not found:
            # Should not happen with 10 colors
            colors[i] = random.randint(0, 9)
            
    # If greedy failed (unlikely), do a quick repair
    if count_violations(colors) > 0:
        for _ in range(1000):
            v_idx = random.randint(0, len(ap_list) - 1)
            a, b, c, e = ap_list[v_idx]
            pos = random.choice([a, b, c, e])
            colors[pos] = random.randint(0, 9)
            if count_violations(colors) == 0:
                break

    # Relabel
    mapping = {}
    new_colors = []
    next_id = 0
    for c in colors:
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        new_colors.append(mapping[c])
    
    return new_colors
# EVOLVE-BLOCK-END

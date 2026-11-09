import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    n = 128
    # Precompute all non-constant four-term progressions
    # (a, a+d, a+2d, a+3d) with 0-based indices
    aps = []
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            aps.append((a, b, c, dd))
    
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # For each position, precompute which APs it appears in and at which slot
    pos_ap_info = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_ap_info[a].append((idx, 0))
        pos_ap_info[b].append((idx, 1))
        pos_ap_info[c].append((idx, 2))
        pos_ap_info[d].append((idx, 3))
    
    # Convert to arrays
    pos_ap_indices = [np.array([x[0] for x in info], dtype=np.int32) for info in pos_ap_info]
    pos_ap_slots = [np.array([x[1] for x in info], dtype=np.int32) for info in pos_ap_info]
    
    def is_bad_ap(colors4):
        """Check if a 4-term progression is bad (uses < 3 distinct colors)."""
        a, b, c, d = colors4
        # Bad if at most 2 distinct colors
        # Check if any three are distinct
        if (a != b) and (a != c) and (b != c):
            return False
        if (a != b) and (a != d) and (b != d):
            return False
        if (a != c) and (a != d) and (c != d):
            return False
        if (b != c) and (b != d) and (c != d):
            return False
        return True
    
    def compute_bad_mask(c):
        """Compute boolean mask of bad progressions from coloring c (list of ints)."""
        bad = np.zeros(n_aps, dtype=bool)
        for idx in range(n_aps):
            a, b, cpos, d = aps_arr[idx]
            if is_bad_ap((c[a], c[b], c[cpos], c[d])):
                bad[idx] = True
        return bad
    
    def compute_bad_count(c):
        """Compute number of bad progressions from coloring c."""
        return int(np.sum(compute_bad_mask(c)))
    
    def try_all_k_colors(K, c, bad_mask, current_bad):
        """Try to improve coloring with K colors using incremental updates.
        Returns (found, c, bad_mask, current_bad)."""
        found = False
        no_improve = 0
        max_no_improve = 2000
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_bad == 0:
                found = True
                break
            
            # Pick a position that's in at least one bad AP, or random
            if current_bad > 0:
                bad_positions = set()
                for idx in np.where(bad_mask)[0]:
                    a, b, cpos, d = aps_arr[idx]
                    bad_positions.add(a)
                    bad_positions.add(b)
                    bad_positions.add(cpos)
                    bad_positions.add(d)
                if bad_positions:
                    pos = random.choice(list(bad_positions))
                else:
                    pos = random.randint(0, n - 1)
            else:
                pos = random.randint(0, n - 1)
            
            old_color = c[pos]
            
            # Try all new colors, compute delta in bad count
            best_new_color = old_color
            best_delta = 0  # delta = new_bad - current_bad, we want most negative
            
            for new_color in range(K):
                if new_color == old_color:
                    continue
                
                # Compute how this change affects each AP containing pos
                # For each AP containing pos, check if it was bad before and after
                delta = 0
                for (ap_idx, slot) in pos_ap_info[pos]:
                    a, b, cpos, d = aps_arr[ap_idx]
                    # Get colors before change
                    old_colors = [c[a], c[b], c[cpos], c[d]]
                    old_colors[slot] = new_color
                    
                    was_bad = bad_mask[ap_idx]
                    is_bad_now = is_bad_ap(tuple(old_colors))
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color == old_color:
                # No improvement, try random move to escape
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                # Apply random move and update bad_mask incrementally
                c[pos] = new_color
                for (ap_idx, slot) in pos_ap_info[pos]:
                    a, b, cpos, d = aps_arr[ap_idx]
                    old_colors = [c[a], c[b], c[cpos], c[d]]
                    is_bad_now = is_bad_ap(tuple(old_colors))
                    was_bad = bad_mask[ap_idx]
                    if was_bad and not is_bad_now:
                        current_bad -= 1
                        bad_mask[ap_idx] = False
                    elif not was_bad and is_bad_now:
                        current_bad += 1
                        bad_mask[ap_idx] = True
                continue
            
            # Apply best move
            c[pos] = best_new_color
            for (ap_idx, slot) in pos_ap_info[pos]:
                a, b, cpos, d = aps_arr[ap_idx]
                old_colors = [c[a], c[b], c[cpos], c[d]]
                is_bad_now = is_bad_ap(tuple(old_colors))
                was_bad = bad_mask[ap_idx]
                if was_bad and not is_bad_now:
                    current_bad -= 1
                    bad_mask[ap_idx] = False
                elif not was_bad and is_bad_now:
                    current_bad += 1
                    bad_mask[ap_idx] = True
            
            no_improve = 0
        
        return found, c, bad_mask, current_bad
    
    # Try to find a valid coloring with K colors
    for K in range(4, 10):
        if time.time() > deadline:
            break
        
        found = False
        best_bad_count = float('inf')
        best_coloring = None
        
        # Multiple restarts for local search
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Initialize random coloring
            c = [random.randint(0, K - 1) for _ in range(n)]
            
            # Compute initial bad mask and count
            bad_mask = compute_bad_mask(c)
            current_bad = int(np.sum(bad_mask))
            
            if current_bad == 0:
                found = True
                break
            
            # Local search
            found, c, bad_mask, current_bad = try_all_k_colors(K, c, bad_mask, current_bad)
            
            if found:
                break
            
            if current_bad < best_bad_count:
                best_bad_count = current_bad
                best_coloring = c[:]
        
        if found:
            return c
        
        if best_coloring is not None and best_bad_count == 0:
            return best_coloring

    # Fallback: Seed program logic (greedy)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                c0 = colors[x - 1 - 3 * d]
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            for d in range(1, (x - 1) // 2 + 1):
                if x - 1 - 2 * d < 0:
                    continue
                if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
            
    return colors

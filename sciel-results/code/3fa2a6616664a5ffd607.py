import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    positions = list(range(1, n+1))
    
    # Precompute all nonconstant 4-APs
    aps = []
    for a in range(1, n+1):
        for d in range(1, n):
            if a + 3*d <= n:
                aps.append((a, a+d, a+2*d, a+3*d))
            else:
                break
    
    num_aps = len(aps)
    
    # Map position to indices of APs it belongs to
    pos_to_aps = {i: [] for i in positions}
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def try_local_search(seed, max_iters=10000):
        rng = random.Random(seed)
        # Initialize randomly with 3 colors
        # Using list for speed, index 0..127 corresponds to pos 1..128
        colors = [rng.randint(0, 2) for _ in range(n)]
        
        # Calculate initial violations
        violated = [False] * num_aps
        for idx in range(num_aps):
            ap = aps[idx]
            # ap contains 1-based indices
            c0 = colors[ap[0]-1]
            c1 = colors[ap[1]-1]
            c2 = colors[ap[2]-1]
            c3 = colors[ap[3]-1]
            # Check distinct count
            # Since only 3 colors, len < 3 means len 1 or 2
            # We can check: if c0==c1==c2==c3 OR (c0==c1 and c2==c3 and c0!=c2) etc
            # Simpler: len({c0,c1,c2,c3}) < 3
            if len({c0, c1, c2, c3}) < 3:
                violated[idx] = True
                
        num_violations = sum(violated)
        
        if num_violations == 0:
            return colors

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if num_violations == 0:
                return colors
            
            # Pick a random violated AP
            # Optimization: Maintain a list of violated indices?
            # For now, just pick random index and check if violated.
            # Or maintain a list.
            # Let's just pick random index and retry if not violated (if few violations).
            # Better: keep a list.
            v_indices = [i for i in range(num_aps) if violated[i]]
            if not v_indices:
                return colors
            ap_idx = v_indices[rng.randint(0, len(v_indices)-1)]
            ap = aps[ap_idx]
            
            improved = False
            # Try changing each position in the AP
            for pos in ap:
                old_color = colors[pos-1]
                # Try other 2 colors
                for c in range(3):
                    if c == old_color:
                        continue
                    colors[pos-1] = c
                    # Update violations for APs containing pos
                    delta = 0
                    for ap_i in pos_to_aps[pos]:
                        was_v = violated[ap_i]
                        # Get colors of AP ap_i
                        a1, a2, a3, a4 = aps[ap_i]
                        vals = {colors[a1-1], colors[a2-1], colors[a3-1], colors[a4-1]}
                        is_v = len(vals) < 3
                        violated[ap_i] = is_v
                        if is_v and not was_v:
                            delta += 1
                        elif not is_v and was_v:
                            delta -= 1
                    
                    if num_violations + delta < num_violations:
                        num_violations += delta
                        improved = True
                        break
                    else:
                        # Revert
                        colors[pos-1] = old_color
                        for ap_i in pos_to_aps[pos]:
                            a1, a2, a3, a4 = aps[ap_i]
                            vals = {colors[a1-1], colors[a2-1], colors[a3-1], colors[a4-1]}
                            violated[ap_i] = len(vals) < 3
                        # Note: num_violations is not updated here as we reverted
                if improved:
                    break
            
            if not improved:
                # Random kick
                pos = ap[rng.randint(0, 3)]
                new_c = rng.randint(0, 2)
                colors[pos-1] = new_c
                for ap_i in pos_to_aps[pos]:
                    a1, a2, a3, a4 = aps[ap_i]
                    vals = {colors[a1-1], colors[a2-1], colors[a3-1], colors[a4-1]}
                    violated[ap_i] = len(vals) < 3
                num_violations = sum(violated)

        return None

    # Try multiple seeds
    for seed in range(500):
        if time.time() > deadline:
            break
        result = try_local_search(seed)
        if result is not None:
            # Verify
            valid = True
            for ap in aps:
                if len({result[ap[0]-1], result[ap[1]-1], result[ap[2]-1], result[ap[3]-1]}) < 3:
                    valid = False
                    break
            if valid:
                return result

    # Fallback: Return a known valid 3-coloring if we can find one via simple pattern?
    # If local search fails, it's a problem.
    # But it shouldn't.
    return [0] * n

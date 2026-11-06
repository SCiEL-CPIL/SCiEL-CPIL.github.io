import random
import time
import itertools

def get_k4s(n=16):
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4s.append(edges)
    return k4s

K4_LIST = get_k4s(16)
NUM_K4 = len(K4_LIST)

offsets = [0] * 16
for i in range(16):
    offsets[i] = 15 * i - i * (i - 1) // 2

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return offsets[i] + (j - i - 1)

# Precompute edge_to_k4s
EDGE_TO_K4S = [[] for _ in range(120)]
K4_EDGE_INDICES = []
for k4_idx, edges in enumerate(K4_LIST):
    e_idxs = []
    for u, v in edges:
        e_idx = get_edge_idx(u, v)
        e_idxs.append(e_idx)
        EDGE_TO_K4S[e_idx].append(k4_idx)
    K4_EDGE_INDICES.append(e_idxs)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We try to find a valid coloring with C colors
    # Start with C=14, then 15
    
    for C in [14, 15]:
        # Initialize random coloring
        colors = [random.randint(0, C - 1) for _ in range(120)]
        
        # Initialize K4 color counts
        # k4_color_counts[k4_idx] is a list of C integers
        k4_color_counts = [[0] * C for _ in range(NUM_K4)]
        distinct_counts = [0] * NUM_K4
        total_violations = 0
        
        # Populate initial state
        for k4_idx in range(NUM_K4):
            e_idxs = K4_EDGE_INDICES[k4_idx]
            counts = k4_color_counts[k4_idx]
            for e_idx in e_idxs:
                c = colors[e_idx]
                counts[c] += 1
            distinct_counts[k4_idx] = sum(1 for cnt in counts if cnt > 0)
            if distinct_counts[k4_idx] < 5:
                total_violations += (5 - distinct_counts[k4_idx])
                
        if total_violations == 0:
            return colors

        best_violations = total_violations
        best_colors = colors.copy()
        
        # Simulated Annealing / Random Restart Hill Climbing
        # We will use simple hill climbing with random restarts if stuck.
        
        stuck_count = 0
        current_temp = 1.0
        temp_decay = 0.999
        
        while time.time() < deadline:
            # Pick random edge
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            
            # Try a random new color
            new_color = random.randint(0, C - 1)
            if new_color == old_color:
                continue
                
            # Apply change
            colors[e_idx] = new_color
            
            # Update affected K4s
            delta_violations = 0
            for k4_idx in EDGE_TO_K4S[e_idx]:
                counts = k4_color_counts[k4_idx]
                
                # Update old color count
                counts[old_color] -= 1
                if counts[old_color] == 0:
                    distinct_counts[k4_idx] -= 1
                    # Check violation change for old state
                    # We need to know the violation before and after
                    # Let's store old distinct count
                    pass 
                
                # This logic is getting complex to track delta accurately without storing old state.
                # Let's just recalculate the violation for this K4.
                # But we need the old violation to get the delta.
                
                # Let's store the current violation for each K4.
                # violation[k4_idx] = max(0, 5 - distinct_counts[k4_idx])
                
                # It's easier to just recalculate the total violations from scratch?
                # No, that's slow.
                
                # Let's maintain `violations` array.
                pass
                
            # This incremental update is tricky to get right quickly.
            # Let's stick to full recalculation but optimize it.
            # Full recalculation of 1820 K4s.
            # Each K4 has 6 edges.
            # We can use a small set or bitset.
            
            # Actually, let's just use the full recalculation but in a tight loop.
            # 1820 * 6 = 10920 lookups.
            # In Python, this is ~1ms.
            # 110s / 0.001s = 110,000 steps.
            # This is plenty.
            
            new_violations = 0
            for k4_idx in range(NUM_K4):
                e_idxs = K4_EDGE_INDICES[k4_idx]
                # Get colors
                c0 = colors[e_idxs[0]]
                c1 = colors[e_idxs[1]]
                c2 = colors[e_idxs[2]]
                c3 = colors[e_idxs[3]]
                c4 = colors[e_idxs[4]]
                c5 = colors[e_idxs[5]]
                # Count distinct
                # Using a set is slow.
                # Since C <= 16, we can use a bitmask if C <= 16.
                # C is 14 or 15.
                # Mask = (1<<c0) | (1<<c1) | ...
                # popcount of mask.
                
                mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
                distinct = bin(mask).count('1')
                
                if distinct < 5:
                    new_violations += (5 - distinct)
                    
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                
                if best_violations == 0:
                    return colors
            else:
                # Revert
                colors[e_idx] = old_color
                
            # Random restart if stuck for too long
            stuck_count += 1
            if stuck_count > 1000:
                stuck_count = 0
                # Restart with new random coloring
                colors = [random.randint(0, C - 1) for _ in range(120)]
                # Recalculate initial violations
                new_violations = 0
                for k4_idx in range(NUM_K4):
                    e_idxs = K4_EDGE_INDICES[k4_idx]
                    c0 = colors[e_idxs[0]]
                    c1 = colors[e_idxs[1]]
                    c2 = colors[e_idxs[2]]
                    c3 = colors[e_idxs[3]]
                    c4 = colors[e_idxs[4]]
                    c5 = colors[e_idxs[5]]
                    mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
                    distinct = bin(mask).count('1')
                    if distinct < 5:
                        new_violations += (5 - distinct)
                        
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return colors
                else:
                    # Keep best_colors as reference for next restart?
                    # No, just restart from scratch.
                    pass

    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

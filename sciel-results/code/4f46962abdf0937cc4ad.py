import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Store as 0-indexed tuples
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)

    # Precompute for each position, which APs it belongs to
    # pos_aps[i] = list of indices of APs that contain position i
    pos_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(ap_idx)

    # Precompute for each position and each AP it belongs to, the other 3 positions
    # other_positions[pos][ap_idx] = tuple of the other 3 positions in that AP
    other_positions = [dict() for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            others = tuple(p for p in ap if p != pos)
            other_positions[pos][ap_idx] = others

    def count_mono(colors):
        """Count monochromatic 4-term APs."""
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    def delta_for_flip(colors, pos):
        """Compute change in objective if we flip position pos."""
        # For each AP containing pos, check if flipping pos makes it mono or breaks mono
        delta = 0
        for ap_idx in pos_aps[pos]:
            ap = aps[ap_idx]
            others = other_positions[pos][ap_idx]
            cur_c = colors[pos]
            new_c = 1 - cur_c
            other_colors = [colors[o] for o in others]
            
            # Currently mono?
            is_mono_now = all(c == cur_c for c in other_colors)
            # Will be mono after flip?
            is_mono_after = all(c == new_c for c in other_colors)
            
            if is_mono_now and not is_mono_after:
                delta -= 1
            elif not is_mono_now and is_mono_after:
                delta += 1
        return delta

    # Initial solution: Thue-Morse sequence (good for avoiding APs)
    def thue_morse(n):
        # Number of 1-bits in binary representation of n
        return bin(n).count('1') % 2

    colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    current_cost = count_mono(colors)
    best_colors = colors.copy()
    best_cost = current_cost

    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    random.seed(42)

    while time.time() < deadline:
        # Pick a random position to flip
        pos = random.randint(0, N - 1)
        
        delta = delta_for_flip(colors, pos)
        
        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            colors[pos] = 1 - colors[pos]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return [int(c) for c in best_colors]

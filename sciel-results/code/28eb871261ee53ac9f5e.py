import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a <= 256
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    # ap_list[i] is a list of (ap_id, other_indices_tuple)
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta(colors, idx):
        """
        Calculate change in mono count if we flip colors[idx].
        """
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def local_search(colors):
        """
        Deterministic local search: repeatedly find the best single flip that reduces the count.
        Continue until no improvement is possible.
        """
        current_count = count_mono(colors)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_idx = -1
            
            for idx in range(N):
                d = get_delta(colors, idx)
                if d < best_delta:
                    best_delta = d
                    best_idx = idx
            
            if best_idx != -1:
                colors[best_idx] = 1 - colors[best_idx]
                current_count += best_delta
                improved = True
                
        return colors, current_count

    best_count = float('inf')
    best_colors = [0] * N
    
    # Run multiple random restarts
    while time.time() < deadline - 2.0:
        # Initialize with random coloring
        colors = [random.randint(0, 1) for _ in range(N)]
        
        # Perform local search
        colors, current_count = local_search(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
        # If we found a very good solution, we might break early, 
        # but let's keep searching for a bit longer to be safe.
        # A good lower bound for random is around 20-30. 
        # If we get below 15, it's quite good.
        if best_count <= 12:
            break

    return best_colors

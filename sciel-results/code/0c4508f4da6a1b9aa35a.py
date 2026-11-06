import numpy as np

def construct():
    n = 256
    # Precompute all 4-APs as (start, d) 0-indexed
    aps_list = []
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d + 1):
            aps_list.append((a, d))
    
    aps = np.array(aps_list, dtype=np.int32)
    starts = aps[:, 0]
    ds = aps[:, 1]
    
    def count_mono(colors):
        c0 = colors[starts]
        c1 = colors[starts + ds]
        c2 = colors[starts + 2 * ds]
        c3 = colors[starts + 3 * ds]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    best = None
    best_count = float('inf')
    
    np.random.seed(42)
    
    # Initial colorings to try
    initial_colorings = []
    # Seed coloring
    seed = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
    initial_colorings.append(seed)
    # Random
    for _ in range(99):
        initial_colorings.append(np.random.randint(0, 2, n, dtype=np.int8))
    
    for colors_init in initial_colorings:
        colors = colors_init.copy()
        current_count = count_mono(colors)
        
        # Hill climbing
        for _ in range(30):
            improved = False
            order = np.random.permutation(n)
            for i in order:
                old = colors[i]
                colors[i] = 1 - old
                new_count = count_mono(colors)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                else:
                    colors[i] = old
            if not improved:
                break
        
        if current_count < best_count:
            best_count = current_count
            best = colors.copy()
    
    return [int(x) for x in best]

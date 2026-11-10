import time
import random
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 0..255."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

_aps_list = _build_aps()
_aps_array = np.array(_aps_list, dtype=np.int32)

def _count_mono(colors):
    """Count monochromatic 4-APs using numpy vectorization."""
    # colors is a numpy array of 0/1
    # For each AP, check if all 4 elements have the same color
    c = colors[_aps_array]  # shape (num_aps, 4)
    # A 4-AP is monochromatic if all 4 colors are equal
    # c[:,0] == c[:,1] == c[:,2] == c[:,3]
    mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
    return int(np.sum(mono))

def _local_search(colors, max_iter=200, time_limit=None):
    """Greedy local search: flip bits that reduce mono-AP count.
    With occasional random moves to escape local minima."""
    colors = np.array(colors, dtype=np.int8)
    current_count = _count_mono(colors)
    best_count = current_count
    best_colors = colors.copy()
    
    n_aps = len(_aps_array)
    start_time = time.time()
    
    # Precompute for each position, which APs it participates in
    # For faster delta computation
    pos_in_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(_aps_list):
        for pos in ap:
            pos_in_aps[pos].append(idx)
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        improved = False
        iters += 1
        # Try flipping each position
        for i in range(N):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            old_color = colors[i]
            new_color = 1 - old_color
            colors[i] = new_color
            new_count = _count_mono(colors)
            if new_count < current_count:
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_colors = colors.copy()
                improved = True
            else:
                colors[i] = old_color
    
    return best_colors, best_count

def _simulated_annealing(colors, max_iter=500, time_limit=None):
    """Simulated annealing to escape local minima."""
    colors = np.array(colors, dtype=np.int8)
    current_count = _count_mono(colors)
    best_count = current_count
    best_colors = colors.copy()
    
    start_time = time.time()
    temp = 5.0
    cool_rate = 0.995
    
    for iters in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Pick a random position to flip
        i = random.randint(0, N - 1)
        old_color = colors[i]
        new_color = 1 - old_color
        colors[i] = new_color
        new_count = _count_mono(colors)
        
        delta = new_count - current_count
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_colors = colors.copy()
        else:
            colors[i] = old_color
        
        temp *= cool_rate
    
    return best_colors, best_count

def _generate_initializations():
    """Generate diverse initial colorings."""
    inits = []
    
    # Random
    for _ in range(5):
        inits.append(np.random.randint(0, 2, N).astype(np.int8))
    
    # Alternating
    inits.append(np.array([(i % 2) for i in range(N)], dtype=np.int8))
    
    # Blocks of 2
    inits.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    
    # Blocks of 4
    inits.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    
    # Thue-Morse like (first 256 bits)
    tm = [0]
    for i in range(1, N):
        tm.append(tm[i // 2] ^ (i % 2))
    inits.append(np.array(tm, dtype=np.int8))
    
    # Walsh-Hadamard like patterns
    for freq in [1, 2, 4, 8, 16]:
        pat = np.array([((i // freq) % 2) for i in range(N)], dtype=np.int8)
        inits.append(pat)
    
    # Random with structure: random blocks
    for _ in range(3):
        block_size = random.choice([2, 4, 8, 16])
        num_blocks = N // block_size
        block_colors = np.random.randint(0, 2, num_blocks).astype(np.int8)
        pat = np.repeat(block_colors, block_size)
        inits.append(pat)
    
    return inits

def construct():
    start_time = time.time()
    time_budget = 140  # Leave 10s margin
    
    best_colors = None
    best_count = float('inf')
    
    # Generate initializations
    inits = _generate_initializations()
    
    # Shuffle for diversity
    random.shuffle(inits)
    
    # Process each initialization
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        
        # First do local search
        colors, count = _local_search(init, max_iter=100, time_limit=time_budget - (time.time() - start_time))
        
        if count < best_count:
            best_count = count
            best_colors = colors.copy()
        
        # Then do simulated annealing from the local search result
        if time.time() - start_time < time_budget:
            colors_sa, count_sa = _simulated_annealing(colors, max_iter=300, time_limit=time_budget - (time.time() - start_time))
            if count_sa < best_count:
                best_count = count_sa
                best_colors = colors_sa.copy()
        
        # Do another local search from SA result
        if time.time() - start_time < time_budget:
            colors_ls, count_ls = _local_search(colors_sa, max_iter=50, time_limit=time_budget - (time.time() - start_time))
            if count_ls < best_count:
                best_count = count_ls
                best_colors = colors_ls.copy()
    
    # If we still have time, do more random restarts
    while time.time() - start_time < time_budget:
        init = np.random.randint(0, 2, N).astype(np.int8)
        colors, count = _local_search(init, max_iter=50, time_limit=time_budget - (time.time() - start_time))
        if count < best_count:
            best_count = count
            best_colors = colors.copy()
        
        if time.time() - start_time < time_budget:
            colors_sa, count_sa = _simulated_annealing(colors, max_iter=200, time_limit=time_budget - (time.time() - start_time))
            if count_sa < best_count:
                best_count = count_sa
                best_colors = colors_sa.copy()
    
    if best_colors is None:
        # Fallback: alternating
        best_colors = np.array([(i % 2) for i in range(N)], dtype=np.int8)
    
    # Convert to list of ints
    return [int(c) for c in best_colors]

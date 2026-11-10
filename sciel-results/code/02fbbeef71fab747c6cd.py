import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """For each position, store (ap_index, position_in_ap) for all APs it belongs to."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c].append((idx, 2))
        pos_to_aps[d].append((idx, 3))
    return pos_to_aps

def _score_full(aps, colors):
    """Count monochromatic 4-APs (full recount) using NumPy for speed."""
    colors = np.array(colors, dtype=np.int8)
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search_incremental(aps, pos_to_aps, colors, max_passes=500, deadline=None):
    """
    Fast greedy local search using incremental delta scoring.
    """
    colors = list(colors)
    n = len(colors)
    
    # Precompute AP positions as numpy array for fast access
    # We need fast access to check if an AP is mono.
    # Storing colors as a list is faster for single element access than numpy array in Python loops sometimes, 
    # but let's stick to list for colors and precompute AP indices.
    
    # Current mono status of each AP
    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())
    
    for _pass in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new
            
            # Compute delta
            delta = 0
            for ap_idx, _ in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                # Check if now mono
                # Since we flipped i, only the status of APs containing i changes.
                # An AP is mono if all 4 elements are equal.
                # If was_mono, it was all same. Flipping i makes it not mono (unless other 3 are different? No, if it was mono, all 4 were same. Flip one, it becomes 3 of one, 1 of other. So it becomes NOT mono).
                # So if was_mono, delta -= 1.
                # If not was_mono, it could become mono if the other 3 were all equal to the NEW color of i.
                
                if was_mono:
                    # It was mono. Now i is flipped. The other 3 are still the old color. 
                    # So we have 3 old, 1 new. Not mono.
                    delta -= 1
                    mono_flags[ap_idx] = 0
                else:
                    # It was not mono. 
                    # Check if the other 3 elements are all equal to the new color.
                    # Get the other 3 indices
                    other_indices = [x for x in (a, b, c, d) if x != i]
                    if colors[other_indices[0]] == colors[other_indices[1]] == colors[other_indices[2]] == new:
                        delta += 1
                        mono_flags[ap_idx] = 1
            
            if delta < 0:
                current_score += delta
                improved = True
            else:
                # Revert
                colors[i] = old
                for ap_idx, _ in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        
        if not improved:
            break
    
    return colors, current_score

def _simulated_annealing(aps, pos_to_aps, colors, start_temp=5.0, end_temp=0.01, max_iter=2000, deadline=None):
    """
    Simulated annealing with single-bit flips.
    """
    colors = list(colors)
    n = len(colors)
    
    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())
    best_score = current_score
    best_colors = colors[:]
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        temp *= cooling
        
        i = random.randint(0, n - 1)
        old = colors[i]
        colors[i] = 1 - old
        
        delta = 0
        for ap_idx, _ in pos_to_aps[i]:
            a, b, c, d = aps[ap_idx]
            was_mono = mono_flags[ap_idx]
            
            if was_mono:
                delta -= 1
                mono_flags[ap_idx] = 0
            else:
                other_indices = [x for x in (a, b, c, d) if x != i]
                new_color = colors[i]
                if colors[other_indices[0]] == colors[other_indices[1]] == colors[other_indices[2]] == new_color:
                    delta += 1
                    mono_flags[ap_idx] = 1
        
        new_score = current_score + delta
        if new_score < current_score or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
        else:
            colors[i] = old
            for ap_idx, _ in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        
    return best_colors, best_score

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return [bin(i).count('1') % 2 for i in range(n)]

def _quadratic_residue(p, n):
    """Generate sequence based on quadratic residues mod p."""
    res = set((x*x) % p for x in range(p))
    return [1 if (i % p) in res else 0 for i in range(n)]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    random.seed(42)
    
    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)
    
    best_score = float('inf')
    best_pop = None
    
    # Initialize with diverse, structured seeds
    seeds = [
        _thue_morse(N),
        [i % 2 for i in range(N)],
        [i % 3 for i in range(N)],
        [i % 4 for i in range(N)],
        _quadratic_residue(5, N),
        _quadratic_residue(7, N),
        [random.randint(0, 1) for _ in range(N)],
        [random.randint(0, 1) for _ in range(N)]
    ]
    
    # Evaluate and local search each seed
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        s = _score_full(aps, seed)
        if s < best_score:
            best_score = s
            best_pop = seed[:]
            
        # Intensive local search
        if time.time() < deadline - 10:
            improved_pop, s_ls = _local_search_incremental(aps, pos_to_aps, seed, max_passes=300, deadline=deadline - 5)
            if s_ls < best_score:
                best_score = s_ls
                best_pop = improved_pop[:]
                
            # SA polish on the best local solution
            if time.time() < deadline - 5:
                sa_pop, s_sa = _simulated_annealing(aps, pos_to_aps, improved_pop, start_temp=3.0, end_temp=0.01, max_iter=1000, deadline=deadline - 2)
                if s_sa < best_score:
                    best_score = s_sa
                    best_pop = sa_pop[:]
                    
    # Final global SA on the best found so far
    if time.time() < deadline - 10:
        sa_pop, s_sa = _simulated_annealing(aps, pos_to_aps, best_pop, start_temp=5.0, end_temp=0.001, max_iter=3000, deadline=deadline - 5)
        if s_sa < best_score:
            best_score = s_sa
            best_pop = sa_pop[:]
            
    # Final local search
    if time.time() < deadline - 2:
        final_pop, s_final = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=500, deadline=deadline - 0.5)
        if s_final < best_score:
            best_score = s_final
            best_pop = final_pop[:]
            
    return best_pop

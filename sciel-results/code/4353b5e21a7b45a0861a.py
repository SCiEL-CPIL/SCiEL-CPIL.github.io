import time
import numpy as np

N = 256

def _build_aps_np():
    """Build NumPy array of all 4-term arithmetic progressions in 0..N-1."""
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps_list.append([a, a + d, a + 2 * d, a + 3 * d])
    return np.array(aps_list, dtype=np.int32)

def _build_neighbors(aps_np):
    """Build neighbor lists and 'others' array for fast local search."""
    num_aps = aps_np.shape[0]
    neighbors = [[] for _ in range(N)]
    others = np.zeros((num_aps, 4, 3), dtype=np.int32)
    
    for idx in range(num_aps):
        a, b, c, d = aps_np[idx]
        neighbors[a].append((idx, 0))
        others[idx, 0] = [b, c, d]
        neighbors[b].append((idx, 1))
        others[idx, 1] = [a, c, d]
        neighbors[c].append((idx, 2))
        others[idx, 2] = [a, b, d]
        neighbors[d].append((idx, 3))
        others[idx, 3] = [a, b, c]
        
    return neighbors, others

def _evaluate_np(colors_np, aps_np):
    """Vectorized evaluation of monochromatic APs."""
    vals = colors_np[aps_np]
    all_same = np.all(vals == vals[:, :1], axis=1)
    return np.sum(all_same)

def _simulated_annealing(colors, aps_np, neighbors, others, time_limit):
    """Simulated Annealing to minimize monochromatic APs."""
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    ap_mono = np.zeros(aps_np.shape[0], dtype=bool)
    vals = colors_np[aps_np]
    ap_mono = np.all(vals == vals[:, :1], axis=1)
    current_mono = np.sum(ap_mono)
    
    best_mono = current_mono
    best_colors = list(colors)
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    
    rng = np.random.default_rng()
    
    start_time = time.time()
    
    # Pre-extract neighbors for faster access? 
    # neighbors is already a list of lists.
    
    while time.time() < start_time + time_limit and T > min_T:
        # Pick random index
        i = rng.integers(0, N)
        
        new_color = 1 - colors[i]
        old_color = colors[i]
        
        delta = 0
        
        # Calculate delta
        for ap_idx, pos in neighbors[i]:
            if ap_mono[ap_idx]:
                # It was mono. Flipping i breaks it.
                delta -= 1
            else:
                # It was not mono. Does it become mono?
                # The other 3 elements must be equal to new_color.
                # other_indices = others[ap_idx, pos]
                # Check manually for speed
                o = others[ap_idx, pos]
                if colors_np[o[0]] == new_color and colors_np[o[1]] == new_color and colors_np[o[2]] == new_color:
                    delta += 1
        
        # Metropolis acceptance
        if delta <= 0:
            accept = True
        else:
            # Probability of accepting uphill move
            # Use math.exp for speed?
            prob = np.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            colors[i] = new_color
            colors_np[i] = new_color
            current_mono += delta
            
            # Update ap_mono for affected APs
            for ap_idx, pos in neighbors[i]:
                if ap_mono[ap_idx]:
                    ap_mono[ap_idx] = False
                else:
                    # Re-check mono status
                    # We can use the same logic: check if all 4 are same
                    # But we know one element changed.
                    # If it became mono, the other 3 were new_color, and now i is new_color. So all 4 are new_color.
                    # If it didn't become mono, it's still not mono.
                    # So we only need to set it to True if the condition was met.
                    # We already checked the condition in the delta loop.
                    # To avoid double work, we could store the result.
                    # Let's just recompute to be safe and simple, it's O(4).
                    ap_vals = colors_np[aps_np[ap_idx]]
                    ap_mono[ap_idx] = np.all(ap_vals == ap_vals[0])
            
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = list(colors)
        
        T *= cooling
        
    return best_colors, best_mono

def _generate_lfsr_sequence(state, taps):
    """Generate a 256-bit sequence from an LFSR."""
    seq = []
    current_state = state[:]
    for _ in range(N):
        seq.append(current_state[0])
        fb = 0
        for t in range(8):
            if taps & (1 << t):
                fb ^= current_state[t]
        current_state = current_state[1:] + [fb]
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    aps_np = _build_aps_np()
    neighbors, others = _build_neighbors(aps_np)
    
    rng = np.random.default_rng(42)
    
    best_mono = float('inf')
    best_colors = None
    
    # Phase 1: Explore LFSR parameter space
    num_lfsr_samples = 5000
    lfsr_results = []
    
    for _ in range(num_lfsr_samples):
        if time.time() > deadline - 30:
            break
            
        state = rng.integers(0, 2, size=8).tolist()
        taps = rng.integers(1, 256)
        
        seq = _generate_lfsr_sequence(state, taps)
        seq_np = np.array(seq, dtype=np.int8)
        
        mono = _evaluate_np(seq_np, aps_np)
        
        if mono < best_mono:
            best_mono = mono
            best_colors = seq
        
        lfsr_results.append((mono, seq))
        
        seq_complement = [1 - x for x in seq]
        seq_comp_np = np.array(seq_complement, dtype=np.int8)
        mono_comp = _evaluate_np(seq_comp_np, aps_np)
        
        if mono_comp < best_mono:
            best_mono = mono_comp
            best_colors = seq_complement
            
        lfsr_results.append((mono_comp, seq_complement))
    
    # Sort by score
    lfsr_results.sort(key=lambda x: x[0])
    
    # Phase 2: Simulated Annealing on top LFSR results
    top_k = min(10, len(lfsr_results))
    
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
            
        mono, colors = lfsr_results[i]
        
        # Run SA
        polished, polished_mono = _simulated_annealing(
            colors, aps_np, neighbors, others, time_limit=10
        )
        
        if polished_mono < best_mono:
            best_mono = polished_mono
            best_colors = polished
    
    # Phase 3: Random starts with SA
    if time.time() < deadline - 20:
        num_random_starts = 20
        for _ in range(num_random_starts):
            if time.time() > deadline - 5:
                break
                
            random_colors = rng.integers(0, 2, size=N).tolist()
            
            polished, polished_mono = _simulated_annealing(
                random_colors, aps_np, neighbors, others, time_limit=5
            )
            
            if polished_mono < best_mono:
                best_mono = polished_mono
                best_colors = polished
    
    return best_colors

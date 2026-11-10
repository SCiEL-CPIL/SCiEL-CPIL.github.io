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
    return int(np.sum(all_same))

def _compute_delta(colors_np, ap_mono, i, neighbors, others, aps_np):
    """Compute delta for flipping position i."""
    new_color = 1 - colors_np[i]
    delta = 0
    for ap_idx, pos in neighbors[i]:
        if ap_mono[ap_idx]:
            delta -= 1
        else:
            o = others[ap_idx, pos]
            if colors_np[o[0]] == new_color and colors_np[o[1]] == new_color and colors_np[o[2]] == new_color:
                delta += 1
    return delta

def _local_search_fast(colors, aps_np, neighbors, others, max_passes=20):
    """Fast local search using pre-computed neighbors."""
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    ap_mono = np.zeros(aps_np.shape[0], dtype=bool)
    vals = colors_np[aps_np]
    ap_mono = np.all(vals == vals[:, :1], axis=1)
    current_mono = int(np.sum(ap_mono))
    
    best_mono = current_mono
    best_colors = list(colors)
    
    for _ in range(max_passes):
        improved = False
        for i in range(N):
            delta = _compute_delta(colors_np, ap_mono, i, neighbors, others, aps_np)
            
            if delta < 0:
                new_color = 1 - colors[i]
                colors[i] = new_color
                colors_np[i] = new_color
                current_mono += delta
                
                for ap_idx, pos in neighbors[i]:
                    ap_vals = colors_np[aps_np[ap_idx]]
                    ap_mono[ap_idx] = np.all(ap_vals == ap_vals[0])
                improved = True
        
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = list(colors)
            
        if not improved:
            break
            
    return best_colors, best_mono

def _simulated_annealing(colors, aps_np, neighbors, others, time_limit, rng=None):
    """Simulated Annealing to minimize monochromatic APs."""
    if rng is None:
        rng = np.random.default_rng()
    
    colors = list(colors)
    colors_np = np.array(colors, dtype=np.int8)
    
    ap_mono = np.zeros(aps_np.shape[0], dtype=bool)
    vals = colors_np[aps_np]
    ap_mono = np.all(vals == vals[:, :1], axis=1)
    current_mono = int(np.sum(ap_mono))
    
    best_mono = current_mono
    best_colors = list(colors)
    
    T = 2.0
    cooling = 0.9995
    min_T = 1e-6
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > min_T:
        i = int(rng.integers(0, N))
        
        delta = _compute_delta(colors_np, ap_mono, i, neighbors, others, aps_np)
        
        if delta <= 0:
            accept = True
        else:
            prob = np.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            new_color = 1 - colors[i]
            colors[i] = new_color
            colors_np[i] = new_color
            current_mono += delta
            
            for ap_idx, pos in neighbors[i]:
                ap_vals = colors_np[aps_np[ap_idx]]
                ap_mono[ap_idx] = np.all(ap_vals == ap_vals[0])
            
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = list(colors)
        
        T *= cooling
    
    return best_colors, best_mono

def _walsh_hadamard_init(rng):
    """Generate coloring via product of two 16-bit Walsh-Hadamard basis functions."""
    # Generate random 16-bit vectors
    v1 = rng.integers(0, 2, size=16)
    v2 = rng.integers(0, 2, size=16)
    
    # Compute outer product (rank-1 matrix)
    M = np.outer(v1, v2)  # 16x16 binary matrix
    
    # Reshape to 256 elements
    colors = M.flatten().tolist()
    
    return colors

def _lfsr_sequence(state, taps):
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
    deadline = start_time + 135  # Safety margin before 150s limit
    
    aps_np = _build_aps_np()
    neighbors, others = _build_neighbors(aps_np)
    
    rng = np.random.default_rng(42)
    
    best_mono = float('inf')
    best_colors = None
    
    # Phase 1: Generate diverse initial colorings
    candidates = []
    
    # Walsh-Hadamard spectral initializations
    for _ in range(200):
        colors = _walsh_hadamard_init(rng)
        colors_np = np.array(colors, dtype=np.int8)
        mono = _evaluate_np(colors_np, aps_np)
        candidates.append((mono, colors))
        
        # Also try complement
        comp = [1 - x for x in colors]
        comp_np = np.array(comp, dtype=np.int8)
        mono_comp = _evaluate_np(comp_np, aps_np)
        candidates.append((mono_comp, comp))
    
    # LFSR initializations
    for _ in range(3000):
        if time.time() > deadline - 40:
            break
        state = rng.integers(0, 2, size=8).tolist()
        taps = int(rng.integers(1, 256))
        seq = _lfsr_sequence(state, taps)
        seq_np = np.array(seq, dtype=np.int8)
        mono = _evaluate_np(seq_np, aps_np)
        candidates.append((mono, seq))
        
        comp = [1 - x for x in seq]
        comp_np = np.array(comp, dtype=np.int8)
        mono_comp = _evaluate_np(comp_np, aps_np)
        candidates.append((mono_comp, comp))
    
    # Random initializations
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        colors = rng.integers(0, 2, size=N).tolist()
        colors_np = np.array(colors, dtype=np.int8)
        mono = _evaluate_np(colors_np, aps_np)
        candidates.append((mono, colors))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top candidates
    top_k_ls = min(15, len(candidates))
    for i in range(top_k_ls):
        if time.time() > deadline - 50:
            break
        mono, colors = candidates[i]
        polished, polished_mono = _local_search_fast(
            colors, aps_np, neighbors, others, max_passes=20
        )
        if polished_mono < best_mono:
            best_mono = polished_mono
            best_colors = polished
    
    # Phase 3: Simulated annealing on best local search results
    # Re-evaluate top candidates after local search
    sa_candidates = []
    for i in range(min(10, len(candidates))):
        mono, colors = candidates[i]
        sa_candidates.append((mono, colors))
    
    sa_candidates.sort(key=lambda x: x[0])
    
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Allocate time for SA runs
        num_sa = min(5, int(remaining_time / 10))
        sa_time_each = max(3, (remaining_time - 5) / max(1, num_sa))
        
        for i in range(num_sa):
            if time.time() > deadline - 5:
                break
            mono, colors = sa_candidates[i]
            polished, polished_mono = _simulated_annealing(
                colors, aps_np, neighbors, others, time_limit=sa_time_each, rng=rng
            )
            if polished_mono < best_mono:
                best_mono = polished_mono
                best_colors = polished
            
            # Also try SA on the locally-searched version
            ls_colors, ls_mono = _local_search_fast(
                colors, aps_np, neighbors, others, max_passes=10
            )
            if time.time() < deadline - 5:
                polished2, polished_mono2 = _simulated_annealing(
                    ls_colors, aps_np, neighbors, others, 
                    time_limit=max(2, (deadline - time.time()) - 5), rng=rng
                )
                if polished_mono2 < best_mono:
                    best_mono = polished_mono2
                    best_colors = polished2
    
    # Phase 4: Final polish with local search on the best result
    if best_colors is not None and time.time() < deadline - 2:
        final_colors, final_mono = _local_search_fast(
            best_colors, aps_np, neighbors, others, max_passes=10
        )
        if final_mono < best_mono:
            best_mono = final_mono
            best_colors = final_colors
    
    return best_colors

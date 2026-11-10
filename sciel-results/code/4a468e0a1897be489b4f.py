import time
import random
import numpy as np

N = 256

def _precompute():
    """Precompute all 4-term arithmetic progressions and position-to-AP mappings."""
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            # 0-indexed positions
            p0 = a
            p1 = a + d
            p2 = a + 2 * d
            p3 = a + 3 * d
            
            ap_idx = len(aps)
            aps.append((p0, p1, p2, p3))
            
            pos_to_aps[p0].append(ap_idx)
            pos_to_aps[p1].append(ap_idx)
            pos_to_aps[p2].append(ap_idx)
            pos_to_aps[p3].append(ap_idx)
            
    return aps, pos_to_aps

APS, POS_TO_APS = _precompute()

def _count_mono_numpy(colors):
    """Count monochromatic 4-APs using vectorized numpy operations."""
    # Convert to numpy array if not already
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.int8)
        
    # Extract colors for each position in each AP
    # APS is a list of tuples, we can convert to a numpy array for indexing
    # But doing this every time is expensive. Let's precompute a numpy array of APs.
    pass

# Let's precompute a numpy array of APs for fast counting
APS_ARRAY = np.array(APS, dtype=np.int32)

def _count_mono_fast(colors):
    """Fast count using precomputed numpy array of APs."""
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.int8)
        
    c0 = colors[APS_ARRAY[:, 0]]
    c1 = colors[APS_ARRAY[:, 1]]
    c2 = colors[APS_ARRAY[:, 2]]
    c3 = colors[APS_ARRAY[:, 3]]
    
    # Check if all four are equal
    return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

def _delta_1bit(colors, pos):
    """Calculate change in mono count if we flip position pos."""
    delta = 0
    c_pos = colors[pos]
    
    for ap_idx in POS_TO_APS[pos]:
        p0, p1, p2, p3 = APS[ap_idx]
        
        # Get the other three positions
        if pos == p0:
            others = (p1, p2, p3)
        elif pos == p1:
            others = (p0, p2, p3)
        elif pos == p2:
            others = (p0, p1, p3)
        else: # pos == p3
            others = (p0, p1, p2)
            
        co1 = colors[others[0]]
        co2 = colors[others[1]]
        co3 = colors[others[2]]
        
        # If the other three are all the same, the AP's mono status depends on pos
        if co1 == co2 == co3:
            if c_pos == co1:
                # Was mono, will not be mono
                delta -= 1
            else:
                # Was not mono, will be mono
                delta += 1
                
    return delta

def _delta_2bit(colors, pos1, pos2):
    """Calculate change in mono count if we flip positions pos1 and pos2."""
    if pos1 == pos2:
        return 0
        
    # Get set of APs involving pos1 or pos2
    involved_aps = set(POS_TO_APS[pos1])
    involved_aps.update(POS_TO_APS[pos2])
    
    delta = 0
    c1 = colors[pos1]
    c2 = colors[pos2]
    
    for ap_idx in involved_aps:
        p0, p1, p2, p3 = APS[ap_idx]
        
        # Get current colors of the 4 positions
        ca = colors[p0]
        cb = colors[p1]
        cc = colors[p2]
        cd = colors[p3]
        
        # Check if mono before
        was_mono = (ca == cb == cc == cd)
        
        # Simulate flip
        new_ca = ca
        new_cb = cb
        new_cc = cc
        new_cd = cd
        
        if p0 == pos1:
            new_ca = 1 - new_ca
        elif p0 == pos2:
            new_ca = 1 - new_ca
            
        if p1 == pos1:
            new_cb = 1 - new_cb
        elif p1 == pos2:
            new_cb = 1 - new_cb
            
        if p2 == pos1:
            new_cc = 1 - new_cc
        elif p2 == pos2:
            new_cc = 1 - new_cc
            
        if p3 == pos1:
            new_cd = 1 - new_cd
        elif p3 == pos2:
            new_cd = 1 - new_cd
            
        # Check if mono after
        is_mono = (new_ca == new_cb == new_cc == new_cd)
        
        if is_mono and not was_mono:
            delta += 1
        elif was_mono and not is_mono:
            delta -= 1
            
    return delta

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Generate diverse initializations
    inits = []
    
    # 1. Random
    inits.append(np.random.randint(0, 2, N).astype(np.int8))
    
    # 2. Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
    inits.append(tm)
    
    # 3. Block 2
    inits.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    
    # 4. Block 4
    inits.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    
    # 5. Block 8
    inits.append(np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8))
    
    # 6. Striped
    inits.append(np.array([i % 2 for i in range(N)], dtype=np.int8))
    
    # Evaluate and sort
    scored_inits = []
    for init in inits:
        score = _count_mono_fast(init)
        scored_inits.append((score, init))
        
    scored_inits.sort(key=lambda x: x[0])
    
    best_global_score = scored_inits[0][0]
    best_global_colors = scored_inits[0][1].copy()
    
    # Use the top 3 initializations for multi-start SA
    start_candidates = [x[1].copy() for x in scored_inits[:3]]
    
    for start_colors in start_candidates:
        if time.time() > deadline - 10.0:
            break
            
        colors = start_colors.copy()
        current_score = _count_mono_fast(colors)
        
        if current_score < best_global_score:
            best_global_score = current_score
            best_global_colors = colors.copy()
            
        # Simulated Annealing parameters
        T = 2.0
        T_min = 0.01
        alpha = 0.99995  # Cooling rate
        steps_since_improve = 0
        max_steps_no_improve = 20000
        
        while time.time() < deadline - 5.0:
            if T < T_min:
                # Reheat with perturbation
                # Perturb the best global solution found so far
                colors = best_global_colors.copy()
                n_perturb = random.randint(10, 30)
                for _ in range(n_perturb):
                    pos = random.randint(0, N - 1)
                    colors[pos] = 1 - colors[pos]
                    
                current_score = _count_mono_fast(colors)
                if current_score < best_global_score:
                    best_global_score = current_score
                    best_global_colors = colors.copy()
                    
                T = 1.0
                steps_since_improve = 0
                continue
                
            # Choose move: 1-bit or 2-bit
            use_2bit = random.random() < 0.3
            
            if use_2bit:
                pos1 = random.randint(0, N - 1)
                pos2 = random.randint(0, N - 1)
                while pos2 == pos1:
                    pos2 = random.randint(0, N - 1)
                    
                delta = _delta_2bit(colors, pos1, pos2)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[pos1] = 1 - colors[pos1]
                    colors[pos2] = 1 - colors[pos2]
                    current_score += delta
                    
                    if current_score < best_global_score:
                        best_global_score = current_score
                        best_global_colors = colors.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            else:
                pos = random.randint(0, N - 1)
                delta = _delta_1bit(colors, pos)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[pos] = 1 - colors[pos]
                    current_score += delta
                    
                    if current_score < best_global_score:
                        best_global_score = current_score
                        best_global_colors = colors.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                        
            T *= alpha
            
            # Optional: Stop if no improvement for a long time and T is low
            if steps_since_improve > max_steps_no_improve and T < 0.5:
                break
                
    # Final local search polish
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for pos in range(N):
                delta = _delta_1bit(best_global_colors, pos)
                if delta < 0:
                    best_global_colors[pos] = 1 - best_global_colors[pos]
                    best_global_score += delta
                    improved = True
                    
    return [int(c) for c in best_global_colors]

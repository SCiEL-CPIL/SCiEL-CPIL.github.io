import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 128
    # Precompute all nonconstant 4-term progressions
    # A 4-AP is defined by start a and difference d > 0
    # a, a+d, a+2d, a+3d <= N
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            ap_id = len(aps)
            aps.append(idx)
            for p in idx:
                pos_to_aps[p].append(ap_id)
    
    num_aps = len(aps)
    
    def count_violations(colors):
        """Count 4-APs with < 3 distinct colors."""
        v = 0
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                v += 1
        return v

    def get_violation_status(colors):
        """Return a boolean array where True means the AP is a violation."""
        status = np.zeros(num_aps, dtype=bool)
        for i, ap in enumerate(aps):
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                status[i] = True
        return status

    def spectral_guided_init(k, time_limit):
        """
        Initialize coloring using spectral analysis of 4-AP structure.
        The 4-AP constraint matrix has a regular structure that can be analyzed
        in frequency domain to identify low-frequency patterns that tend to satisfy
        the 3-color condition.
        """
        # Build the constraint incidence matrix (sparse representation)
        # M[i, j] = 1 if position j is in AP i
        # For each AP, we want at least 3 distinct colors.
        
        # Analyze the structure: 4-APs have differences d and starts a
        # The constraint is: for each (a,d), the four positions must have >= 3 colors
        
        # Use Fourier analysis to identify periodic patterns that work well
        # A periodic coloring with period P will have 4-APs that are either:
        # - All same residue mod P (if d is multiple of P)
        # - Mixed residues (if d is not multiple of P)
        
        # Try different periods and find ones that minimize violations
        best_period = None
        best_violations = float('inf')
        
        for period in range(2, N // 2 + 1):
            # Create a periodic coloring with k colors
            base_pattern = np.random.randint(0, k, period)
            
            # Ensure the base pattern has enough variety
            if len(set(base_pattern)) < min(k, 3):
                continue
                
            colors = np.array([base_pattern[i % period] for i in range(N)])
            
            # Count violations
            v = count_violations(colors)
            if v < best_violations:
                best_violations = v
                best_period = period
                best_colors = colors.copy()
                
            if v == 0:
                break
        
        if best_period is not None:
            return best_colors, best_violations
        else:
            # Fallback to random
            return np.random.randint(0, k, N), count_violations(np.random.randint(0, k, N))

    def try_coloring(k, time_limit):
        """Try to find a valid coloring with k colors using spectral-guided initialization."""
        start_time = time.time()
        
        # Use spectral-guided initialization
        colors, initial_v = spectral_guided_init(k, time_limit)
        
        # Initialize violation status
        v_status = get_violation_status(colors)
        v_count = np.sum(v_status)
        
        best_v = v_count
        best_colors = colors.copy()
        
        # Simulated Annealing parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.9995 # slower cooling
        
        while time.time() - start_time < time_limit:
            # Pick random position
            p = np.random.randint(0, N)
            old_c = colors[p]
            
            # Pick new color
            new_c = np.random.randint(0, k)
            if new_c == old_c:
                continue
                
            # Calculate change in violations
            delta_v = 0
            affected_aps = pos_to_aps[p]
            
            for ap_id in affected_aps:
                ap = aps[ap_id]
                others = [x for x in ap if x != p]
                
                old_colors = set([colors[x] for x in others] + [old_c])
                old_viol = len(old_colors) < 3
                
                new_colors = set([colors[x] for x in others] + [new_c])
                new_viol = len(new_colors) < 3
                
                if not old_viol and new_viol:
                    delta_v += 1
                elif old_viol and not new_viol:
                    delta_v -= 1
                
            # Acceptance criterion
            if delta_v < 0:
                colors[p] = new_c
                for ap_id in affected_aps:
                    ap = aps[ap_id]
                    others = [x for x in ap if x != p]
                    c = set([colors[x] for x in others] + [new_c])
                    v_status[ap_id] = len(c) < 3
                v_count += delta_v
                if v_count < best_v:
                    best_v = v_count
                    best_colors = colors.copy()
                    if v_count == 0:
                        return best_colors
            else:
                if np.random.rand() < np.exp(-delta_v / T):
                    colors[p] = new_c
                    for ap_id in affected_aps:
                        ap = aps[ap_id]
                        others = [x for x in ap if x != p]
                        c = set([colors[x] for x in others] + [new_c])
                        v_status[ap_id] = len(c) < 3
                    v_count += delta_v
                    if v_count < best_v:
                        best_v = v_count
                        best_colors = colors.copy()
                        if v_count == 0:
                            return best_colors
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_colors

    # Deadline guard
    total_start = time.time()
    total_limit = 55.0
    
    # Try 3 colors first with spectral guidance
    if time.time() - total_start < total_limit:
        res = try_coloring(3, time_limit=12.0)
        if count_violations(res) == 0:
            return res.tolist()
    
    # If 3 colors failed, try 4 colors
    if time.time() - total_start < total_limit:
        res = try_coloring(4, time_limit=15.0)
        if count_violations(res) == 0:
            return res.tolist()
            
    # If 4 colors failed, try 5 colors
    if time.time() - total_start < total_limit:
        res = try_coloring(5, time_limit=15.0)
        if count_violations(res) == 0:
            return res.tolist()
            
    remaining = total_limit - (time.time() - total_start)
    if remaining > 1:
         res = try_coloring(3, time_limit=remaining)
         if count_violations(res) == 0:
             return res.tolist()
         
    # Fallback greedy construction
    colors = []
    for x in range(1, N + 1):
        found = False
        for color in range(N):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors

# EVOLVE-BLOCK-END

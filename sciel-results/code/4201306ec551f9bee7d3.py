import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
    
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Map each index (0..255) to the list of AP indices it belongs to
    # This allows O(1) lookup of affected APs when flipping a bit
    index_to_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for pos in aps[i]:
            index_to_aps[pos].append(i)
    
    # Convert to numpy arrays for faster iteration if needed, 
    # but lists are fine for small counts per index
    # Actually, let's keep as list of lists for simplicity, 
    # but we can optimize the delta calculation.
    
    # Initial random coloring
    coloring = np.random.randint(0, 2, size=N)
    
    # Calculate initial score
    # Monochromatic if all 4 bits are same
    # c[a] == c[a+d] == c[a+2d] == c[a+3d]
    # This is equivalent to:
    # (c[a] == c[a+d]) & (c[a+d] == c[a+2d]) & (c[a+2d] == c[a+3d])
    
    def calc_score(coloring):
        # Vectorized calculation
        a = aps[:, 0]
        b = aps[:, 1]
        c = aps[:, 2]
        d = aps[:, 3]
        
        ca = coloring[a]
        cb = coloring[b]
        cc = coloring[c]
        cd = coloring[d]
        
        mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(mono)
    
    current_score = calc_score(coloring)
    best_coloring = coloring.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while T > T_min and time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(0, N)
        
        # Calculate delta score
        # Only need to check APs containing idx
        delta = 0
        for ap_idx in index_to_aps[idx]:
            a, b, c, d = aps[ap_idx]
            
            # Check if current AP is monochromatic
            ca = coloring[a]
            cb = coloring[b]
            cc = coloring[c]
            cd = coloring[d]
            is_mono_before = (ca == cb) and (cb == cc) and (cc == cd)
            
            # Simulate flip
            new_val = 1 - coloring[idx]
            
            # Determine new values for the AP
            if idx == a:
                nca = new_val
                ncb = cb
                ncc = cc
                ncd = cd
            elif idx == b:
                nca = ca
                ncb = new_val
                ncc = cc
                ncd = cd
            elif idx == c:
                nca = ca
                ncb = cb
                ncc = new_val
                ncd = cd
            else: # idx == d
                nca = ca
                ncb = cb
                ncc = cc
                ncd = new_val
            
            is_mono_after = (nca == ncb) and (ncb == ncc) and (ncc == ncd)
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
        
        # Acceptance criterion
        if delta <= 0:
            # Improvement or neutral
            coloring[idx] = 1 - coloring[idx]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_coloring = coloring.copy()
        else:
            # Deterioration, accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta / T):
                coloring[idx] = 1 - coloring[idx]
                current_score += delta
        
        T *= alpha
        
        # Restart if stuck for a long time? 
        # For now, simple cooling.
        
    # Return best coloring found
    # The task requires a list of 256 colours in {0,1}
    return best_coloring.tolist()
# EVOLVE-BLOCK-END

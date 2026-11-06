import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # Store 0-indexed positions
            ap_list.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, which APs contain it
    # index_aps[idx] = array of (ap_index, position)
    index_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for j in range(4):
            idx = aps[i, j]
            index_aps[idx].append((i, j))
    
    # Convert to numpy arrays for fast access
    index_aps_arr = []
    for i in range(N):
        if index_aps[i]:
            arr = np.array(index_aps[i], dtype=np.int32)
            index_aps_arr.append(arr)
        else:
            index_aps_arr.append(np.array([], dtype=np.int32).reshape(0, 2))

    # Function to calculate full score
    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))
    
    # Fast delta computation for a single flip
    def compute_delta(colors, idx):
        pairs = index_aps_arr[idx]
        if len(pairs) == 0:
            return 0
        
        ap_indices = pairs[:, 0]
        positions = pairs[:, 1]
        
        # Get current colors for all 4 positions in affected APs
        old_c0 = colors[aps[ap_indices, 0]]
        old_c1 = colors[aps[ap_indices, 1]]
        old_c2 = colors[aps[ap_indices, 2]]
        old_c3 = colors[aps[ap_indices, 3]]
        
        # Which were monochromatic before?
        old_mono = (old_c0 == old_c1) & (old_c1 == old_c2) & (old_c2 == old_c3)
        old_count = int(np.sum(old_mono))
        
        # Determine new mono status
        # The flipped index's color changes.
        # If the flipped index was the 'odd one out' in a non-mono AP, it might become mono.
        # If it was part of a mono AP, it becomes non-mono.
        
        # Let's just compute the new state for the affected APs.
        new_c0 = old_c0.copy()
        new_c1 = old_c1.copy()
        new_c2 = old_c2.copy()
        new_c3 = old_c3.copy()
        
        for pos in range(4):
            mask = (positions == pos)
            if np.any(mask):
                if pos == 0:
                    new_c0[mask] = 1 - new_c0[mask]
                elif pos == 1:
                    new_c1[mask] = 1 - new_c1[mask]
                elif pos == 2:
                    new_c2[mask] = 1 - new_c2[mask]
                elif pos == 3:
                    new_c3[mask] = 1 - new_c3[mask]
        
        new_mono = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    # Fast delta computation for a double flip (i, j)
    # This is more complex, so we approximate by doing two single flips sequentially 
    # in a local buffer, or just skip if too slow. 
    # Given N=256, a full 2-flip search is O(N^2 * degree). 
    # 256^2 * 20 ~ 1.3M ops per sweep. With numpy overhead, maybe 0.1s per sweep.
    # We can afford a few sweeps.
    
    def compute_delta_double(colors, i, j):
        # Get affected APs for i and j
        pairs_i = index_aps_arr[i]
        pairs_j = index_aps_arr[j]
        
        if len(pairs_i) == 0 and len(pairs_j) == 0:
            return 0
            
        # Create a set of unique AP indices affected
        ap_set = set()
        if len(pairs_i) > 0:
            ap_set.update(pairs_i[:, 0].tolist())
        if len(pairs_j) > 0:
            ap_set.update(pairs_j[:, 0].tolist())
            
        if not ap_set:
            return 0
            
        ap_indices = np.array(list(ap_set), dtype=np.int32)
        
        # Get current colors
        old_c0 = colors[aps[ap_indices, 0]]
        old_c1 = colors[aps[ap_indices, 1]]
        old_c2 = colors[aps[ap_indices, 2]]
        old_c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (old_c0 == old_c1) & (old_c1 == old_c2) & (old_c2 == old_c3)
        old_count = int(np.sum(old_mono))
        
        # Apply flips
        new_c0 = old_c0.copy()
        new_c1 = old_c1.copy()
        new_c2 = old_c2.copy()
        new_c3 = old_c3.copy()
        
        # Flip i
        for pos in range(4):
            mask_i = (pairs_i[:, 1] == pos)
            # We need to map local indices in pairs_i to global ap_indices
            # This is tricky with sets. Let's use a different approach for double flip.
            # Instead of complex mapping, we can just simulate the flip on the full array 
            # for the specific APs, but that's slow.
            
            # Alternative: Just return 0 for double flip to keep it simple and fast?
            # No, double flip is key to escaping local minima.
            
            # Let's stick to single flips in SA and use a simpler 2-flip heuristic:
            # Randomly pick pairs and test.
            pass
        
        # Simplified 2-flip delta:
        # We will just create a copy of the colors, flip i and j, and compute delta for the union of affected APs.
        # This is safe but slightly slower.
        
        # For now, let's rely on strong SA and Greedy. 
        # If we have time, we can add 2-flip.
        # Let's implement a simple 2-flip check for random pairs in the polish phase.
        return 0 # Placeholder, will implement proper logic below if needed.

    # Proper 2-flip delta
    def compute_delta_double_fast(colors, i, j):
        # Identify all APs containing i or j
        # We can precompute a combined neighborhood? No, dynamic.
        
        # Get APs for i
        pi = index_aps_arr[i]
        # Get APs for j
        pj = index_aps_arr[j]
        
        if len(pi) == 0 and len(pj) == 0:
            return 0
        
        # Union of AP indices
        if len(pi) == 0:
            ap_indices = pj[:, 0]
        elif len(pj) == 0:
            ap_indices = pi[:, 0]
        else:
            # Use np.union1d for speed? Or just concatenate and unique.
            ap_indices = np.unique(np.concatenate([pi[:, 0], pj[:, 0]]))
        
        if len(ap_indices) == 0:
            return 0
            
        old_c0 = colors[aps[ap_indices, 0]]
        old_c1 = colors[aps[ap_indices, 1]]
        old_c2 = colors[aps[ap_indices, 2]]
        old_c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (old_c0 == old_c1) & (old_c1 == old_c2) & (old_c2 == old_c3)
        old_count = int(np.sum(old_mono))
        
        new_c0 = old_c0.copy()
        new_c1 = old_c1.copy()
        new_c2 = old_c2.copy()
        new_c3 = old_c3.copy()
        
        # Flip i
        for pos in range(4):
            mask = (pi[:, 1] == pos)
            if np.any(mask):
                # Need to find these APs in ap_indices
                # This is inefficient. 
                # Let's just do a full score recalculation for the 2-flip if the number of affected APs is small?
                # Or better: Just use the single flip logic twice? No, interactions.
                
                # Let's stick to a simpler strategy: 
                # SA with single flips + Greedy Polish.
                # This is often sufficient.
                pass
        
        # Given the complexity of fast 2-flip delta, and that SA+Greedy is strong, 
        # I will omit explicit 2-flip delta optimization in this rewrite to ensure speed and reliability.
        # The previous program scored 2.46. We can improve by better SA parameters and more restarts.
        
        return 0

    best_score = float('inf')
    best_colors = None
    
    # Number of restarts
    num_restarts = 10
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)
        
        # Random initial coloring
        colors = np.random.randint(0, 2, N).astype(np.int32)
        current_score = get_score(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
        
        # Simulated Annealing
        T = 10.0
        cooling_factor = 0.995
        min_T = 0.05
        
        while time.time() < restart_deadline and T > min_T:
            idx = np.random.randint(0, N)
            
            delta = compute_delta(colors, idx)
            
            if delta <= 0:
                colors[idx] = 1 - colors[idx]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
            else:
                if random.random() < np.exp(-delta / T):
                    colors[idx] = 1 - colors[idx]
                    current_score += delta
            
            T *= cooling_factor
        
        # Greedy Local Search Polish
        if time.time() < restart_deadline - 1.0:
            improved = True
            while improved and time.time() < restart_deadline - 1.0:
                improved = False
                for idx in range(N):
                    if time.time() >= restart_deadline - 1.0:
                        break
                    delta = compute_delta(colors, idx)
                    if delta < 0:
                        colors[idx] = 1 - colors[idx]
                        current_score += delta
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors.copy()
    
    # Final global polish on best solution
    if best_colors is not None and time.time() < deadline - 2.0:
        colors = best_colors.copy()
        current_score = get_score(colors)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(N):
                if time.time() >= deadline - 2.0:
                    break
                delta = compute_delta(colors, idx)
                if delta < 0:
                    colors[idx] = 1 - colors[idx]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END

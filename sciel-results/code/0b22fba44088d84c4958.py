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
    index_to_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for pos in aps[i]:
            index_to_aps[pos].append(i)
    
    # Convert to numpy arrays for faster iteration if needed
    # We will use a list of numpy arrays for the AP indices per position
    index_to_aps_np = [np.array(lst, dtype=np.int32) for lst in index_to_aps]

    def calc_score(coloring):
        """
        Vectorized calculation of the number of monochromatic 4-APs.
        """
        if len(aps) == 0:
            return 0
        a = aps[:, 0]
        b = aps[:, 1]
        c = aps[:, 2]
        d = aps[:, 3]
        
        ca = coloring[a]
        cb = coloring[b]
        cc = coloring[c]
        cd = coloring[d]
        
        # Monochromatic if all four are equal
        # (ca == cb) & (cb == cc) & (cc == cd)
        mono = (ca == cb) & (cb == cc) & (cc == cd)
        return int(np.sum(mono))

    def get_deltas(coloring, indices):
        """
        Calculate the score delta for flipping a set of indices.
        Returns a numpy array of deltas for each index in 'indices'.
        This is an approximation: it assumes flips are independent or calculates
        the net effect of flipping all bits in 'indices' simultaneously.
        For strict correctness in multi-flip, we should just calculate the score difference.
        However, for speed in hill climbing, we often use single-flip deltas.
        For multi-flip, let's just try the flip and check score if promising?
        No, that's slow.
        
        Alternative: Calculate the exact delta for a specific set of flips.
        This is complex. Let's stick to single-flip for the main loop, 
        and use a specific multi-flip routine that evaluates the full score
        for a small number of candidate multi-flips.
        """
        pass # Not used directly in this simplified version, relying on calc_score for multi-flips

    def try_multi_flip(coloring, best_score, best_coloring):
        """
        Attempt to improve by flipping 2 or 3 random bits simultaneously.
        Returns updated best_score, best_coloring.
        """
        current_score = calc_score(coloring)
        
        # Try a few random multi-flips
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Pick 2 or 3 distinct random indices
            k = random.choice([2, 3])
            flip_indices = random.sample(range(N), k)
            
            # Apply flip
            for idx in flip_indices:
                coloring[idx] = 1 - coloring[idx]
            
            new_score = calc_score(coloring)
            
            if new_score < best_score:
                best_score = new_score
                best_coloring = coloring.copy()
            else:
                # Revert
                for idx in flip_indices:
                    coloring[idx] = 1 - coloring[idx]
        
        return best_score, best_coloring

    # --- Phase 1: Structured Initialization ---
    # Thue-Morse sequence is known to have low arithmetic progression density.
    # t_n = parity of number of 1s in binary representation of n.
    def thue_morse(n):
        # Use numpy bit_count equivalent or simple loop
        # For N=256, simple loop is fast enough
        seq = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            # Count bits
            c = 0
            x = i
            while x:
                c += x & 1
                x >>= 1
            seq[i] = c % 2
        return seq

    # Initialize with Thue-Morse
    coloring = thue_morse(N).copy()
    current_score = calc_score(coloring)
    best_score = current_score
    best_coloring = coloring.copy()
    
    # Also try a random start just in case Thue-Morse is bad for this specific N
    random_coloring = np.random.randint(0, 2, size=N)
    rand_score = calc_score(random_coloring)
    if rand_score < best_score:
        best_score = rand_score
        best_coloring = random_coloring.copy()
        coloring = random_coloring.copy()
        current_score = rand_score

    # --- Phase 2: Deterministic Hill Climbing (Single Bit Flips) ---
    # Iterate until no improvement is found in a full pass.
    # To avoid getting stuck in a very specific local minimum immediately,
    # we can do multiple passes with random order of indices.
    
    max_passes = 100
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        improved = False
        # Randomize the order of indices to break symmetry
        indices_order = np.random.permutation(N)
        
        for idx in indices_order:
            if time.time() > deadline:
                break
            
            # Calculate delta for flipping idx
            # Only check APs containing idx
            ap_indices = index_to_aps_np[idx]
            
            if len(ap_indices) == 0:
                continue
                
            delta = 0
            # We need to check each AP containing idx
            # To speed this up, we can vectorize the check for all APs containing idx
            # Get the APs
            if len(ap_indices) > 0:
                sub_aps = aps[ap_indices]
                
                # Get current values for these APs
                # sub_aps shape: (M, 4)
                ca = coloring[sub_aps[:, 0]]
                cb = coloring[sub_aps[:, 1]]
                cc = coloring[sub_aps[:, 2]]
                cd = coloring[sub_aps[:, 3]]
                
                # Identify which APs are currently monochromatic
                is_mono_before = (ca == cb) & (cb == cc) & (cc == cd)
                
                # Simulate flip
                # We need to determine the new value for the position idx in each AP
                # Since idx is one of the 4 positions, we can figure out which one
                # However, vectorizing "which position" is tricky.
                
                # Alternative: Loop over the APs containing idx.
                # Given N=256, the number of APs per index is small (~20-30).
                # So a python loop over ~30 items is fast.
                
                delta = 0
                for ap_idx in ap_indices:
                    a, b, c, d = aps[ap_idx]
                    
                    ca = coloring[a]
                    cb = coloring[b]
                    cc = coloring[c]
                    cd = coloring[d]
                    
                    is_mono_before = (ca == cb) and (cb == cc) and (cc == cd)
                    
                    new_val = 1 - coloring[idx]
                    
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
            
            if delta < 0:
                # Improvement
                coloring[idx] = 1 - coloring[idx]
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            elif delta == 0:
                # Neutral, might help escape? 
                # In strict hill climbing, we skip. 
                # But sometimes neutral moves lead to improvements.
                # Let's accept neutral moves with low probability to add some randomness
                if random.random() < 0.1:
                    coloring[idx] = 1 - coloring[idx]
                    # current_score unchanged
                    improved = True # Count as progress to reset pass counter? No, just keep going.
            
        if not improved:
            # No single flip improves the score.
            # Try multi-flips
            best_score, best_coloring = try_multi_flip(coloring, best_score, best_coloring)
            current_score = calc_score(coloring) # Sync score
            
            # If multi-flip didn't help, try a random restart with perturbation
            # Perturb best coloring slightly
            if time.time() < deadline - 10:
                # Create a perturbed version of the best coloring
                perturbed = best_coloring.copy()
                # Flip 10% of bits
                num_flips = N // 10
                flip_idx = np.random.choice(N, num_flips, replace=False)
                perturbed[flip_idx] = 1 - perturbed[flip_idx]
                
                # Run a few steps of hill climbing on the perturbed version
                temp_coloring = perturbed
                temp_score = calc_score(temp_coloring)
                
                # Quick local search
                for _ in range(50): # 50 random flips
                    if time.time() > deadline:
                        break
                    idx = np.random.randint(0, N)
                    # Calculate delta (simplified: just flip and check score if small N? No, use delta logic)
                    # Re-use the delta logic
                    ap_indices = index_to_aps_np[idx]
                    delta = 0
                    for ap_idx in ap_indices:
                        a, b, c, d = aps[ap_idx]
                        ca = temp_coloring[a]
                        cb = temp_coloring[b]
                        cc = temp_coloring[c]
                        cd = temp_coloring[d]
                        is_mono_before = (ca == cb) and (cb == cc) and (cc == cd)
                        new_val = 1 - temp_coloring[idx]
                        if idx == a:
                            nca, ncb, ncc, ncd = new_val, cb, cc, cd
                        elif idx == b:
                            nca, ncb, ncc, ncd = ca, new_val, cc, cd
                        elif idx == c:
                            nca, ncb, ncc, ncd = ca, cb, new_val, cd
                        else:
                            nca, ncb, ncc, ncd = ca, cb, cc, new_val
                        is_mono_after = (nca == ncb) and (ncb == ncc) and (ncc == ncd)
                        if is_mono_before and not is_mono_after:
                            delta -= 1
                        elif not is_mono_before and is_mono_after:
                            delta += 1
                    
                    if delta < 0:
                        temp_coloring[idx] = 1 - temp_coloring[idx]
                        temp_score += delta
                    elif delta == 0 and random.random() < 0.5:
                        temp_coloring[idx] = 1 - temp_coloring[idx]
                    
                if temp_score < best_score:
                    best_score = temp_score
                    best_coloring = temp_coloring.copy()
                    coloring = temp_coloring.copy()
                    current_score = temp_score
            else:
                break # Out of time

    return best_coloring.tolist()
# EVOLVE-BLOCK-END

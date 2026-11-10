import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    # Store as indices 0..N-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Convert to numpy array for vectorized counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] contains indices of other elements in APs with i
    # We need to know, for a flip at i, which APs are affected.
    # An AP is affected if it contains i.
    # For each i, store the other 3 indices of every AP containing i.
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))
        
    # Convert to numpy arrays
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic 4-APs using vectorized operations."""
        # colors is a 1D numpy array of 0s and 1s
        c1 = colors[aps_arr[:, 0]]
        c2 = colors[aps_arr[:, 1]]
        c3 = colors[aps_arr[:, 2]]
        c4 = colors[aps_arr[:, 3]]
        return np.sum((c1 == c2) & (c2 == c3) & (c3 == c4))

    best_score = float('inf')
    best_coloring = None

    # 3. Structured Initialization Strategy
    # Base pattern: blocks of two (00110011...)
    # This pattern is known to break many small-step APs.
    base_pattern = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)

    while time.time() < deadline - 2.0:  # Leave 2s for final return
        # Initialize with a perturbed version of the structured pattern
        colors = base_pattern.copy()
        
        # Apply random perturbations to escape the exact pattern if it's not optimal
        num_perturb = random.randint(5, 15)
        perturb_indices = random.sample(range(N), num_perturb)
        for idx in perturb_indices:
            colors[idx] = 1 - colors[idx]
            
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = colors.copy()

        # Simulated Annealing with Block Flips
        T = 1.5
        alpha = 0.999
        min_T = 0.05
        
        # Track stagnation for kicking
        stagnation_count = 0
        last_improve_score = current_score
        
        while T > min_T and time.time() < deadline - 2.0:
            # Decide move type: Single Flip vs Block Flip
            # Higher probability of block flip if stuck
            if stagnation_count > 100:
                use_block_flip = random.random() < 0.3
            else:
                use_block_flip = random.random() < 0.05

            if use_block_flip:
                # Block Flip: Flip a contiguous segment
                block_len = random.randint(4, 16)
                start_idx = random.randint(0, N - block_len)
                end_idx = start_idx + block_len
                
                # Calculate delta for block flip
                # This is expensive if done naively, but block flips are rare.
                # We can just apply and recount if we are in a "kick" phase, 
                # or estimate. Given the frequency, we will do a quick local check 
                # or just accept the cost of a full recount for the kick if it's rare enough.
                # However, to be safe on time, let's only do full recount for kicks 
                # if we are close to the deadline, otherwise we rely on single flips 
                # mostly and use kicks as "teleports" that we accept if they look promising 
                # via a fast heuristic or just take the risk.
                
                # Actually, let's just do a single flip for speed, but if we are stuck, 
                # we do a "random restart" of the local search by flipping a few bits 
                # and accepting the score jump if it's not too bad, or just force it.
                
                # Simpler structural kick: Flip a random set of bits (not contiguous)
                # This is effectively a larger step in the solution space.
                kick_size = random.randint(3, 8)
                kick_indices = random.sample(range(N), kick_size)
                
                old_colors = colors[kick_indices].copy()
                new_colors = 1 - old_colors
                colors[kick_indices] = new_colors
                
                new_score = count_mono(colors)
                delta = new_score - current_score
                
                # Accept with probability exp(-delta/T)
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                else:
                    # Revert
                    colors[kick_indices] = old_colors
                    stagnation_count += 1
            else:
                # Single Flip
                i = random.randint(0, N - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                others = neighbors_np[i]
                if len(others) == 0:
                    delta = 0
                else:
                    cols = colors[others]
                    c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                    all_eq = (c0 == c1) & (c1 == c2)
                    val = c0
                    was_mono = all_eq & (val == old_c)
                    becomes_mono = all_eq & (val == new_c)
                    delta = np.sum(becomes_mono) - np.sum(was_mono)
                    
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[i] = new_c
                    current_score += delta
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

            T *= alpha

    return best_coloring.tolist()
# EVOLVE-BLOCK-END

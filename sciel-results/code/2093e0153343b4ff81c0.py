import time
import random
import numpy as np

def construct():
    n = 256
    start = time.time()
    deadline = start + 118.0

    # Initialize with Thue-Morse sequence
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # d >= 1, a+3d <= n
    # Indices are 0-based
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)
    if num_aps == 0:
        return color.tolist()

    # Convert APs to a numpy array for vectorized operations
    # Shape: (num_aps, 4)
    ap_array = np.array(aps, dtype=np.int32)

    # Build list of AP indices for each position
    ap_indices_at_pos = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_at_pos[pos].append(idx)

    # Helper to compute initial score and states
    def compute_initial_states(c):
        states = np.zeros(num_aps, dtype=np.int8)
        for idx in range(num_aps):
            a, b, c_, d = ap_array[idx]
            if c[a] == c[b] == c[c_] == c[d]:
                states[idx] = 1
        return states

    current_states = compute_initial_states(color)
    current_score = np.sum(current_states)

    best_color = color.copy()
    best_score = current_score

    # Iterated Local Search
    # We will alternate between Local Search and Perturbation.
    # To handle the "island" inspiration and avoid getting stuck, we will use a simple multi-start approach
    # or a more aggressive perturbation strategy.
    # However, the previous attempt at multi-bit moves was rejected as duplicate or invalid.
    # Let's stick to a robust single-bit ILS but with varied perturbation strengths and random restarts if stuck.
    # Actually, the inspiration mentions "island" changing chance of improvement.
    # A simple way to emulate this is to maintain a few "islands" (states) and occasionally cross-pollinate
    # or just use different random seeds for restarts.
    # Given the time limit, let's do a simple ILS with adaptive perturbation size.

    max_time_per_phase = 5.0
    total_deadline = deadline

    iteration = 0
    while time.time() < total_deadline - 1.0:
        iteration += 1
        
        # --- Local Search Phase ---
        # First-improvement or Best-improvement? Best-improvement is slower but more directed.
        # First-improvement with random ordering is a good balance.
        improved = True
        while improved and time.time() < total_deadline - 1.0:
            improved = False
            # Randomize order to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > total_deadline - 1.0:
                    break
                
                # Calculate delta for flipping bit i
                delta = 0
                for ap_idx in ap_indices_at_pos[i]:
                    if current_states[ap_idx] == 1:
                        # Currently mono, flipping breaks it
                        delta -= 1
                    else:
                        # Currently not mono
                        # Check if flipping i makes it mono
                        a, b, c_, d = ap_array[ap_idx]
                        # Identify the 3 other indices
                        # Since i is one of a,b,c,d
                        if a == i:
                            others = (b, c_, d)
                        elif b == i:
                            others = (a, c_, d)
                        elif c_ == i:
                            others = (a, b, d)
                        else:
                            others = (a, b, c_)
                        
                        # If the other 3 are the same, and different from i, then flipping i makes them all same
                        if color[others[0]] == color[others[1]] == color[others[2]]:
                            # Since AP is not mono, color[i] must be different from color[others[0]]
                            # So flipping i will make it match
                            delta += 1
                
                if delta < 0:
                    # Accept flip
                    color[i] = 1 - color[i]
                    current_score += delta
                    improved = True
                    
                    # Update states for all APs containing i
                    for ap_idx in ap_indices_at_pos[i]:
                        a, b, c_, d = ap_array[ap_idx]
                        if color[a] == color[b] == color[c_] == color[d]:
                            current_states[ap_idx] = 1
                        else:
                            current_states[ap_idx] = 0
                        
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
                        
        # --- Perturbation Phase ---
        # Determine perturbation size
        # Start with small, increase if stuck for many iterations without improvement
        # Or just random
        k = random.randint(3, 8)
        pert_indices = random.sample(range(n), k)
        for i in pert_indices:
            color[i] = 1 - color[i]
            # Update states
            for ap_idx in ap_indices_at_pos[i]:
                a, b, c_, d = ap_array[ap_idx]
                if color[a] == color[b] == color[c_] == color[d]:
                    current_states[ap_idx] = 1
                else:
                    current_states[ap_idx] = 0
            
        # Recalculate score
        current_score = np.sum(current_states)
        
        # If perturbation made it much worse, and we are close to deadline, maybe revert?
        # Standard ILS: accept. But let's check if we should restart completely.
        # If we are in a very bad state (score > best_score + 20), and we have time,
        # maybe revert to best and perturb again?
        # Let's just stick to standard ILS for simplicity and robustness.
        
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()

    return best_color.tolist()

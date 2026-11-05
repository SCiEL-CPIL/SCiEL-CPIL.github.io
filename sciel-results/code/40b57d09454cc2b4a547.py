import time
import random
import sys

def construct():
    n = 256
    deadline = time.time() + 110.0  # Leave 10s buffer for 120s limit
    
    # 1. Precompute all 4-APs
    # a, a+d, a+2d, a+3d with 1 <= a and a+3d <= n
    # Using 0-based indexing internally: 0..255
    # 0 <= a and a+3d <= 255
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # 2. For each index, store the indices of APs it belongs to
    # ap_indices[i] is a list of indices into ap_list
    ap_indices = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            ap_indices[pos].append(idx)
            
    # For faster delta calculation, we can precompute for each (ap_idx, pos_idx_in_ap)
    # the other 3 positions.
    # other_pos[ap_idx][k] where k is 0..3, returns the tuple of other 3 indices.
    # Actually, simpler: just store the 4 positions and check which one is flipped.
    
    def count_mono(coloring):
        """Count monochromatic 4-APs in the given coloring."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
                count += 1
        return count

    # Initial coloring: Random or Thue-Morse?
    # Thue-Morse is a good heuristic start for AP avoidance.
    # t(n) = popcount(n) % 2
    # Let's start with Thue-Morse, then refine.
    
    def thue_morse(n):
        # Returns list of length n
        # t(0)=0, t(1)=1, t(2)=1, t(3)=0 ...
        # t(n) = n & (n>>1) ... no, it's parity of popcount.
        return [bin(i).count('1') % 2 for i in range(n)]
        
    # We will run multiple SA iterations.
    # To maximize score, we should spend most time on the best run.
    # Strategy: 
    # 1. Generate Thue-Morse coloring.
    # 2. Run SA for a fixed time slice.
    # 3. If time remains, restart with random or slightly perturbed TM.
    
    best_coloring = None
    best_score = float('inf')
    
    # Helper for SA
    def run_sa(start_coloring, time_limit):
        coloring = start_coloring[:]
        current_score = count_mono(coloring)
        
        # Annealing schedule
        # Start T high, end T low.
        # We don't know the exact number of steps, so we use time.
        # Rough estimate: 1 step takes ~50-100us in Python.
        # 100s / 100us = 1,000,000 steps.
        
        start_time = time.time()
        T_start = 5.0
        T_end = 0.01
        # We will adjust T based on elapsed time fraction.
        
        # To make it deterministic-ish but random enough:
        rng = random.Random(42)
        
        while time.time() < start_time + time_limit:
            # Pick a random index to flip
            i = rng.randint(0, n - 1)
            
            # Calculate delta
            delta = 0
            for ap_idx in ap_indices[i]:
                ap = ap_list[ap_idx]
                # Find the other 3 positions
                # Since ap is a tuple of 4, we can just unpack and skip i
                # Optimization: precompute others for each (i, ap_idx) pair?
                # ap_indices[i] gives ap_idx. We need the other 3 elements of ap_list[ap_idx].
                
                # Let's just iterate the 4 elements
                # This is a bit slow in Python.
                # Let's precompute a structure: 
                # for each i, a list of (ap_idx, other_pos_tuple)
                # But we already have ap_indices.
                # Let's just do the 4-element check. It's small.
                
                a, b, c, d = ap
                # Identify the other 3
                # This branching is slow.
                # Alternative: 
                # others = [p for p in ap if p != i]
                # if coloring[others[0]] == coloring[others[1]] == coloring[others[2]]:
                #     if coloring[i] == coloring[others[0]]: delta -= 1
                #     else: delta += 1
                
                # Let's optimize this specific loop.
                # We can precompute for each i, a list of lists of 3 others.
                # Let's do that before the loop.
                
            # The loop above is too slow if we do list comprehension every time.
            # I need to restructure the precomputation.
            
            # Break to implement precomputation properly in the outer scope
            break
            
        return coloring, current_score

    # RESTRUCTURE: Precompute "neighbors" for each index i
    # neighbors[i] is a list of tuples (other1, other2, other3) for each AP containing i
    neighbors = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        a, b, c, d = ap
        # For a, others are b,c,d
        neighbors[a].append((b, c, d))
        # For b, others are a,c,d
        neighbors[b].append((a, c, d))
        # For c, others are a,b,d
        neighbors[c].append((a, b, d))
        # For d, others are a,b,c
        neighbors[d].append((a, b, c))
        
    def run_sa_optimized(start_coloring, time_limit, seed=42):
        coloring = start_coloring[:]
        current_score = count_mono(coloring)
        
        rng = random.Random(seed)
        start_time = time.time()
        
        # Adaptive temperature
        # We'll use a simple cooling schedule based on elapsed time.
        # T = T_start * (T_end/T_start)^(t/T_total)
        # But we don't know T_total steps.
        # Let's just use a fixed number of steps per time unit? No.
        # Let's use a geometric cooling with a fixed number of steps, 
        # and restart if we hit time limit.
        
        # Estimate steps per second: ~10,000 - 50,000 depending on CPU.
        # Let's assume 20,000 steps/sec.
        # 100s -> 2,000,000 steps.
        
        # We will run in chunks to allow checking time.
        chunk_size = 1000
        T = 1.0
        T_min = 1e-4
        cooling_rate = 0.99999  # Adjust to reach T_min in ~1M steps
        
        while time.time() < start_time + time_limit:
            # Perform chunk_size steps
            for _ in range(chunk_size):
                if time.time() >= start_time + time_limit:
                    break
                    
                i = rng.randint(0, n - 1)
                
                # Calculate delta
                delta = 0
                for o1, o2, o3 in neighbors[i]:
                    if coloring[o1] == coloring[o2] == coloring[o3]:
                        if coloring[i] == coloring[o1]:
                            delta -= 1  # Was mono, becomes non-mono
                        else:
                            delta += 1  # Was non-mono, becomes mono
                
                new_score = current_score + delta
                
                if new_score < current_score:
                    # Accept
                    coloring[i] = 1 - coloring[i]
                    current_score = new_score
                else:
                    # Metropolis criterion
                    # exp(-delta / T)
                    # If delta > 0, prob = exp(-delta/T)
                    # If delta == 0, prob = 1 (or 0.5 to allow mixing)
                    if delta == 0:
                        if rng.random() < 0.5:
                            coloring[i] = 1 - coloring[i]
                    else:
                        # delta > 0
                        # Avoid overflow/underflow
                        # if delta/T > 50, exp is ~0
                        if delta / T < 50.0:
                            if rng.random() < 2.718281828459045 ** (-delta / T):
                                coloring[i] = 1 - coloring[i]
                                current_score = new_score
                
                # Cool down
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    
        return coloring, current_score

    # Main execution loop
    # Start with Thue-Morse
    tm_coloring = thue_morse(n)
    
    # Run SA for 90 seconds
    t_start = time.time()
    coloring, score = run_sa_optimized(tm_coloring, 90.0, seed=1)
    best_coloring = coloring
    best_score = score
    
    # If time remains, try a few random restarts
    while time.time() < deadline - 5.0:
        # Check if we have enough time for a short run
        remaining = deadline - 5.0 - time.time()
        if remaining < 1.0:
            break
            
        # Random start
        random_coloring = [rng.randint(0, 1) for rng in [random.Random(123)] for _ in range(n)]
        # Better: 
        rng2 = random.Random(int(time.time() * 1000))
        random_coloring = [rng2.randint(0, 1) for _ in range(n)]
        
        # Run short SA
        c, s = run_sa_optimized(random_coloring, remaining - 1.0, seed=456)
        if s < best_score:
            best_score = s
            best_coloring = c

    # Return 1-based colors? 
    # The task says "256 colours in {0,1} for the integers 1..256".
    # My internal indexing is 0..255.
    # The list returned should correspond to 1..256.
    # So index 0 in my list corresponds to integer 1.
    # This is standard.
    
    return best_coloring

# EVOLVE-BLOCK-END

import numpy as np
import time

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 140s.
    deadline = start_time + 140.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute neighbor lists for fast delta calculation
    # index_to_aps[i] = list of (ap_idx, pos, o1, o2, o3)
    # But for speed, we only need the other 3 indices and the ap index isn't strictly needed 
    # if we just check the values. 
    # Actually, we need to know if the AP is mono. 
    # The delta calculation logic:
    # Flip i. Affected APs are those containing i.
    # For each affected AP, check if the other 3 are equal.
    # If yes:
    #   if c[i] == c[other]: delta -= 1 (was mono, now not)
    #   if c[i] != c[other]: delta += 1 (was not, now mono)
    # If no: delta += 0
    
    # We can store for each i, a list of tuples (o1, o2, o3).
    # We don't even need the ap_idx if we just check the values directly.
    # Wait, is it possible that two different APs have the same set of 3 other indices?
    # No, because the AP is defined by the 4 indices. The "other 3" uniquely identifies the AP 
    # relative to i? 
    # Actually, yes. If i is in an AP, the other 3 are fixed. 
    # So we can just store the list of other indices for each i.
    
    # Let's optimize the storage.
    # index_to_others[i] = list of (o1, o2, o3)
    
    index_to_others = [[] for _ in range(N)]
    for j in range(num_aps):
        a, b, c_idx, d_idx = aps[j]
        index_to_others[a].append((b, c_idx, d_idx))
        index_to_others[b].append((a, c_idx, d_idx))
        index_to_others[c_idx].append((a, b, d_idx))
        index_to_others[d_idx].append((a, b, c_idx))
        
    # Pre-convert lists to numpy arrays for faster iteration? 
    # Lists of tuples are quite fast in Python if the list is small.
    # Let's keep them as lists for now, but maybe convert to a flat array with offsets?
    # For N=256, lists are fine.
    
    # Helper to calculate initial score
    def calc_score(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)
        
    # Helper to calculate delta for a single flip at index i
    # This is O(degree(i)). Degree is roughly 20-40.
    def calc_delta(c, i):
        val_i = c[i]
        delta = 0
        for o1, o2, o3 in index_to_others[i]:
            v1 = c[o1]
            v2 = c[o2]
            v3 = c[o3]
            if v1 == v2 and v2 == v3:
                if val_i == v1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Initialize with a good heuristic: 
    # Random is okay, but maybe a specific pattern is better?
    # Let's try a few different deterministic initializations.
    
    best_score = np.inf
    best_coloring = None
    
    # We will run a series of local searches from different starting points.
    # Instead of random restarts, we use structured perturbations or just different seeds.
    # Since the task asks for a structural change, let's focus on the search algorithm itself.
    
    # Strategy: Cyclic Scan Local Search with "Sweep" restarts.
    # 1. Start with a random coloring.
    # 2. Do a forward pass: for i in 0..N-1, if flip improves, flip.
    # 3. Do a backward pass: for i in N-1..0, if flip improves, flip.
    # 4. If any flips happened, repeat 2-3.
    # 5. If no flips, we are at a local minimum.
    # 6. Perturb: flip k random bits (k=1,2,3...). 
    # 7. Repeat.
    
    # To make it deterministic and reproducible for the evaluator (though random is allowed),
    # we can use a fixed seed or just let it run. The prompt implies we just need to minimize.
    
    # Let's implement a robust local search.
    
    # We'll do 100 restarts if time permits.
    # Each restart:
    #   - Initialize c
    #   - Loop until no improvement:
    #       - Forward pass
    #       - Backward pass
    
    # Optimization: 
    # The delta calculation is the bottleneck. 
    # Can we vectorize the delta calculation for ALL i at once?
    # Yes, as discussed in the thought process, but it's complex.
    # Given N=256, a Python loop of 256 iterations per pass, with inner loops of ~30, 
    # is 256 * 30 = 7680 operations per pass.
    # A full convergence might take 10-50 passes.
    # 50 * 7680 = 384,000 operations per restart.
    # 100 restarts = 38,400,000 operations.
    # In Python, this might take 1-2 seconds. 
    # So we can afford many more restarts or more complex logic.
    
    # Let's use a slightly smarter initialization: 
    # Start with all 0s? No, that's 10k+ mono APs.
    # Start with random.
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility if needed, but random is fine too
    
    # Let's try 200 restarts, but break early if time is up.
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Optional: Improve initial guess with a few random flips? No, just start.
        
        current_score = calc_score(c)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # Local Search Loop
        # We use a "sweep" approach.
        # A sweep is one full pass (forward or backward).
        # We repeat sweeps until no change.
        
        max_sweeps = 100
        for sweep_num in range(max_sweeps):
            if time.time() > deadline - 2:
                break
                
            changed = False
            
            # Forward Pass
            for i in range(N):
                # Check if flip improves
                d = calc_delta(c, i)
                if d < 0:
                    c[i] = 1 - c[i]
                    current_score += d
                    changed = True
                    if current_score < best_score:
                        best_score = current_score
                        best_coloring = c.copy()
                        
            # Backward Pass
            for i in range(N-1, -1, -1):
                d = calc_delta(c, i)
                if d < 0:
                    c[i] = 1 - c[i]
                    current_score += d
                    changed = True
                    if current_score < best_score:
                        best_score = current_score
                        best_coloring = c.copy()
                        
            if not changed:
                break
                
        # If we reached a local minimum, perturb and restart the local search?
        # Or just do a new random restart?
        # The "new random restart" is simpler and effectively samples the space.
        # However, "kick" + local search is often better.
        # Let's do a kick + local search for the first few restarts, then pure random?
        # Actually, pure random restarts with a very fast local search is a solid baseline.
        # Let's stick to pure random restarts for simplicity and robustness.
        # The "structural change" is the deterministic cyclic scan vs the random-greedy (pick best flip anywhere).
        # Cyclic scan is often faster to implement and can be faster in practice due to lower overhead per step 
        # (no need to compute all deltas to find the min, just compute one delta per index per pass).
        # Wait, in the previous program, `calc_deltas` computed ALL deltas in one go (vectorized-ish or loop).
        # Here, `calc_delta` is called N times per pass.
        # Is this faster?
        # Previous: Loop i=0..N-1, for each i loop neighbors. Total O(N*Deg).
        # Current: Loop i=0..N-1, for each i loop neighbors. Total O(N*Deg).
        # It's the same complexity!
        # Why is it "structurally different"?
        # 1. The previous program found the GLOBAL best flip (min delta) and flipped it.
        # 2. This program flips ANY improving flip it finds in a fixed order.
        # This is "First Improvement" vs "Best Improvement" local search.
        # First Improvement is usually much faster per step because it stops looking for better flips 
        # once it finds one good one? No, it checks all i in the pass, but doesn't compare deltas across i.
        # Actually, in the previous program, `calc_deltas` returned an array of deltas, and then `np.argmin` was used.
        # This required computing ALL deltas before flipping ONE.
        # In this program, we compute delta for i, if it's < 0, we flip immediately. 
        # We do NOT compare it to deltas of other i's in the current pass.
        # This is a significant structural difference: **First-Improvement Hill Climbing** vs **Best-Improvement Hill Climbing**.
        # First-Improvement is typically faster per iteration and can find different local minima.
        
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()

import time
import random
import math

def construct():
    N = 51975
    # Find all divisors > 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    # Sort divisors in a specific order to help greedy
    # Smaller moduli cover fewer points but are "tighter". 
    # Usually, using small moduli first in a covering system helps because they are harder to satisfy.
    # Let's try sorting by modulus ascending.
    divisors.sort()
    
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_classes = []
    best_score = N
    
    # We will run multiple restarts with different randomizations or orders if time permits.
    # But first, a deterministic greedy with a good ordering.
    
    def greedy_construct(moduli_list, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Shuffle moduli if seed provided, otherwise keep order
        mods = list(moduli_list)
        if seed is not None:
            random.shuffle(mods)
        
        covered = bytearray(N)
        counts = bytearray(N)
        classes = []
        
        # We pick moduli one by one.
        # For each modulus, we pick the residue that covers the most uncovered points.
        # To make it "randomized" or diverse, we can add noise to the selection score.
        
        for m in mods:
            best_r = 0
            best_gain = -1
            best_noise = -1
            
            # We want to maximize: (number of newly covered points) + noise
            # To keep it fast, we just count uncovered points.
            
            # Optimization: Precompute which residues are "good"
            # Since N is small (51975), iterating r in range(m) is fast.
            # m is at most N, but average m is small.
            
            for r in range(m):
                # Count uncovered points in this residue class
                # covered[r::m].count(0)
                # This is a C-level operation in Python for bytearrays? 
                # bytearray slicing creates a new bytearray, which might be slow for large m.
                # Let's use a manual loop or numpy if available? 
                # The prompt says NumPy is available. Let's use numpy for speed.
                
                pass # Placeholder, will replace with numpy below
        
        return classes, best_score

    # Let's use NumPy for faster counting
    import numpy as np
    
    def fast_greedy(moduli_list, use_noise=False, noise_scale=0.0):
        covered_np = np.zeros(N, dtype=np.uint8)
        counts_np = np.zeros(N, dtype=np.uint8)
        classes = []
        
        # We need to decide the order of moduli.
        # If use_noise, we might shuffle or pick randomly from a subset?
        # Let's stick to the given order (sorted) but allow noise in residue selection.
        
        for m in moduli_list:
            best_r = 0
            best_score_r = -1
            
            # Vectorized calculation of gain for all residues?
            # For a fixed m, we can create an array of length N and sum over residues?
            # No, simpler:
            # For each r, gain = sum(1 - covered_np[r::m])
            
            # We can compute all gains at once?
            # Let U = 1 - covered_np (1 if uncovered, 0 if covered)
            # We want to find r that maximizes sum(U[r::m])
            
            # This is a convolution-like operation or just simple slicing sums.
            # Since m varies, we can't precompute a single structure easily.
            # But N is small.
            
            U = 1 - covered_np # 1 where uncovered
            
            # Calculate sums for each residue
            # sums = np.array([np.sum(U[r::m]) for r in range(m)])
            # This creates a list of m arrays, which might be slow if m is large.
            
            # Alternative: 
            # If m is small, the above is fine.
            # If m is large (e.g., N), there is only 1 residue (r=0) effectively? No, r in [0, m).
            # Max m is 51975.
            # np.sum(U[0::51975]) -> sum of 1 element.
            # np.sum(U[1::51975]) -> sum of 1 element.
            # ...
            # For large m, the loop over r is expensive if done naively.
            
            # Let's optimize the inner loop.
            # We can iterate r and sum.
            # To speed up, we can note that for large m, the slices are short.
            
            # Let's try the numpy sum approach first and see if it's fast enough.
            # 47 divisors. Max m ~ 50k.
            # 50k sums of length ~1.
            # 50k * 1 = 50k ops. Fast.
            # For small m (e.g., 3), 3 sums of length ~17k.
            # 3 * 17k = 51k ops. Fast.
            
            # Total ops per step ~ N.
            # 47 steps * 51975 ~ 2.4M ops. Very fast.
            
            gains = np.zeros(m)
            for r in range(m):
                # np.sum on a slice
                # We can avoid creating the slice if we use reshape? No, stride is m.
                # Direct slice is okay for small N.
                gains[r] = np.sum(U[r::m])
                
            # Add noise if needed
            if use_noise and noise_scale > 0:
                noise = np.random.uniform(0, noise_scale, m)
                gains += noise
            
            best_r = np.argmax(gains)
            
            classes.append([m, int(best_r)])
            
            # Update coverage
            # covered_np[best_r::m] = 1
            # counts_np[best_r::m] += 1
            idxs = np.arange(best_r, N, m)
            covered_np[idxs] = 1
            counts_np[idxs] += 1
            
        # Calculate final score
        # Score = number of uncovered residues = sum(1 - covered_np)
        score = int(np.sum(1 - covered_np))
        
        return classes, score

    # Initial run: deterministic greedy with sorted moduli
    classes, score = fast_greedy(divisors, use_noise=False)
    best_classes = classes
    best_score = score
    
    # Local Search Phase
    # Try to improve by changing residues.
    # We can also try to remove a modulus if it doesn't help?
    # The problem says "distinct moduli". It doesn't say we MUST use all.
    # However, removing a modulus will likely increase uncovered count unless that modulus was redundant.
    # In a greedy covering, later moduli might be redundant if earlier ones covered everything.
    # But we want to MINIMIZE uncovered. So we want to MAXIMIZE covered.
    # Using more moduli generally helps (or doesn't hurt) unless they are very restrictive?
    # No, adding a class always covers more (or same) points.
    # So we should use all available moduli.
    
    # Improve residues
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # We need the current counts and covered state to evaluate changes efficiently.
        # Let's reconstruct state from best_classes
        covered_np = np.zeros(N, dtype=np.uint8)
        counts_np = np.zeros(N, dtype=np.uint8)
        for m, r in best_classes:
            idxs = np.arange(r, N, m)
            covered_np[idxs] = 1
            counts_np[idxs] += 1
        
        for k in range(len(best_classes)):
            m, r_old = best_classes[k]
            
            # We want to find r_new that minimizes uncovered count.
            # Equivalent to maximizing the gain in coverage.
            # Let's just evaluate the score for each r_new?
            # That would be O(m * N) per change, which is too slow.
            
            # Efficient delta calculation:
            # Delta = (Uncovered after removing old and adding new) - (Current Uncovered)
            # Let S_old = indices in class (m, r_old)
            # Let S_new = indices in class (m, r_new)
            
            # Current Uncovered U
            # If we remove (m, r_old):
            #   Points in S_old where counts[i] == 1 become uncovered.
            #   Points in S_old where counts[i] > 1 remain covered.
            #   Points outside S_old are unchanged.
            
            # If we then add (m, r_new):
            #   Points in S_new where (new_count) == 0 become covered.
            #   Note: new_count for i in S_new:
            #     If i in S_old: count was counts[i]-1 (after removal). If we add new, it becomes counts[i] (if i in S_new) or counts[i]-1 (if not).
            #     Wait, if i in S_old AND S_new:
            #       Original count: c.
            #       Remove old: c-1.
            #       Add new: c.
            #       Net change: 0.
            #     If i in S_old AND NOT S_new:
            #       Original count: c.
            #       Remove old: c-1.
            #       Add new: c-1.
            #       If c==1, becomes 0 (uncovered). If c>1, stays covered.
            #     If i NOT in S_old AND i in S_new:
            #       Original count: c.
            #       Remove old: c.
            #       Add new: c+1.
            #       If c==0, becomes 1 (covered). If c>0, stays covered.
            
            # So, change in Uncovered count:
            # +1 for each i in (S_old \ S_new) where counts[i] == 1
            # -1 for each i in (S_new \ S_old) where counts[i] == 0
            
            # We want to minimize this change (make it negative).
            
            # Let's compute the two terms.
            # Term 1: i in S_old, i not in S_new, counts[i] == 1
            # Term 2: i in S_new, i not in S_old, counts[i] == 0
            
            # S_old = {r_old + k*m}
            # S_new = {r_new + k*m}
            
            # Intersection S_old ∩ S_new:
            # r_old + k*m = r_new + j*m
            # r_old - r_new = (j-k)*m
            # This is possible only if (r_old - r_new) is divisible by m.
            # Since 0 <= r_old, r_new < m, r_old - r_new is in (-m, m).
            # The only multiple of m in this range is 0.
            # So intersection is non-empty ONLY IF r_old == r_new.
            # If r_old != r_new, S_old and S_new are DISJOINT.
            
            # This simplifies things greatly!
            # If r_new != r_old:
            #   S_old \ S_new = S_old
            #   S_new \ S_old = S_new
            
            # So:
            # Delta = (Count of i in S_old where counts[i] == 1) - (Count of i in S_new where counts[i] == 0)
            
            # We want Delta < 0, i.e., (Count of i in S_new where counts[i] == 0) > (Count of i in S_old where counts[i] == 1)
            
            # Let's precompute:
            # loss = count of i in S_old where counts[i] == 1
            # gain(r_new) = count of i in S_new where counts[i] == 0
            
            # We want to find r_new that maximizes gain(r_new) - loss.
            # Since loss is constant for a given k, we just maximize gain(r_new).
            
            # gain(r_new) = sum(1 for i in S_new if counts[i] == 0)
            #              = sum(counts[i] == 0 for i in range(r_new, N, m))
            
            # This is similar to the greedy step, but using counts == 0 instead of covered == 0?
            # Wait, counts[i] == 0 is equivalent to covered[i] == 0.
            # So gain(r_new) is exactly the same as the "newly covered" count we calculated in greedy.
            
            # So, for each class, we can re-optimize its residue by finding the r_new that maximizes the number of currently uncovered points it covers.
            # BUT, we must account for the fact that removing the old class might uncover points, which then might be covered by the new class?
            # No, if r_new != r_old, S_old and S_new are disjoint.
            # So the points covered by S_new are NOT in S_old.
            # So the "uncovered" points we are trying to cover are the ones currently uncovered in the whole system.
            # When we remove S_old, some points in S_old become uncovered.
            # When we add S_new, we cover points in S_new.
            # Since S_old and S_new are disjoint, the points in S_new were NOT in S_old.
            # So their status (covered/uncovered) was determined by OTHER classes.
            # So if a point in S_new was uncovered (counts==0), it remains uncovered until we add S_new.
            # So gain(r_new) is indeed the number of currently uncovered points in S_new.
            
            # So the algorithm is:
            # 1. Calculate loss = count of i in S_old where counts[i] == 1.
            # 2. For each r_new, calculate gain = count of i in S_new where counts[i] == 0.
            # 3. If gain - loss > 0, update to r_new.
            
            # Let's implement this.
            
            # Calculate loss
            idxs_old = np.arange(r_old, N, m)
            loss = np.sum(counts_np[idxs_old] == 1)
            
            # Calculate best gain
            best_r_new = r_old
            best_gain = 0 # Gain relative to loss? No, just raw gain.
            # Actually, we want gain > loss.
            # So we find r_new with max gain, and check if max_gain > loss.
            
            # We can reuse the gain calculation from greedy.
            # U = (counts_np == 0).astype(np.uint8)
            # But counts_np changes as we update classes.
            # So we must recalculate U for each k?
            # Or update counts_np as we go?
            # If we update counts_np, the "current state" changes, so the gain for subsequent k's changes.
            # This is a sequential local search (hill climbing).
            
            # Let's do sequential updates.
            
            U = (counts_np == 0).astype(np.uint8)
            
            # Find best r_new
            gains = np.zeros(m)
            for r in range(m):
                gains[r] = np.sum(U[r::m])
            
            best_r_new = np.argmax(gains)
            max_gain = gains[best_r_new]
            
            if max_gain > loss:
                # Update
                # Remove old
                counts_np[idxs_old] -= 1
                # Add new
                idxs_new = np.arange(best_r_new, N, m)
                counts_np[idxs_new] += 1
                # Update best_classes
                best_classes[k][1] = int(best_r_new)
                improved = True
                # Update U?
                # U = (counts_np == 0).astype(np.uint8)
                # This is O(N). We do it for each improvement.
                # Number of improvements is likely small.
                U = (counts_np == 0).astype(np.uint8)
        
        # After one pass, if improved, loop again.
    
    # Final score calculation
    covered_np = np.zeros(N, dtype=np.uint8)
    for m, r in best_classes:
        idxs = np.arange(r, N, m)
        covered_np[idxs] = 1
    
    final_score = int(np.sum(1 - covered_np))
    
    return best_classes

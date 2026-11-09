import numpy as np
import time
import random

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    num_divs = len(divisors)
    available_moduli = set(divisors)
    
    # State: list of [m, r]
    # covered: array of size N, counts how many times each residue is covered
    # score: number of zeros in covered
    
    def score_and_cover(classes):
        covered = np.zeros(N, dtype=np.int8)
        for m, r in classes:
            covered[r::m] += 1
        score = np.count_nonzero(covered == 0)
        return score, covered

    def get_delta_change(classes, covered, idx, new_m, new_r, is_add=False):
        """
        Calculate the change in uncovered count (score) for a move.
        If is_add is False, we are replacing class at idx with (new_m, new_r).
        If is_add is True, we are adding (new_m, new_r) to the list.
        """
        # Note: This function assumes 'covered' reflects the state BEFORE the move.
        
        if is_add:
            # Adding (new_m, new_r)
            # Indices covered: new_r, new_r+new_m, ...
            # We gain coverage where covered[x] == 0
            # Delta = - (count of x in new class where covered[x] == 0)
            indices = np.arange(new_r, N, new_m)
            gain = np.count_nonzero(covered[indices] == 0)
            return -gain

        else:
            # Replacing class at idx: (old_m, old_r) -> (new_m, new_r)
            old_m, old_r = classes[idx]
            
            # If same class, delta is 0
            if old_m == new_m and old_r == new_r:
                return 0
            
            # We need to remove old class and add new class.
            # But careful: if new_m is already used by another class, that's fine, 
            # the problem says "distinct moduli", so we cannot add a class with a modulus 
            # already in the list unless we are replacing that same modulus's class.
            # The move "replace class at idx" implies the modulus remains the same?
            # No, the problem says "distinct moduli". 
            # Usually, local search moves are:
            # 1. Change residue of an existing modulus. (Modulus stays same)
            # 2. Add a new modulus (not currently used).
            # 3. Remove an existing modulus.
            # 4. Swap? (Complex)
            
            # The prompt says "distinct moduli m". 
            # So if I have class [m1, r1] and [m2, r2], m1 != m2.
            # Move 1: Change r1 to r1'. m1 stays.
            # Move 2: Add [m3, r3] where m3 not in {m1, m2}.
            # Move 3: Remove [m1, r1].
            
            # So "replace class at idx" should only change the residue if we stick to valid moves.
            # However, we can also consider swapping which modulus is used?
            # If I remove m1 and add m3, that's effectively a swap.
            
            # Let's implement specific moves to avoid invalid states.
            pass

    # We will implement a Simulated Annealing style search.
    # State: list of tuples (m, r)
    # We maintain 'covered' array and 'score'.
    
    # Initial Greedy Solution
    classes = []
    covered = np.zeros(N, dtype=np.int8)
    score = N
    used_moduli = set()
    
    # Greedy Phase
    available = set(divisors)
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        
        # Optimization: We can't iterate all m, r if it's too slow.
        # But N is small enough (51975) and divisors are few (47).
        # Max r is 51975. Total iterations ~ 47 * 50000 ~ 2.3M.
        # Each iteration involves slicing and counting.
        # Let's try to do it faster.
        
        for m in available:
            # For a given m, we want to find r that maximizes coverage of zeros.
            # This is equivalent to minimizing sum(covered[r::m])? No.
            # We want to maximize count of zeros in covered[r::m].
            # count_zeros = len(indices) - count_nonzeros.
            # So we want to minimize count_nonzeros.
            
            min_nonzero = N
            best_r_for_m = 0
            
            # We can compute this using FFT or just loop?
            # Looping r from 0 to m-1.
            # m can be large (51975).
            # If m is large, the number of residues r is large, but the slice is small.
            # If m is small, the slice is large, but number of r is small.
            
            # Let's just loop.
            for r in range(m):
                # indices = r, r+m, ...
                # We need count of non-zeros in covered[r::m]
                # covered is numpy array.
                # covered[r::m] creates a view? No, it creates a copy if not contiguous?
                # Actually, slicing with step creates a copy in numpy?
                # No, it's a view if possible, but counting is O(N/m).
                
                # Let's try to optimize this.
                # We can precompute for each m? No, covered changes.
                
                # Just do it.
                # To speed up, we can break early if we find 0 non-zeros (perfect coverage for this m)
                # But that's rare.
                
                # Optimization:
                # If m is small, N/m is large.
                # If m is large, N/m is small.
                
                # Let's use a helper to count non-zeros in a strided slice.
                # np.count_nonzero(covered[r::m])
                
                nz = np.count_nonzero(covered[r::m])
                if nz < min_nonzero:
                    min_nonzero = nz
                    best_r_for_m = r
                    if min_nonzero == 0:
                        break # Can't do better than 0 non-zeros (all zeros covered)
            
            gain = len(range(0, N, m)) - min_nonzero # Total elements - non-zeros = zeros
            # Actually, gain is the number of NEWLY covered elements.
            # Newly covered = elements that were 0 and become 1.
            # So gain = count of zeros in covered[r::m].
            # count_zeros = total_len - count_nonzeros.
            total_len = (N - 1 - r) // m + 1 if r < N else 0
            gain = total_len - min_nonzero
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available.remove(best_m)
        
        # Update covered and score
        covered[best_r::best_m] += 1
        # Recalculate score? Or update?
        # We know we covered 'best_gain' new elements.
        # But some might have been covered by others? No, we only count if it was 0.
        # Wait, the greedy logic above calculated gain based on CURRENT covered.
        # So score -= best_gain.
        score -= best_gain

    # Now we have a starting point.
    # Let's do Local Search / Simulated Annealing.
    
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = score
    best_classes = [c[:] for c in classes]
    
    # Current state
    curr_classes = [c[:] for c in classes]
    curr_covered = covered.copy()
    curr_score = score
    curr_used_moduli = set(used_moduli)
    
    # Temperature for Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    # Precompute indices for all possible (m, r)? No, too much memory.
    # We will calculate delta on the fly.
    
    while time.time() < deadline:
        # Try to find a move
        # Moves:
        # 1. Change residue of an existing class.
        # 2. Add a new class (modulus not used).
        # 3. Remove an existing class.
        
        best_move_delta = 0 # We want to minimize score, so negative delta is good
        best_move_type = None # 1, 2, 3
        best_move_idx = None
        best_move_new_val = None
        
        # 1. Change residue
        for i, (m, r) in enumerate(curr_classes):
            # Try other residues
            # To speed up, we can sample residues if m is large?
            # Or just iterate all.
            # If m is large, N/m is small, so counting is fast.
            # If m is small, N/m is large, but m is small, so few residues to try.
            
            # We can skip if m is very large and we've already tried many?
            # Let's just iterate all r in range(m).
            
            # Optimization: If m > N/2, there is only 1 or 2 residues?
            # No, r is in [0, m).
            
            # We can calculate the "potential" gain for each r.
            # We want to maximize:
            # - (count of x in new_r::m where curr_covered[x]==1) # Becomes uncovered if only covered by this class
            # + (count of x in new_r::m where curr_covered[x]==0) # Becomes covered
            
            # Wait, the delta for changing r to r' for class i (modulus m):
            # Old set A = r::m
            # New set B = r'::m
            # Delta = 
            #   Sum_{x in A \ B} (1 if curr_covered[x]==1 else 0)  # Lost coverage, becomes uncovered
            # - Sum_{x in B \ A} (1 if curr_covered[x]==0 else 0)  # Gained coverage, was uncovered
            
            # This is expensive to calculate for all r'.
            # Alternative:
            # Just calculate the score change by simulating?
            # No, too slow.
            
            # Let's use the fact that we only change one class.
            # Let C be the coverage array.
            # Let v[x] = 1 if C[x] > 1 else 0. (Covered by others)
            # Let u[x] = 1 if C[x] == 0 else 0. (Uncovered)
            
            # If we remove class i (set A):
            # Uncovered increases by count of x in A where C[x]==1.
            # Let L = count_{x in A} (C[x]==1)
            
            # If we add class j (set B):
            # Uncovered decreases by count of x in B where C[x]==0.
            # Let G = count_{x in B} (C[x]==0)
            
            # For changing r to r':
            # Delta = L_old - L_new_part? No.
            # Delta = (Count_{x in A \ B} C[x]==1) - (Count_{x in B \ A} C[x]==0)
            
            # This is still hard to compute fast for all r'.
            
            # Let's try a different heuristic:
            # Randomly pick a class and a new residue, calculate delta, accept if better.
            # This is Random Local Search.
            
            pass
            
        # Given the complexity of efficient delta calculation for all residues, 
        # and the time limit, let's use a simpler Random Restart + Hill Climbing 
        # where we only evaluate a few random moves per step.
        
        # Actually, the greedy solution is often quite good.
        # Let's try to improve by just trying to add/remove classes randomly and see if score improves.
        
        # Simpler Local Search:
        # 1. Pick a random class index i.
        # 2. Pick a random new residue r' for that class.
        # 3. Calculate delta.
        # 4. If delta < 0, accept.
        # 5. Also try adding a random unused modulus.
        # 6. Also try removing a random class.
        
        # To make it faster, we will calculate delta using numpy operations on the specific slices.
        
        # We'll do 10000 such random steps or until time runs out.
        
        steps = 0
        while steps < 5000 and time.time() < deadline:
            steps += 1
            move_type = random.choice([1, 1, 1, 2, 3]) # Bias towards changing residue
            
            if move_type == 1:
                if not curr_classes:
                    continue
                i = random.randint(0, len(curr_classes) - 1)
                m, r = curr_classes[i]
                if m == 1: continue # Should not happen
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    continue
                
                # Calculate delta
                # Old slice: r::m
                # New slice: r_new::m
                
                # We need:
                # 1. Count of x in r::m where curr_covered[x] == 1 (these will become uncovered if not covered by new slice)
                # 2. Count of x in r_new::m where curr_covered[x] == 0 (these will become covered)
                # 3. Count of x in intersection where curr_covered[x] == 1? 
                #    If x is in both, it stays covered. 
                #    So we only care about x in Old \ New and New \ Old.
                
                # Efficient way:
                # Create a mask for Old \ New and New \ Old?
                # Too slow.
                
                # Let's just simulate the change on a copy? No, O(N).
                
                # Alternative:
                # We can approximate or just accept if it looks promising?
                # No, we need exact score.
                
                # Let's use the property that N is small.
                # We can just update the covered array and recalculate score?
                # O(N) per step. 5000 steps * 51975 ~ 2.5 * 10^8 operations.
                # In Python, this might be too slow.
                
                # Let's optimize the delta calculation.
                # Delta = 
                #   Sum_{x in Old, x not in New} (curr_covered[x] == 1)
                # - Sum_{x in New, x not in Old} (curr_covered[x] == 0)
                
                # We can compute:
                # S_old = curr_covered[r::m]
                # S_new = curr_covered[r_new::m]
                
                # This creates arrays of size N/m.
                # Then we need to find which indices are in Old but not New.
                # Indices in Old: r, r+m, ...
                # Indices in New: r_new, r_new+m, ...
                
                # Intersection: x such that x ≡ r (mod m) and x ≡ r_new (mod m).
                # This implies r ≡ r_new (mod m).
                # Since r, r_new in [0, m), intersection is non-empty only if r == r_new.
                # But we checked r_new != r.
                # So Old and New are DISJOINT!
                
                # Wait, is that true?
                # x = r + k*m
                # x = r_new + j*m
                # r + k*m = r_new + j*m => r - r_new = (j-k)*m
                # Since |r - r_new| < m, the only solution is r = r_new and j=k.
                # So yes, if r != r_new, the sets of indices are disjoint.
                
                # This simplifies things greatly!
                # Delta = 
                #   Count_{x in Old} (curr_covered[x] == 1)  # All old indices are lost
                # - Count_{x in New} (curr_covered[x] == 0)  # All new indices are gained
                
                # So:
                old_slice = curr_covered[r::m]
                new_slice = curr_covered[r_new::m]
                
                lost = np.count_nonzero(old_slice == 1)
                gained = np.count_nonzero(new_slice == 0)
                
                delta = lost - gained
                
                if delta < 0:
                    # Apply move
                    curr_covered[r::m] -= 1
                    curr_covered[r_new::m] += 1
                    curr_classes[i][1] = r_new
                    curr_score += delta
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
                    break # Restart loop? Or continue?
                    # Usually, we accept and continue.
                    
            elif move_type == 2:
                # Add a class
                unused = [m for m in divisors if m not in curr_used_moduli]
                if not unused:
                    continue
                m_new = random.choice(unused)
                r_new = random.randint(0, m_new - 1)
                
                new_slice = curr_covered[r_new::m_new]
                gained = np.count_nonzero(new_slice == 0)
                delta = -gained
                
                if delta < 0:
                    # Apply move
                    curr_covered[r_new::m_new] += 1
                    curr_classes.append([m_new, r_new])
                    curr_used_moduli.add(m_new)
                    curr_score += delta
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
                    break
            else:
                # Remove a class
                if not curr_classes:
                    continue
                i = random.randint(0, len(curr_classes) - 1)
                m, r = curr_classes[i]
                
                old_slice = curr_covered[r::m]
                lost = np.count_nonzero(old_slice == 1)
                delta = lost
                
                if delta < 0: # Wait, removing a class increases uncovered count (bad) or keeps it same?
                    # If we remove a class, we lose coverage.
                    # Uncovered count increases by number of x in Old where covered[x]==1.
                    # So delta = lost.
                    # We want delta < 0? No, we want to minimize score.
                    # If delta > 0, score increases (bad).
                    # If delta == 0, score same (neutral).
                    # We only remove if it doesn't hurt? Or if it allows adding a better one?
                    # For now, only remove if delta == 0? Or if we are stuck?
                    # Actually, removing a class that covers nothing new (all its elements covered by others) has delta=0.
                    # Removing a class that covers unique elements has delta>0.
                    # So we should only remove if delta == 0?
                    # Or maybe we should try to remove and then add a different one?
                    # Let's only remove if delta == 0 to simplify.
                    if delta == 0:
                         curr_covered[r::m] -= 1
                         curr_used_moduli.remove(m)
                         curr_classes.pop(i)
                         # Score unchanged
                         break
                    else:
                         # Don't accept
                         pass

        # If we broke out of the inner loop, we made a move.
        # We can restart the random steps counter.
        steps = 0

    return best_classes

# Run construct to test? No, just define it.
# The function is construct.

# One issue: The greedy phase in the code above might be slow if implemented naively.
# Let's refine the greedy phase to be faster.
# We can precompute the "potential" of each (m, r) pair? No, covered changes.
# But we can speed up the inner loop of greedy.

# Let's rewrite the whole thing cleanly.

import numpy as np
import time
import random

def construct():
    N = 51975
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    # Greedy Initialization
    classes = []
    covered = np.zeros(N, dtype=np.int8)
    score = N
    used_moduli = set()
    available = set(divisors)
    
    # To speed up greedy, we can prioritize smaller moduli? 
    # Smaller moduli cover more residues per class (N/m is larger).
    # But they are also more likely to overlap.
    # Let's just iterate.
    
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        
        for m in available:
            # Find r that maximizes gain
            # Gain = count of zeros in covered[r::m]
            
            # We can compute this for all r at once?
            # Let's try to vectorize.
            # For a fixed m, we want to find r such that sum(covered[r::m] == 0) is max.
            
            # We can create an array of length m, where arr[r] = count of zeros in covered[r::m]
            # This is a circular convolution or similar?
            # Or just loop r.
            
            # Optimization: If m is large, N/m is small.
            # If m is small, N/m is large.
            
            # Let's just loop r.
            # To make it faster, we can skip if m > N/2?
            # If m > N/2, N/m is 1 or 2.
            
            min_nonzero = N
            best_r_for_m = 0
            
            # We can use numpy to count non-zeros for all residues?
            # No, easy way is to loop.
            
            # But we can break early if we find a r with 0 non-zeros.
            
            for r in range(m):
                # Check if we can improve
                # We want to minimize non-zeros.
                
                # Optimization: 
                # If we have already found a gain of (N-1-r)//m + 1, we can't do better.
                # Max possible gain for this m is ceil(N/m).
                
                # Let's just do it.
                nz = np.count_nonzero(covered[r::m])
                if nz < min_nonzero:
                    min_nonzero = nz
                    best_r_for_m = r
                    if min_nonzero == 0:
                        break
            
            total_len = (N - 1 - best_r_for_m) // m + 1
            gain = total_len - min_nonzero
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available.remove(best_m)
        
        covered[best_r::best_m] += 1
        score -= best_gain

    # Local Search
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = score
    best_classes = [c[:] for c in classes]
    
    curr_classes = [c[:] for c in classes]
    curr_covered = covered.copy()
    curr_score = score
    curr_used_moduli = set(used_moduli)
    
    # Random Local Search
    # We will perform random moves and accept if they improve the score.
    # To avoid getting stuck, we can also accept neutral moves (delta=0) with some probability.
    
    # We'll use a counter to limit steps per "epoch"
    
    while time.time() < deadline:
        steps = 0
        while steps < 10000 and time.time() < deadline:
            steps += 1
            move_type = random.choice([1, 1, 1, 2, 3])
            
            if move_type == 1:
                if not curr_classes:
                    continue
                i = random.randint(0, len(curr_classes) - 1)
                m, r = curr_classes[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    continue
                
                old_slice = curr_covered[r::m]
                new_slice = curr_covered[r_new::m]
                
                lost = np.count_nonzero(old_slice == 1)
                gained = np.count_nonzero(new_slice == 0)
                
                delta = lost - gained
                
                if delta < 0:
                    curr_covered[r::m] -= 1
                    curr_covered[r_new::m] += 1
                    curr_classes[i][1] = r_new
                    curr_score += delta
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
                    steps = 0 # Reset steps on success
            elif move_type == 2:
                unused = [m for m in divisors if m not in curr_used_moduli]
                if not unused:
                    continue
                m_new = random.choice(unused)
                r_new = random.randint(0, m_new - 1)
                
                new_slice = curr_covered[r_new::m_new]
                gained = np.count_nonzero(new_slice == 0)
                delta = -gained
                
                if delta < 0:
                    curr_covered[r_new::m_new] += 1
                    curr_classes.append([m_new, r_new])
                    curr_used_moduli.add(m_new)
                    curr_score += delta
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
                    steps = 0
            else:
                if not curr_classes:
                    continue
                i = random.randint(0, len(curr_classes) - 1)
                m, r = curr_classes[i]
                
                old_slice = curr_covered[r::m]
                lost = np.count_nonzero(old_slice == 1)
                delta = lost
                
                if delta == 0:
                    curr_covered[r::m] -= 1
                    curr_used_moduli.remove(m)
                    curr_classes.pop(i)
                    steps = 0

    return best_classes

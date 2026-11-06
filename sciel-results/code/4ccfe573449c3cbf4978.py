import numpy as np
import time
import random
import math

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # Generate divisors of 51975 (odd divisors > 1)
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # cover_count[x] = number of classes covering residue x
    cover_count = np.zeros(N, dtype=np.int8)
    uncovered = N
    classes = [] # list of [m, r]
    selected_moduli = set()
    unselected_moduli = list(divisors)
    
    def get_uncovered_count():
        return int(np.count_nonzero(cover_count == 0))

    def add_class(m, r):
        """Adds class (m, r) to the system. Returns delta in uncovered count."""
        nonlocal uncovered
        # Count how many of these were previously uncovered
        mask = cover_count[r::m] == 0
        delta = int(np.count_nonzero(mask))
        # Update counts
        cover_count[r::m] += 1
        uncovered -= delta
        classes.append([m, r])
        selected_moduli.add(m)
        unselected_moduli.remove(m)
        return delta

    def remove_class(m, r):
        """Removes class (m, r) from the system. Returns delta in uncovered count (negative if uncovered increased)."""
        nonlocal uncovered
        # Count how many of these will become uncovered
        mask = cover_count[r::m] == 1
        delta = -int(np.count_nonzero(mask))
        # Update counts
        cover_count[r::m] -= 1
        uncovered += int(np.count_nonzero(mask))
        # Remove from classes list
        # Find index
        for i, c in enumerate(classes):
            if c[0] == m and c[1] == r:
                classes.pop(i)
                break
        selected_moduli.remove(m)
        unselected_moduli.append(m)
        return delta

    def find_best_residue(m):
        """Finds residue r for m that minimizes uncovered (maximizes gain) given current cover_count."""
        max_gain = -1
        best_r = 0
        for r in range(m):
            gain = int(np.count_nonzero(cover_count[r::m] == 0))
            if gain > max_gain:
                max_gain = gain
                best_r = r
        return best_r, max_gain

    # 1. Greedy Initialization
    while unselected_moduli:
        if time.time() > deadline:
            break
        best_gain = -1
        best_m = None
        best_r = None
        for m in unselected_moduli:
            r, gain = find_best_residue(m)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        if best_m is None or best_gain <= 0:
            break
        add_class(best_m, best_r)
        if uncovered == 0:
            return classes

    # 2. Simulated Annealing
    T = 50.0
    T_min = 0.01
    alpha = 0.99
    best_uncovered = uncovered
    best_classes = [c[:] for c in classes]
    
    # Precompute indices for faster random access?
    # We can just use random.choice on lists
    
    while T > T_min and time.time() < deadline:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Change residue of a random selected modulus
            if not classes: continue
            idx = random.randrange(len(classes))
            m, r_old = classes[idx]
            r_new = random.randrange(m)
            if r_new == r_old: continue
            
            # Try move
            # Remove old
            mask_old = cover_count[r_old::m] == 1
            lost_old = int(np.count_nonzero(mask_old))
            cover_count[r_old::m] -= 1
            uncovered += lost_old
            
            # Add new
            mask_new = cover_count[r_new::m] == 0
            gained_new = int(np.count_nonzero(mask_new))
            cover_count[r_new::m] += 1
            uncovered -= gained_new
            
            delta_E = (uncovered - best_uncovered) # Wait, delta_E should be relative to current state before move
            # Let's track current_uncovered_before
            # Re-implementation for clarity:
            # Store state before
            # Actually, easier to just calculate delta directly
            
            # Re-doing logic for delta_E calculation properly
            # We modified cover_count. We need to revert if rejected.
            # Let's calculate delta_E = new_uncovered - old_uncovered
            # We updated 'uncovered' variable.
            # delta_E = uncovered - (uncovered - lost_old + gained_new) ? No.
            
            # Let's use a helper to calculate delta without modifying state first?
            # No, modifying is faster.
            
            # Let's restart this block with proper delta tracking
            # Undo the changes
            cover_count[r_new::m] -= 1
            cover_count[r_old::m] += 1
            uncovered -= gained_new
            uncovered -= lost_old
            
            # Calculate delta_E
            # Removing r_old: +lost_old to uncovered
            # Adding r_new: -gained_new to uncovered
            # But gained_new depends on state after removing r_old.
            
            # Step 1: Remove r_old
            cover_count[r_old::m] -= 1
            # State A
            # Step 2: Add r_new
            # Count zeros in r_new::m in State A
            gained_new = int(np.count_nonzero(cover_count[r_new::m] == 0))
            cover_count[r_new::m] += 1
            # State B
            # Step 3: Undo
            cover_count[r_new::m] -= 1
            cover_count[r_old::m] += 1
            
            # Delta E = (uncovered_A + gained_new) - uncovered_Start
            # uncovered_A = uncovered_Start + lost_old
            # So new_uncovered = uncovered_Start + lost_old - gained_new
            # delta_E = lost_old - gained_new
            
            lost_old = int(np.count_nonzero(cover_count[r_old::m] == 1))
            # Note: cover_count is currently in Start State
            # To get gained_new correctly, we need to know how many zeros are in r_new::m
            # after removing r_old.
            # Zeros in r_new::m (Start) might become 1s if they were covered only by r_old?
            # No, r_old and r_new are different residues mod m. They are disjoint sets of residues.
            # So removing r_old does NOT affect cover_count values in r_new::m.
            # Thus, gained_new is simply count of zeros in r_new::m in Start State.
            
            gained_new = int(np.count_nonzero(cover_count[r_new::m] == 0))
            
            delta_E = lost_old - gained_new
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                cover_count[r_old::m] -= 1
                cover_count[r_new::m] += 1
                uncovered += lost_old - gained_new
                classes[idx][1] = r_new
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_classes = [c[:] for c in classes]
                    
        elif move_type < 0.8:
            # Swap modulus
            if not classes or not unselected_moduli: continue
            idx = random.randrange(len(classes))
            m_out, r_out = classes[idx]
            m_in = random.choice(unselected_moduli)
            
            # Find best r_in for m_in
            # Note: This is expensive O(m_in).
            # We can approximate or just use random r_in?
            # Let's use best r_in for better moves.
            # But we must do it on the state where m_out is removed.
            # Since sets are disjoint, removing m_out doesn't affect m_in's gain calculation?
            # Yes, disjoint.
            
            # Calculate delta for removing m_out
            lost_out = int(np.count_nonzero(cover_count[r_out::m_out] == 1))
            
            # Calculate best gain for m_in
            # We want to maximize gain_in.
            # gain_in = count(cover_count[r::m_in] == 0)
            # Since m_out and m_in are distinct, their residue sets are not necessarily disjoint?
            # Wait. m_out and m_in are distinct moduli.
            # The set of residues covered by m_out is $R_{out}$.
            # The set of residues covered by m_in is $R_{in}$.
            # They CAN overlap.
            # So removing m_out DOES affect the count of zeros in $R_{in}$.
            # Specifically, residues in $R_{in} \cap R_{out}$ that were covered ONLY by m_out will become uncovered.
            # So we MUST calculate gain_in AFTER removing m_out.
            
            # Perform removal temporarily
            cover_count[r_out::m_out] -= 1
            
            # Find best r_in
            best_gain_in = -1
            best_r_in = 0
            for r in range(m_in):
                gain = int(np.count_nonzero(cover_count[r::m_in] == 0))
                if gain > best_gain_in:
                    best_gain_in = gain
                    best_r_in = r
            
            # Undo removal
            cover_count[r_out::m_out] += 1
            
            delta_E = lost_out - best_gain_in
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                cover_count[r_out::m_out] -= 1
                cover_count[best_r_in::m_in] += 1
                uncovered += lost_out - best_gain_in
                classes[idx][0] = m_in
                classes[idx][1] = best_r_in
                selected_moduli.remove(m_out)
                selected_moduli.add(m_in)
                unselected_moduli.remove(m_in)
                unselected_moduli.append(m_out)
                
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_classes = [c[:] for c in classes]
                    
        else:
            # Add/Remove single modulus (less frequent)
            if random.random() < 0.5 and unselected_moduli:
                m_in = random.choice(unselected_moduli)
                best_r_in, gain_in = find_best_residue(m_in)
                delta_E = -gain_in
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    add_class(m_in, best_r_in)
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
            elif classes:
                idx = random.randrange(len(classes))
                m_out, r_out = classes[idx]
                lost_out = int(np.count_nonzero(cover_count[r_out::m_out] == 1))
                delta_E = lost_out
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    remove_class(m_out, r_out)
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
        
        T *= alpha

    return best_classes

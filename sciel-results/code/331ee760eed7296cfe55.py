import time
import math

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    # Sort divisors
    divisors.sort()
    
    # 2. Precompute masks and residue lists for each (m, r)
    # masks[m][r] is an integer with bits set at r, r+m, ...
    # residues[m][r] is a list of residues for faster iteration if needed
    masks = {}
    residues = {}
    for m in divisors:
        masks[m] = [0] * m
        residues[m] = [[] for _ in range(m)]
        for r in range(m):
            mask_val = 0
            res_list = []
            for i in range(r, base, m):
                mask_val |= (1 << i)
                res_list.append(i)
            masks[m][r] = mask_val
            residues[m][r] = res_list
            
    # 3. Greedy Set Cover
    uncovered_mask = (1 << base) - 1
    remaining_moduli = set(divisors)
    classes = []
    used_moduli = set()
    
    # freq[x] = number of classes covering residue x
    freq = [0] * base
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        # Find the modulus and residue that maximizes coverage of uncovered
        for m in remaining_moduli:
            # For this m, find r that maximizes intersection with uncovered
            for r in range(m):
                # Gain is number of bits in (uncovered_mask & masks[m][r])
                gain = (uncovered_mask & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        # Select the best class
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        remaining_moduli.remove(best_m)
        
        # Update uncovered and freq
        m = best_m
        r = best_r
        for x in residues[m][r]:
            if freq[x] == 0:
                uncovered_mask &= ~(1 << x)
            freq[x] += 1
        
        # Time check
        if time.time() > deadline:
            break
            
    # 4. Local Search
    # We try to improve the solution by adding unused classes or swapping.
    # Using freq array to determine gains and losses efficiently.
    
    # Helper to calculate gain of adding (m, r)
    def calc_gain(m, r):
        count = 0
        for x in residues[m][r]:
            if freq[x] == 0:
                count += 1
        return count

    # Helper to calculate loss of removing (m, r)
    # Loss is number of x in class where freq[x] == 1
    def calc_loss(m, r):
        count = 0
        for x in residues[m][r]:
            if freq[x] == 1:
                count += 1
        return count

    # Add phase: Add any class that reduces uncovered count
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_m = None
        best_r = None
        
        # Try adding an unused modulus
        for m in divisors:
            if m in used_moduli:
                continue
            for r in range(m):
                gain = calc_gain(m, r)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain > 0:
            m = best_m
            r = best_r
            classes.append([m, r])
            used_moduli.add(m)
            for x in residues[m][r]:
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                freq[x] += 1
            improved = True

    # Swap phase: Try to swap a used class with an unused class
    # We iterate a few times to allow chain reactions
    for _ in range(10): # Limit iterations to avoid infinite loops or excessive time
        if time.time() > deadline:
            break
        best_swap_gain = 0 # Net gain = Gain(new) - Loss(old)
        best_old_idx = -1
        best_new_m = None
        best_new_r = None
        
        # Try all pairs (old, new)
        # This is O(N_used * N_unused * m) which can be large.
        # N_used <= 50, N_unused <= 50.
        # Max m is 51975.
        # 50 * 50 * 51975 is too big for Python in 120s if we do it naively many times.
        # We need to be smarter or limit the search.
        
        # Strategy: For each unused m, find best r.
        # For each used m, find "worst" (lowest loss) r? No, we want to replace specific classes.
        
        # Let's try a simpler swap:
        # 1. Identify candidate new classes (unused moduli) with high gain.
        # 2. Identify candidate old classes (used moduli) with low loss.
        # 3. Try swapping top K new with top K old.
        
        # Calculate gains for all unused
        unused_candidates = []
        for m in divisors:
            if m in used_moduli:
                continue
            best_r_for_m = 0
            best_gain_for_m = -1
            for r in range(m):
                g = calc_gain(m, r)
                if g > best_gain_for_m:
                    best_gain_for_m = g
                    best_r_for_m = r
            if best_gain_for_m > 0:
                unused_candidates.append((best_gain_for_m, m, best_r_for_m))
        
        unused_candidates.sort(reverse=True)
        
        # Calculate losses for all used
        used_candidates = []
        for idx, (m, r) in enumerate(classes):
            loss = calc_loss(m, r)
            used_candidates.append((loss, idx, m, r))
        
        used_candidates.sort() # Lowest loss first
        
        # Try top 5 unused with top 5 used
        for g_new, m_new, r_new in unused_candidates[:5]:
            for l_old, idx_old, m_old, r_old in used_candidates[:5]:
                net_gain = g_new - l_old
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_old_idx = idx_old
                    best_new_m = m_new
                    best_new_r = r_new
        
        if best_swap_gain > 0 and best_old_idx != -1:
            # Perform swap
            m_old, r_old = classes[best_old_idx]
            
            # Remove old
            used_moduli.remove(m_old)
            for x in residues[m_old][r_old]:
                freq[x] -= 1
                if freq[x] == 0:
                    uncovered_mask |= (1 << x)
            classes.pop(best_old_idx)
            
            # Add new
            used_moduli.add(best_new_m)
            for x in residues[best_new_m][best_new_r]:
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                freq[x] += 1
            classes.append([best_new_m, best_new_r])
            
    return classes

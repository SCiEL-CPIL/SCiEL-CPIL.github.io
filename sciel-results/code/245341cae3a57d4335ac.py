# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Bitset approach: use integer bit masks for O(1) lookups
    # Bit i represents whether element (i+1) is in the set
    
    def greedy_init():
        """Greedy: add smallest valid element first to maximize reciprocal sum."""
        mask = 0
        selected = []
        for x in range(1, N + 1):
            valid = True
            for a in selected:
                for b in selected:
                    if a < b:
                        if 2 * b - a == x:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                selected.append(x)
                mask |= (1 << x)
        return mask, selected
    
    def is_3ap_free(mask):
        """Check if the set represented by mask has no 3-term AP."""
        for i in range(1, N + 1):
            if mask & (1 << i):
                for j in range(i + 1, N + 1):
                    if mask & (1 << j):
                        k = 2 * j - i
                        if k > j and k <= N and (mask & (1 << k)):
                            return False
        return True
    
    def check_compatible(z, mask_without_y):
        """Check if z can be added to the set represented by mask_without_y without creating a 3-AP.
        Uses bit operations for speed."""
        # For each a in the set, check if adding z creates a 3-AP
        # 3-AP: (a, z, b) where 2z = a+b, or (a, b, z) where 2b = a+z, or (z, a, b) where 2a = z+b
        # Efficient check: for each a in set, compute the two other elements that would form AP with a and z
        for a in range(1, N + 1):
            if not (mask_without_y & (1 << a)):
                continue
            # Case 1: a, z, b => b = 2z - a
            b = 2 * z - a
            if b != a and 1 <= b <= N and (mask_without_y & (1 << b)):
                return False
            # Case 2: a, b, z => b = (a+z)//2 (only if a+z is even)
            if (a + z) % 2 == 0:
                b = (a + z) // 2
                if b != a and b != z and 1 <= b <= N and (mask_without_y & (1 << b)):
                    return False
            # Case 3: z, a, b => b = 2a - z
            b = 2 * a - z
            if b != a and b != z and 1 <= b <= N and (mask_without_y & (1 << b)):
                return False
        return True
    
    def reciprocal_sum(selected):
        """Compute exact reciprocal sum."""
        s = Fraction(0, 1)
        for x in selected:
            s += Fraction(1, x)
        return s
    
    def mask_to_list(mask):
        return [x for x in range(1, N + 1) if mask & (1 << x)]
    
    # Strategy 1: Greedy initialization
    mask, selected = greedy_init()
    best_mask = mask
    best_selected = selected[:]
    best_score = reciprocal_sum(best_selected)
    
    # Local search: try swaps to improve reciprocal sum
    # Key optimization: precompute for each z which elements conflict with it
    # But that's complex, so use the bitset check but with early termination
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to remove larger elements and add smaller ones
        for y in sorted(selected, reverse=True):
            if time.time() > deadline - 5:
                break
            mask_without_y = mask & ~(1 << y)
            # Try adding z < y not in set
            for z in range(1, y):
                if mask & (1 << z):
                    continue
                if check_compatible(z, mask_without_y):
                    # Perform swap
                    mask = mask_without_y | (1 << z)
                    selected.remove(y)
                    selected.append(z)
                    new_score = reciprocal_sum(selected)
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = mask
                        best_selected = selected[:]
                    improved = True
                    break
            if improved:
                break
    
    # Strategy 2: Simulated annealing-style search with more aggressive swaps
    if time.time() < deadline - 10:
        random.seed(42)
        sa_mask = best_mask
        sa_selected = best_selected[:]
        sa_score = best_score
        
        temp = 0.1
        for iteration in range(5000):
            if time.time() > deadline - 10:
                break
            if temp < 0.001:
                break
            
            # Randomly pick an element to remove
            y = random.choice(sa_selected)
            mask_without_y = sa_mask & ~(1 << y)
            
            # Try to add a random smaller element
            candidates = [z for z in range(1, y) if not (sa_mask & (1 << z))]
            if not candidates:
                continue
            
            z = random.choice(candidates)
            if check_compatible(z, mask_without_y):
                new_mask = mask_without_y | (1 << z)
                new_selected = sa_selected[:]
                new_selected.remove(y)
                new_selected.append(z)
                new_score = reciprocal_sum(new_selected)
                delta = new_score - sa_score
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    sa_mask = new_mask
                    sa_selected = new_selected
                    sa_score = new_score
                    if sa_score > best_score:
                        best_score = sa_score
                        best_mask = sa_mask
                        best_selected = sa_selected[:]
            
            temp *= 0.999
    
    # Strategy 3: Try different greedy orders
    if time.time() < deadline - 15:
        # Greedy with priority on small numbers (already done)
        # Try: start with 1, then greedily add
        # Try: multiple random restarts with local search
        for restart in range(20):
            if time.time() > deadline - 15:
                break
            
            # Randomized greedy
            random.seed(100 + restart)
            mask_r = 0
            selected_r = []
            order = list(range(1, N + 1))
            random.shuffle(order)
            # But we want to favor small numbers for reciprocal sum
            # Weighted shuffle: smaller numbers more likely to come first
            for x in order:
                if x > 100 and random.random() < 0.5:
                    continue  # skip some large numbers
                valid = True
                for a in selected_r:
                    for b in selected_r:
                        if a < b:
                            if 2 * b - a == x:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    selected_r.append(x)
                    mask_r |= (1 << x)
            
            if not selected_r:
                continue
            
            score_r = reciprocal_sum(selected_r)
            
            # Quick local search
            for _ in range(50):
                if time.time() > deadline - 15:
                    break
                y = max(selected_r)
                mask_without_y = mask_r & ~(1 << y)
                found = False
                for z in range(1, y):
                    if mask_r & (1 << z):
                        continue
                    if check_compatible(z, mask_without_y):
                        mask_r = mask_without_y | (1 << z)
                        selected_r.remove(y)
                        selected_r.append(z)
                        score_r = reciprocal_sum(selected_r)
                        found = True
                        break
                if not found:
                    break
            
            if score_r > best_score:
                best_score = score_r
                best_mask = mask_r
                best_selected = selected_r[:]
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END

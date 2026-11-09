import time
import random
import math

def construct():
    n = 12
    m = 40
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # Precompute popcount for all masks for quick size checks if needed, 
    # but Python 3.10+ has int.bit_count(). Assuming modern Python.
    
    def count_comparable(family):
        """Count unordered pairs where one set is a strict subset of the other."""
        count = 0
        for i in range(len(family)):
            mi = family[i]
            for j in range(i + 1, len(family)):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count

    def greedy_init():
        """
        Initialize with a maximal chain, then greedily add sets 
        that are comparable to the most existing sets.
        """
        family = []
        used = set()
        
        # 1. Maximal chain: 0, 1, 3, 7, ..., 4095
        cur = 0
        for i in range(n + 1):
            family.append(cur)
            used.add(cur)
            if i < n:
                cur |= (1 << i)
        
        # 2. Greedily add remaining m - (n+1) sets
        # We want to maximize comparable pairs.
        # A set S is comparable to T if S < T or T < S.
        
        remaining_count = m - len(family)
        
        # To speed up greedy selection, we can precompute comparability 
        # with the current family, but the family grows.
        # Since m=40 is small, O(N * M^2) is fine for init.
        
        # Candidate pool: all masks not in family
        all_masks = list(range(1 << n))
        
        for _ in range(remaining_count):
            best_mask = -1
            best_score = -1
            
            # We can restrict candidates to those that are "close" to the chain?
            # No, let's just scan all unused masks. 4096 * 40 * 40 is ~6M ops, fast enough.
            # Actually, we just need to check against current family (size k).
            
            for mask in all_masks:
                if mask in used:
                    continue
                
                # Check if this mask is comparable to many existing sets
                comp_count = 0
                for f in family:
                    if (mask & f) == mask and mask != f:
                        comp_count += 1
                    elif (mask & f) == f and mask != f:
                        comp_count += 1
                
                # Tie-breaking: prefer smaller size? Or random? 
                # Let's prefer sets that are comparable to the "middle" of the chain 
                # to maintain density. Or just random tie-break.
                if comp_count > best_score:
                    best_score = comp_count
                    best_mask = mask
                elif comp_count == best_score and best_mask != -1:
                    # Random tie-break among best candidates
                    if random.random() < 0.5:
                        best_mask = mask
            
            if best_mask != -1:
                family.append(best_mask)
                used.add(best_mask)
            else:
                # Should not happen as there are 4096 masks and we only pick 40
                break
        
        return family[:m]

    # Start with greedy initialization
    current_family = greedy_init()
    current_score = count_comparable(current_family)
    
    best_family = current_family[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute list of masks for random selection? 
    # No, random.randint is fine.
    
    while time.time() < deadline and T > T_min:
        # Choose a move
        # Strategies:
        # 1. Add element (expand)
        # 2. Remove element (contract)
        # 3. Swap element
        # 4. Random replacement (escape)
        
        idx = random.randint(0, m - 1)
        old_mask = current_family[idx]
        
        move_type = random.random()
        new_mask = None
        
        if move_type < 0.3:
            # Add element
            if old_mask != (1 << n) - 1:
                # Find available bits
                available = []
                for b in range(n):
                    if not (old_mask >> b) & 1:
                        available.append(b)
                if available:
                    b = random.choice(available)
                    new_mask = old_mask | (1 << b)
        elif move_type < 0.6:
            # Remove element
            if old_mask != 0:
                # Find present bits
                present = []
                for b in range(n):
                    if (old_mask >> b) & 1:
                        present.append(b)
                if present:
                    b = random.choice(present)
                    new_mask = old_mask & ~(1 << b)
        elif move_type < 0.9:
            # Swap element
            # Pick one to remove, one to add
            present = []
            absent = []
            for b in range(n):
                if (old_mask >> b) & 1:
                    present.append(b)
                else:
                    absent.append(b)
            if present and absent:
                b_out = random.choice(present)
                b_in = random.choice(absent)
                new_mask = (old_mask & ~(1 << b_out)) | (1 << b_in)
        else:
            # Random replacement
            new_mask = random.randint(0, (1 << n) - 1)
            
        if new_mask is None or new_mask == old_mask:
            continue
            
        # Check distinctness
        is_dup = False
        for i in range(m):
            if i != idx and current_family[i] == new_mask:
                is_dup = True
                break
        
        if is_dup:
            continue
            
        # Calculate delta
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            mj = current_family[j]
            
            # Old contribution
            if (old_mask & mj) == old_mask and old_mask != mj:
                delta -= 1
            elif (old_mask & mj) == mj and old_mask != mj:
                delta -= 1
                
            # New contribution
            if (new_mask & mj) == new_mask and new_mask != mj:
                delta += 1
            elif (new_mask & mj) == mj and new_mask != mj:
                delta += 1
                
        new_score = current_score + delta
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0:
            accept = True
        else:
            # If delta is 0, accept with prob 1? Or small? 
            # Usually delta=0 is neutral. Let's accept to explore.
            if delta == 0:
                accept = random.random() < 0.5
            else:
                # Avoid overflow/underflow
                if delta < -100 * T:
                    accept = False
                else:
                    accept = random.random() < math.exp(delta / T)
                    
        if accept:
            current_family[idx] = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert best family to list of lists
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

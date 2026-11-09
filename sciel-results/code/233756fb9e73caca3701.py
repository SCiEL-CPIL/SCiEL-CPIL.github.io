"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def _greedy_init(n, m):
    """Initialize family using a greedy approach with bitmasks."""
    family = []
    used = set()
    
    # Precompute all subsets as bitmasks? No, just iterate 0 to 2^n - 1
    max_mask = 1 << n
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        # Iterate through all possible sets
        # To speed up, we can randomize the order or just scan
        # Scanning 4096 items is fast.
        for s in range(max_mask):
            if s in used:
                continue
            
            gain = 0
            # Check comparability with existing family
            for t in family:
                # Check if s is subset of t
                if (s & t) == s:
                    gain += 1
                # Check if t is subset of s
                elif (t & s) == t:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s
            elif gain == best_gain and best_set != -1:
                # Tie-breaker: prefer smaller size? Or random?
                # Let's stick to first found or randomize scan order?
                # Randomizing scan order makes it non-deterministic but might escape symmetry.
                # For now, let's just take the first max.
                pass

        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            # Should not happen for m=40, n=12 (4096 subsets available)
            break
            
    return family

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i+1, m):
            sj = family[j]
            if (si & sj) == si:
                count += 1
            elif (si & sj) == sj:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Initial family using Greedy
    current_family = _greedy_init(n, m)
    current_fs = set(current_family)
    current_score = _calc_score_bitmask(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        idx = random.randint(0, m - 1)
        old_set = current_family[idx]
        
        new_set = None
        strategy = random.random()
        
        if strategy < 0.4:
            # Add element
            # Find a bit not set
            # We can try random bits
            for _ in range(n):
                b = random.randint(0, n - 1)
                if not (old_set & (1 << b)):
                    new_set = old_set | (1 << b)
                    break
        elif strategy < 0.8:
            # Remove element
            # Find a bit set
            # Get list of set bits? Or just try random
            for _ in range(n):
                b = random.randint(0, n - 1)
                if (old_set & (1 << b)):
                    new_set = old_set & ~(1 << b)
                    break
        else:
            # Swap element
            # Find a set bit and an unset bit
            set_bits = []
            unset_bits = []
            for b in range(n):
                if (old_set & (1 << b)):
                    set_bits.append(b)
                else:
                    unset_bits.append(b)
            
            if set_bits and unset_bits:
                b_out = random.choice(set_bits)
                b_in = random.choice(unset_bits)
                new_set = (old_set & ~(1 << b_out)) | (1 << b_in)
        
        if new_set is None:
            continue
            
        if new_set in current_fs:
            continue
            
        # Calculate delta score
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            other = current_family[j]
            
            # Remove old contributions
            if (old_set & other) == old_set:
                delta -= 1
            elif (old_set & other) == other:
                delta -= 1
                
            # Add new contributions
            if (new_set & other) == new_set:
                delta += 1
            elif (new_set & other) == other:
                delta += 1
                
        new_score = current_score + delta
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Update
            current_fs.remove(old_set)
            current_fs.add(new_set)
            current_family[idx] = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(current_family)
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
        
    return result
# EVOLVE-BLOCK-END

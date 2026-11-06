# EVOLVE-BLOCK-START
import math
import random
from math import gcd

def construct():
    n = 512
    # Precompute weights
    weights = [0.0] * n
    for a in range(1, n):
        weights[a] = 1.0 / (n - a)
        
    # Precompute conflicts as bitsets
    # conflicts[a] is an int where bit b is 1 if gcd(a,b) > 1
    conflicts = [0] * n
    for a in range(1, n):
        mask = 0
        for b in range(1, n):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflicts[a] = mask
        
    # Helper to get list of indices from bitset
    def bits_to_list(mask):
        res = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res

    # Helper to get sum of weights for bits in mask
    def mask_weight(mask):
        total = 0.0
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            total += weights[idx]
            mask ^= lsb
        return total

    # Initialize with Greedy: Add 1, then primes, then others if possible
    # Actually, let's start with a randomized greedy to have a decent start
    # Or just start with all primes + 1
    selected_mask = 0
    current_score = 0.0
    
    # Start with 1
    selected_mask |= (1 << 1)
    current_score += weights[1]
    
    # Add primes
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0: return False
        return True

    for p in range(2, n):
        if is_prime(p):
            # Check if p conflicts with current selection
            if (conflicts[p] & selected_mask) == 0:
                selected_mask |= (1 << p)
                current_score += weights[p]

    # Greedy fill
    # Iterate in order of weight (highest first)
    candidates = list(range(1, n))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    for a in candidates:
        if (selected_mask >> a) & 1:
            continue
        # Check conflicts
        if (conflicts[a] & selected_mask) == 0:
            selected_mask |= (1 << a)
            current_score += weights[a]

    # Simulated Annealing
    # T_start = 1.0, T_end = 1e-5
    # Steps = 50000
    T = 0.5
    cooling = 0.9999
    min_T = 1e-5
    steps = 50000
    
    # Precompute list of all nodes 1..511
    all_nodes = list(range(1, n))
    
    for step in range(steps):
        if T < min_T:
            break
            
        # Choose a random move
        # 50% add, 50% remove
        if random.random() < 0.5:
            # Add random node
            a = random.choice(all_nodes)
            if (selected_mask >> a) & 1:
                continue
            # Calculate change
            # Remove conflicts
            removed_mask = conflicts[a] & selected_mask
            cost = mask_weight(removed_mask)
            gain = weights[a] - cost
            new_score = current_score + gain
            
            # Accept?
            if gain > 0 or random.random() < math.exp(gain / T):
                # Apply
                selected_mask = (selected_mask & ~removed_mask) | (1 << a)
                current_score = new_score
        else:
            # Remove random node
            # Get list of selected
            # To be fast, we can pick a random bit from selected_mask
            # But random.choice on a list is O(N). 
            # Let's just pick a random node and if it's selected, remove it.
            # If not selected, skip.
            a = random.choice(all_nodes)
            if not ((selected_mask >> a) & 1):
                continue
            
            # Remove a
            gain = -weights[a]
            new_score = current_score + gain
            
            # Accept?
            if gain > 0 or random.random() < math.exp(gain / T):
                selected_mask &= ~(1 << a)
                current_score = new_score
                
        T *= cooling

    # Final Greedy Polish
    # Try to add any node that fits
    for a in range(1, n):
        if (selected_mask >> a) & 1:
            continue
        if (conflicts[a] & selected_mask) == 0:
            selected_mask |= (1 << a)
            current_score += weights[a]
            
    # Try swaps again
    improved = True
    while improved:
        improved = False
        for a in range(1, n):
            if (selected_mask >> a) & 1:
                continue
            removed_mask = conflicts[a] & selected_mask
            if not removed_mask:
                continue
            cost = mask_weight(removed_mask)
            gain = weights[a] - cost
            if gain > 1e-9:
                selected_mask = (selected_mask & ~removed_mask) | (1 << a)
                current_score += gain
                improved = True
                break

    return bits_to_list(selected_mask)
# EVOLVE-BLOCK-END

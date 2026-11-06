import time
import sys
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility relationships
    # div_mask[i] is a bitmask where bit j is set if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div_mask[i] |= (1 << j)
    
    # For antichain check: a set of indices is an antichain if no one divides another
    # We can check: for each i in the set, div_mask[i] & set_mask == 0 (no other element in set is divisible by smooth[i])
    # But we need to be careful: if smooth[i] divides smooth[j], then j is in div_mask[i]
    # So for a set S (as bitmask), S is an antichain iff for all i in S, div_mask[i] & S == 0
    # Actually, we need: for all i,j in S with i != j, smooth[i] does not divide smooth[j] and smooth[j] does not divide smooth[i]
    # This is equivalent to: for all i in S, (div_mask[i] & S) == 0
    # Because if smooth[i] divides smooth[j] and both are in S, then bit j is in div_mask[i] and in S, so div_mask[i] & S != 0
    
    # Precompute antichain property for small subsets? Too many subsets.
    # Instead, we'll check on the fly with a helper.
    
    def is_antichain(indices):
        """Check if the given indices form an antichain."""
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        for idx in indices:
            if div_mask[idx] & mask:
                return False
        return True
    
    # Precompute which smooth numbers can be used for each target
    # smooth_idx_by_value: dict from value to index in smooth list
    smooth_idx = {v: i for i, v in enumerate(smooth)}
    
    # For each target n, find the minimum number of terms
    answers = []
    
    # We'll search for each target from 1 to 128
    # For each target, try counts 1, 2, 3, ... until we find a valid antichain sum
    # To be efficient, we can precompute all valid antichain subsets up to some size
    
    # Let's precompute all antichain subsets of size 1, 2, 3, 4, 5
    # and store them by their sum
    
    # Actually, let's just do a direct search for each target, trying small counts first
    
    # Precompute: for each count k, list all antichain subsets of size k with their sum
    # This might be too expensive. Let's do it on the fly with pruning.
    
    # Alternative: for each target n, generate candidate subsets of smooth numbers <= n
    # and check if they sum to n and form an antichain
    
    # Let's precompute antichain subsets up to size 5 or so, indexed by sum
    # This is feasible since there aren't too many smooth numbers
    
    # Number of smooth numbers up to 128: let's count
    # 1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,108,120,125,128
    # That's about 38 numbers
    
    # C(38,5) is about 501,890 which is manageable
    
    # Let's precompute all antichain subsets up to size 5
    antichains_by_sum = {}  # sum -> list of (indices_tuple, count)
    
    for k in range(1, 6):
        for comb in combinations(range(n_smooth), k):
            if is_antichain(comb):
                s = sum(smooth[i] for i in comb)
                if s <= 128:
                    if s not in antichains_by_sum:
                        antichains_by_sum[s] = []
                    antichains_by_sum[s].append(comb)
    
    # Now for each target, find the antichain with minimum count
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: just return what we have, fill remaining with None (invalid)
            # This shouldn't happen, but safety first
            while len(answers) < 128:
                answers.append(None)
            return answers
        
        answer = None
        for k in range(1, 6):
            if n in antichains_by_sum:
                for comb in antichains_by_sum[n]:
                    if len(comb) == k:
                        answer = [smooth[i] for i in comb]
                        break
            if answer is not None:
                break
        
        if answer is None:
            # Try larger counts up to 10
            for k in range(6, 11):
                # This is more expensive, do it on the fly
                found = False
                for comb in combinations(range(n_smooth), k):
                    if time.time() > deadline:
                        break
                    if sum(smooth[i] for i in comb) == n and is_antichain(comb):
                        answer = [smooth[i] for i in comb]
                        found = True
                        break
                if found:
                    break
                if time.time() > deadline:
                    break
        
        answers.append(answer)
    
    return answers

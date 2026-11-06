import time
import random
from collections import defaultdict

def construct():
    # Precompute all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    # Precompute submasks of size 0, 1, 2, 3 for each set
    submasks_cache = {}
    for mask in all_sets:
        subs = []
        sub = mask
        while sub:
            if bin(sub).count('1') < 4:
                subs.append(sub)
            sub = (sub - 1) & mask
        subs.append(0)
        submasks_cache[mask] = subs
    
    def greedy_run(random_seed):
        random.seed(random_seed)
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        F = []
        F_set = set()
        Pairs = defaultdict(list)
        
        for C in candidates:
            valid = True
            for K in submasks_cache[C]:
                if K in Pairs:
                    for A, B in Pairs[K]:
                        if (C & (A ^ K)) == 0 and (C & (B ^ K)) == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                F.append(C)
                F_set.add(C)
                for A in F[:-1]:
                    K = A & C
                    Pairs[K].append((A, C))
        
        return F, F_set, Pairs

    def try_add_more(F, F_set, Pairs, candidates_to_try):
        """Try to add more sets from candidates_to_try to the existing family."""
        for C in candidates_to_try:
            if C in F_set:
                continue
            valid = True
            for K in submasks_cache[C]:
                if K in Pairs:
                    for A, B in Pairs[K]:
                        if (C & (A ^ K)) == 0 and (C & (B ^ K)) == 0:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                F.append(C)
                F_set.add(C)
                for A in F[:-1]:
                    K = A & C
                    Pairs[K].append((A, C))

    deadline = time.time() + 50.0
    best_family = []
    
    # Run multiple random greedy instances
    while time.time() < deadline - 5.0:
        seed = random.randint(0, 10**9)
        F, F_set, Pairs = greedy_run(seed)
        
        # Try to add more sets from all_sets that weren't successfully added in greedy
        if time.time() < deadline - 5.0:
            remaining = [c for c in all_sets if c not in F_set]
            if remaining:
                random.shuffle(remaining)
                try_add_more(F, F_set, Pairs, remaining)
        
        if len(F) > len(best_family):
            best_family = F
            if len(best_family) > 1500:
                break
    
    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result

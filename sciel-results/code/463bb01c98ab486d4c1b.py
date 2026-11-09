"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total = len(all_sets)  # 1820
    
    def is_compatible(family, C):
        """
        Check if C can be added to family without forming a sunflower.
        Uses bucketing by intersection with C for efficiency.
        """
        n_fam = len(family)
        if n_fam < 2:
            return True
        
        # Bucket family members by their intersection with C
        buckets = {}
        for A in family:
            I = A & C
            if I not in buckets:
                buckets[I] = []
            buckets[I].append(A)
        
        # For each bucket (same intersection I with C), check if any pair (A, B) in the bucket
        # has A & B == I. If so, A, B, C form a sunflower.
        for I, bl in buckets.items():
            if len(bl) < 2:
                continue
            # Check pairs in this bucket
            for ia in range(len(bl)):
                A = bl[ia]
                for ib in range(ia + 1, len(bl)):
                    B = bl[ib]
                    if (A & B) == I:
                        return False
        return True
    
    def greedy_solve(order):
        family = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            C = all_sets[idx]
            if is_compatible(family, C):
                family.append(C)
        return family
    
    best_family = []
    best_size = 0
    
    # 1. Lexicographic Order (Strong baseline)
    order_lex = list(range(total))
    fam = greedy_solve(order_lex)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam[:]
    
    # 2. Reversed Lexicographic Order
    order_rev = list(range(total))
    order_rev.reverse()
    fam = greedy_solve(order_rev)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam[:]
    
    # 3. Random Orders with many restarts
    max_restarts = 500
    rng = random.Random(42)
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        order = list(range(total))
        rng.shuffle(order)
        
        fam = greedy_solve(order)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam[:]
            if best_size >= 250:
                break
    
    # 4. Local Search / Kick-starts on the best family
    if time.time() < deadline - 5.0 and best_size > 0:
        for ks_round in range(200):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random subset of 2 elements to remove
            if len(best_family) < 2:
                break
                
            remove_indices = rng.sample(range(len(best_family)), 2)
            remove_set = set(remove_indices)
            
            # Create new family without these elements
            new_family = [x for i, x in enumerate(best_family) if i not in remove_set]
            new_family_set = set(new_family)
            
            # Candidates: all masks not in new_family
            candidates = [m for m in all_sets if m not in new_family_set]
            rng.shuffle(candidates)
            
            # Re-greedy
            improved_family = new_family[:]
            for C in candidates:
                if time.time() > deadline - 2.0:
                    break
                if is_compatible(improved_family, C):
                    improved_family.append(C)
            
            if len(improved_family) > best_size:
                best_size = len(improved_family)
                best_family = improved_family
                if best_size >= 250:
                    break
    
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
    
    return result
# EVOLVE-BLOCK-END

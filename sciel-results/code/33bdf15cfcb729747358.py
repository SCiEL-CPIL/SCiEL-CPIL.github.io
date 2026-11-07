import time
import random
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 12
    m = 40
    N = 1 << n
    
    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        len_fam = len(family)
        for i in range(len_fam):
            ai = family[i]
            for j in range(i + 1, len_fam):
                aj = family[j]
                if is_subset(ai, aj) or is_subset(aj, ai):
                    count += 1
        return count

    def greedy_start():
        """Greedy construction: add set that maximizes comparability with current family."""
        family = []
        used = set()
        for _ in range(m):
            best_mask = -1
            best_gain = -1
            # Iterate through all masks
            for mask in range(N):
                if mask in used:
                    continue
                gain = 0
                for f in family:
                    if (mask & f) == mask or (f & mask) == f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                # Fallback to random
                while True:
                    mask = random.randint(0, N - 1)
                    if mask not in used:
                        best_mask = mask
                        break
            family.append(best_mask)
            used.add(best_mask)
        return family

    def random_start():
        """Random distinct sets."""
        fam = []
        used = set()
        while len(fam) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                used.add(mask)
                fam.append(mask)
        return fam
    
    def local_search(initial_family, deadline):
        family = initial_family[:]
        family_set = set(family)
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        temp = 10.0
        no_improve_count = 0
        
        iter_count = 0
        while time.time() < deadline:
            iter_count += 1
            
            if iter_count % 1000 == 0 and temp > 0.1:
                temp *= 0.995
                if no_improve_count > 5000:
                    # Restart with perturbation
                    for _ in range(5):
                        idx = random.randint(0, m - 1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in family_set:
                            family[idx] = new_mask
                            family_set.remove(initial_family[idx] if initial_family[idx] != family[idx] else family[idx])
                            # This is getting complex, let's just do simple restart
                            break
                    no_improve_count = 0
                    temp = 5.0
                    continue
            
            idx = random.randint(0, m - 1)
            old_mask = family[idx]
            
            # Generate candidate: bias toward neighbors and comparable sets
            if random.random() < 0.6:
                # Try a random mask
                new_mask = random.randint(0, N - 1)
            elif random.random() < 0.5:
                # Flip 1 bit
                new_mask = old_mask ^ (1 << random.randint(0, n - 1))
            else:
                # Flip 2 bits
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            
            if new_mask == old_mask:
                continue
            if new_mask in family_set:
                continue
            
            # Calculate delta
            lost = 0
            gained = 0
            for j in range(m):
                if j == idx:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1
            
            delta = gained - lost
            
            accept = False
            if delta > 0:
                accept = True
            elif temp > 0.1:
                # Simulated annealing acceptance
                if random.random() < math.exp(delta / temp):
                    accept = True
            
            if accept:
                family[idx] = new_mask
                family_set.remove(old_mask)
                family_set.add(new_mask)
                current_score += delta
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                no_improve_count += 1
        
        return best_family, best_score
    
    best_overall_family = None
    best_overall_score = 0
    
    # Run multiple restarts
    # Start with greedy to get a good baseline
    fam = greedy_start()
    fam, score = local_search(fam, deadline)
    
    if score > best_overall_score:
        best_overall_score = score
        best_overall_family = fam

    # Try random restarts if time permits
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        fam = random_start()
        fam, score = local_search(fam, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam
    
    if best_overall_family is None:
        best_overall_family = list(range(m))
    
    result = []
    for mask in best_overall_family:
        # Ensure types are correct: list of ints
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result

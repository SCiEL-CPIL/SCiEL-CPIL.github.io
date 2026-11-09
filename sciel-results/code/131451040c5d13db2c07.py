import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def count_sum_collisions(S):
    """Count how many pair sums are duplicated (higher = more collisions = harder for Sidon)"""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum of (count-1) for each sum that appears more than once
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def greedy_construct_collision_maximizing():
    """Greedy construction that maximizes sum collisions"""
    selected = []
    used = set()
    
    # Start with a random seed
    seed_val = random.randint(0, 128)
    selected.append(seed_val)
    used.add(seed_val)
    
    for _ in range(11):
        best_candidate = None
        best_collisions = -1
        
        for candidate in range(0, 129):
            if candidate in used:
                continue
            collisions = count_sum_collisions(selected + [candidate])
            if collisions > best_collisions:
                best_collisions = collisions
                best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            used.add(best_candidate)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_set = None
    best_val = 12
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7 + 13)
        candidate = greedy_construct_collision_maximizing()
        val = largest_sidon_subset_size(candidate)
        
        if val < best_val:
            best_val = val
            best_set = candidate[:]
    
    # Phase 2: Random sampling from small range [0, 30]
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        candidate = random.sample(range(0, 30), 12)
        val = largest_sidon_subset_size(candidate)
        
        if val < best_val:
            best_val = val
            best_set = candidate[:]
    
    # Phase 3: Local search with single-element replacements
    if best_set is not None:
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            
            for i in range(12):
                if time.time() > deadline:
                    break
                
                old_val = best_set[i]
                
                # Try replacing with values that might create more collisions
                # Focus on values close to existing ones to create sum overlaps
                for new_val in range(0, 30):
                    if new_val == old_val:
                        continue
                    if new_val in best_set:
                        continue
                    
                    new_set = best_set[:]
                    new_set[i] = new_val
                    
                    val = largest_sidon_subset_size(new_set)
                    
                    if val < best_val:
                        best_val = val
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Random perturbation
                if random.random() < 0.3:
                    idx = random.randint(0, 11)
                    old = best_set[idx]
                    best_set[idx] = random.randint(0, 25)
                    if len(set(best_set)) < 12:
                        best_set[idx] = old
    
    if best_set is None:
        # Fallback: return a reasonable set
        return list(range(12))
    
    return best_set

import time
import math
import random
import numpy as np

def _score_set_bitset(U):
    """
    Compute the score for a finite set U containing 0 using bitsets for efficiency.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if not U:
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Bitset representation
    # Bit i is set if i is in the set.
    # We need bits up to 2*max_u for sums, and -max_u to max_u for diffs.
    # To handle negative differences, we shift by max_u.
    # Diff range: -max_u to max_u. Size: 2*max_u + 1.
    # Sum range: 0 to 2*max_u. Size: 2*max_u + 1.
    
    # Create bitsets
    sum_mask = 0
    diff_mask = 0
    
    for a in U:
        for b in U:
            sum_mask |= (1 << (a + b))
            diff_mask |= (1 << (a - b + max_u))
    
    sumset_size = sum_mask.bit_count()
    diffset_size = diff_mask.bit_count()
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline, max_size=12, search_max_cap=500):
    """
    Perform local search starting from set U.
    Moves: 
    1. Change value of an existing element.
    2. Add a new element.
    3. Remove an element (if size > 1).
    """
    U = sorted(set(U))
    best_score = _score_set_bitset(U)
    best_U = U[:]
    
    # Determine search bounds for new values
    max_u = max(best_U) if best_U else 1
    search_max = max(100, 2 * max_u + 10)
    
    improved = True
    stall_count = 0
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_new_score = best_score
        
        n = len(best_U)
        
        # 1. Try changing existing elements
        for i in range(n):
            if best_U[i] == 0:
                continue # Keep 0 fixed
            old_val = best_U[i]
            candidates = set()
            
            # Local window
            low = max(1, old_val - 5)
            high = min(search_max, old_val + 5)
            candidates.update(range(low, high + 1))
            
            # Random global samples
            for _ in range(5):
                candidates.add(random.randint(1, search_max))
            
            # Structured candidates
            candidates.update([1, 2, 3, 4, 5, 10, 20, 50, 100, 200])
            
            for new_val in candidates:
                if new_val == old_val or new_val == 0:
                    continue
                if new_val in best_U:
                    continue
                
                new_U = best_U[:]
                new_U[i] = new_val
                new_U.sort()
                
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('change', i, new_val)
        
        # 2. Try adding a new element
        if len(best_U) < max_size:
            candidates = set()
            for _ in range(10):
                candidates.add(random.randint(1, search_max))
            candidates.update([1, 2, 3, 4, 5, 10, 20, 50, 100, 200])
            if best_U:
                candidates.add(max(best_U) + 1)
                candidates.add(max(best_U) + 2)
                candidates.add(max(best_U) + 5)
            
            for new_val in candidates:
                if new_val in best_U or new_val == 0:
                    continue
                new_U = best_U + [new_val]
                new_U.sort()
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('add', new_val)
        
        # 3. Try removing an element
        if len(best_U) > 2:
            for i in range(len(best_U)):
                if best_U[i] == 0:
                    continue
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('remove', i)
        
        if best_move:
            improved = True
            stall_count = 0
            if best_move[0] == 'change':
                _, i, new_val = best_move
                best_U[i] = new_val
                best_U.sort()
            elif best_move[0] == 'add':
                _, new_val = best_move
                best_U.append(new_val)
                best_U.sort()
            elif best_move[0] == 'remove':
                _, i = best_move
                best_U.pop(i)
            
            best_score = best_new_score
            max_u = max(best_U) if best_U else 1
            search_max = max(search_max, 2 * max_u + 10)
            search_max = min(search_max, search_max_cap)
        else:
            # No improvement found, try a perturbation to escape local optimum
            stall_count += 1
            if stall_count > 3:
                break
            
            # Randomly modify the set
            if random.random() < 0.5:
                # Add a random element
                if len(best_U) < max_size:
                    new_val = random.randint(1, search_max)
                    if new_val not in best_U and new_val != 0:
                        best_U.append(new_val)
                        best_U.sort()
                        best_score = _score_set_bitset(best_U)
                else:
                    # Change a random element
                    i = random.randint(1, len(best_U)-1) 
                    new_val = random.randint(1, search_max)
                    if new_val not in best_U:
                        best_U[i] = new_val
                        best_U.sort()
                        best_score = _score_set_bitset(best_U)
            else:
                # Remove a random element
                if len(best_U) > 2:
                    i = random.randint(1, len(best_U)-1)
                    best_U.pop(i)
                    best_score = _score_set_bitset(best_U)
                
            # If perturbation didn't help or made it worse, we might get stuck.
            # For simplicity, we just break and rely on restarts.
            
    return best_U, best_score

def _generate_heuristic_seeds():
    """Generate a variety of initial sets for local search."""
    seeds = []
    
    # Small sets
    seeds.append([0, 1, 2])
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 4])
    seeds.append([0, 2, 3])
    seeds.append([0, 1, 2, 4])
    seeds.append([0, 1, 2, 3, 4])
    
    # Geometric progressions
    for base in [2, 3, 4]:
        U = [0]
        val = 1
        while val < 200:
            U.append(val)
            val *= base
        seeds.append(U)
        
    # Random sets
    for _ in range(30):
        size = random.randint(3, 12)
        max_val = random.randint(50, 500)
        U = [0] + random.sample(range(1, max_val + 1), size - 1)
        U.sort()
        seeds.append(U)
        
    # Sets with gaps
    for _ in range(20):
        U = [0]
        val = 1
        for _ in range(random.randint(2, 8)):
            val += random.randint(1, 20)
            U.append(val)
        U.sort()
        seeds.append(U)
        
    # Golomb-like rulers (distinct differences)
    # Try to construct sets with many distinct differences
    for _ in range(10):
        U = [0]
        current = 0
        for _ in range(random.randint(3, 8)):
            # Add a gap that is likely to create new differences
            gap = random.randint(2, 10)
            current += gap
            U.append(current)
        seeds.append(U)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_U = [0, 1, 4]
    best_score = _score_set_bitset(best_U)
    
    seeds = _generate_heuristic_seeds()
    
    # Randomly shuffle seeds to ensure diversity in case time runs out
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Run local search from this seed
        U_sol, s_sol = _local_search(seed, deadline)
        
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol[:]
            
    # Final polish: Run a few more local searches from the best found so far
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb best_U
            U_perturbed = best_U[:]
            # Randomly add/remove/change
            op = random.choice(['add', 'remove', 'change'])
            if op == 'add' and len(U_perturbed) < 12:
                U_perturbed.append(random.randint(1, 2 * max(U_perturbed) + 10))
            elif op == 'remove' and len(U_perturbed) > 2:
                idx = random.randint(1, len(U_perturbed)-1)
                U_perturbed.pop(idx)
            elif op == 'change' and len(U_perturbed) > 1:
                idx = random.randint(1, len(U_perturbed)-1)
                U_perturbed[idx] = random.randint(1, 2 * max(U_perturbed) + 10)
            
            U_perturbed.sort()
            U_sol, s_sol = _local_search(U_perturbed, deadline)
            if s_sol > best_score:
                best_score = s_sol
                best_U = U_sol[:]
                
    return sorted(set(best_U))

# Test
if __name__ == "__main__":
    start = time.time()
    U = construct()
    end = time.time()
    score = _score_set_bitset(U)
    print(f"Time: {end-start:.2f}s")
    print(f"Set: {U}")
    print(f"Score: {score}")

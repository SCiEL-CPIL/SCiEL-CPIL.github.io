import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def score_set(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        # Compute sumset
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        P = len(sum_set)
        # Compute difference set
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        D = len(diff_set)
        if P == 0:
            return 0.0
        ratio = D / P
        if ratio <= 1:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        return 1 + math.log(ratio) / denom
    
    def local_search(U, max_iters=5000):
        U = list(set(U))
        U.sort()
        if 0 not in U:
            U.append(0)
            U.sort()
        current_score = score_set(U)
        best_score = current_score
        best_U = list(U)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Random move: add, remove, or replace an element
            move_type = random.random()
            new_U = list(U)
            
            if move_type < 0.3 and len(U) < 50:
                # Add a new element
                max_val = max(U) if U else 0
                new_val = random.randint(1, max_val + 20)
                if new_val not in new_U:
                    new_U.append(new_val)
            elif move_type < 0.6 and len(U) > 2:
                # Remove an element (keep 0)
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_U.remove(idx)
            else:
                # Replace an element
                if len(U) > 1:
                    removable = [x for x in new_U if x != 0]
                    if removable:
                        old_val = random.choice(removable)
                        max_val = max(U) if U else 0
                        new_val = random.randint(1, max_val + 10)
                        new_U.remove(old_val)
                        if new_val not in new_U:
                            new_U.append(new_val)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = score_set(new_U)
            
            if new_score > current_score:
                current_score = new_score
                U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(U)
            else:
                # Simulated annealing: accept with probability
                if current_score > 0:
                    delta = new_score - current_score
                    if delta < 0:
                        accept_prob = math.exp(delta / max(0.01, current_score * 0.1))
                        if random.random() < accept_prob:
                            current_score = new_score
                            U = new_U
        
        return best_U, best_score
    
    # Initial candidates
    candidates = []
    
    # Simple arithmetic-like structures
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255, 511])
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    candidates.append([0, 1, 3, 9, 27, 81, 243])
    
    # Greedy construction
    def greedy_construct(max_size=30):
        U = [0]
        for _ in range(max_size - 1):
            best_val = None
            best_sc = -1
            max_val = max(U)
            for candidate in range(1, max_val + 30):
                if candidate in U:
                    continue
                test_U = U + [candidate]
                sc = score_set(test_U)
                if sc > best_sc:
                    best_sc = sc
                    best_val = candidate
            if best_val is not None:
                U.append(best_val)
            else:
                break
        return U
    
    candidates.append(greedy_construct(20))
    
    # Random initial sets
    for _ in range(5):
        size = random.randint(5, 25)
        max_val = random.randint(50, 500)
        U = [0] + random.sample(range(1, max_val + 1), size - 1)
        U.sort()
        candidates.append(U)
    
    best_U = None
    best_score = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        refined_U, refined_score = local_search(cand, max_iters=3000)
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
    
    # Additional refinement on best
    if best_U is not None:
        final_U, final_score = local_search(best_U, max_iters=10000)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    if best_U is None:
        best_U = [0, 1, 3]
    
    return sorted(set(best_U))

# EVOLVE-BLOCK-END

import time
import math
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def score_set(U):
        """Compute the score for a set U."""
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        # Use numpy for faster sumset/difference set computation
        U_arr = np.array(U)
        
        # Sumset: all pairs a+b
        sum_set = set()
        for i in range(n):
            for j in range(n):
                sum_set.add(U_arr[i] + U_arr[j])
        P = len(sum_set)
        
        # Difference set: all pairs a-b
        diff_set = set()
        for i in range(n):
            for j in range(n):
                diff_set.add(U_arr[i] - U_arr[j])
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

    def score_set_fast(U):
        """Faster scoring using sorted arrays and two-pointer or set operations."""
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        # Sumset
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        P = len(sum_set)
        
        # Difference set
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

    def local_search(U, max_iters=5000, initial_temp=1.0):
        """Simulated annealing with adaptive temperature."""
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U.sort()
        
        current_score = score_set(U)
        best_score = current_score
        best_U = list(U)
        
        temp = initial_temp
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Adaptive temperature: cool over time
            temp = max(0.01, initial_temp * (cooling ** it))
            
            # Generate a perturbation
            new_U = list(U)
            move_type = random.random()
            
            if move_type < 0.25 and len(U) < 60:
                # Add a new element
                max_val = max(U) if U else 0
                # Try values near existing elements or slightly beyond
                if random.random() < 0.5:
                    base = random.choice(U)
                    new_val = base + random.randint(1, max(5, max_val // 10))
                else:
                    new_val = random.randint(1, max_val + max(10, max_val // 5))
                if new_val not in new_U:
                    new_U.append(new_val)
            elif move_type < 0.5 and len(U) > 3:
                # Remove an element (keep 0)
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_U.remove(idx)
            elif move_type < 0.75 and len(U) > 1:
                # Replace an element
                removable = [x for x in new_U if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    max_val = max(U) if U else 0
                    if random.random() < 0.5:
                        new_val = old_val + random.randint(-max(5, old_val//5), max(5, old_val//5))
                    else:
                        new_val = random.randint(1, max_val + max(10, max_val // 5))
                    new_val = max(1, new_val)
                    new_U.remove(old_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
            else:
                # Swap two elements
                if len(U) >= 3:
                    removable = [x for x in new_U if x != 0]
                    if len(removable) >= 2:
                        idx1, idx2 = random.sample(removable, 2)
                        max_val = max(U) if U else 0
                        new_val1 = max(1, idx1 + random.randint(-max(5, idx1//5), max(5, idx1//5)))
                        new_val2 = max(1, idx2 + random.randint(-max(5, idx2//5), max(5, idx2//5)))
                        new_U.remove(idx1)
                        new_U.remove(idx2)
                        if new_val1 not in new_U:
                            new_U.append(new_val1)
                        if new_val2 not in new_U:
                            new_U.append(new_val2)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = score_set(new_U)
            
            if new_score >= current_score:
                current_score = new_score
                U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(U)
            else:
                # Accept with probability exp(delta / temp)
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = math.exp(delta / temp)
                    if random.random() < accept_prob:
                        current_score = new_score
                        U = new_U
        
        return best_U, best_score

    def final_polish(U, max_iters=20000):
        """High-precision polish with very low temperature."""
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U.sort()
        
        current_score = score_set(U)
        best_score = current_score
        best_U = list(U)
        
        temp = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline - 5:  # Leave 5s buffer
                break
            
            new_U = list(U)
            move_type = random.random()
            
            if move_type < 0.4 and len(U) > 2:
                # Fine-tune an element
                removable = [x for x in new_U if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    step = max(1, old_val // 20)
                    new_val = max(1, old_val + random.randint(-step, step))
                    new_U.remove(old_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
            elif move_type < 0.7 and len(U) < 50:
                # Add element near existing ones
                max_val = max(U) if U else 0
                if U:
                    base = random.choice(U)
                    new_val = base + random.randint(1, max(3, max_val // 20))
                    if new_val not in new_U:
                        new_U.append(new_val)
            else:
                # Remove element
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_U.remove(idx)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = score_set(new_U)
            
            if new_score >= current_score:
                current_score = new_score
                U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(U)
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = math.exp(delta / temp)
                    if random.random() < accept_prob:
                        current_score = new_score
                        U = new_U
        
        return best_U, best_score

    # Generate diverse initial candidates
    candidates = []
    
    # Known good structures
    # Geometric progression variants
    for r in [2, 3, 4, 5]:
        U = [0]
        val = 1
        while val < 10000:
            U.append(val)
            val *= r
        candidates.append(U[:min(len(U), 20)])
    
    # Mixed structures
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047])
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    candidates.append([0, 1, 3, 9, 27, 81, 243, 729, 2187])
    candidates.append([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    candidates.append([0, 1, 5, 11, 19, 29, 41, 55, 71, 89])
    
    # Arithmetic progression
    for diff in [2, 3, 5]:
        U = [0]
        val = diff
        while val < 200:
            U.append(val)
            val += diff
        candidates.append(U[:min(len(U), 25)])
    
    # Random structured sets
    random.seed(42)
    for _ in range(8):
        size = random.randint(8, 30)
        # Use a base geometric progression with noise
        ratio = random.uniform(1.5, 3.0)
        U = [0]
        val = 1
        for i in range(size - 1):
            noise = random.randint(-int(val * 0.3), int(val * 0.3))
            val = int(val * ratio + noise)
            val = max(val, U[-1] + 1 if len(U) > 0 else 1)
            U.append(val)
        U = sorted(set(U))
        if 0 in U:
            candidates.append(U)
    
    # Greedy construction
    def greedy_construct(max_size=25):
        U = [0]
        for _ in range(max_size - 1):
            best_val = None
            best_sc = -1
            max_val = max(U)
            # Try candidates near existing elements and beyond
            candidates_to_try = set()
            for existing in U:
                for offset in range(1, max(10, max_val // 5) + 1):
                    candidates_to_try.add(existing + offset)
                    if existing - offset > 0:
                        candidates_to_try.add(existing - offset)
            candidates_to_try.add(max_val + random.randint(1, max(10, max_val // 3)))
            
            for candidate in sorted(candidates_to_try):
                if candidate in U or candidate < 0:
                    continue
                test_U = sorted(U + [candidate])
                sc = score_set(test_U)
                if sc > best_sc:
                    best_sc = sc
                    best_val = candidate
            
            if best_val is not None:
                U.append(best_val)
                U.sort()
            else:
                break
        return U
    
    candidates.append(greedy_construct(20))
    
    best_U = None
    best_score = 0
    
    # Run local search on each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:  # Leave 30s for final polish
            break
        refined_U, refined_score = local_search(cand, max_iters=3000, initial_temp=1.0)
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
    
    # Final high-precision polish on the best
    if best_U is not None:
        final_U, final_score = final_polish(best_U, max_iters=15000)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    # One more round of local search with different parameters
    if best_U is not None and time.time() < deadline - 10:
        refined_U, refined_score = local_search(best_U, max_iters=5000, initial_temp=0.5)
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
        
        # Final polish again
        if time.time() < deadline - 5:
            final_U, final_score = final_polish(best_U, max_iters=10000)
            if final_score > best_score:
                best_score = final_score
                best_U = final_U
    
    if best_U is None:
        best_U = [0, 1, 3]
    
    return sorted(set(best_U))

# EVOLVE-BLOCK-END

import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def compute_score_fast(U):
        """Compute score using numpy for speed."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        if n < 2:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        
        # Compute sumset: all pairwise sums
        sums = U_arr[:, None] + U_arr[None, :]
        sumset = np.unique(sums)
        n_sum = len(sumset)
        
        # Compute difference set: all pairwise differences
        diffs = U_arr[:, None] - U_arr[None, :]
        diffset = np.unique(diffs)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def compute_score_small(U):
        """Compute score using pure Python for small sets (faster overhead)."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        if n < 2:
            return 0
        
        sumset = set()
        diffset = set()
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def get_score(U):
        n = len(U)
        if n <= 8:
            return compute_score_small(U)
        else:
            return compute_score_fast(U)
    
    # Known good starting points
    initial_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 2, 5, 11, 17],
        [0, 1, 4, 10, 12, 17, 22, 28],
    ]
    
    best_score = 0
    best_set = [0, 1, 3]
    
    for s in initial_sets:
        sc = get_score(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # Simulated annealing with various initializations
    random.seed(42)
    
    def simulate_anneal(initial, max_iters=5000, t0=1.0, cooling=0.999):
        """Simulated annealing to optimize the set."""
        current = list(initial)
        current_score = get_score(current)
        best_local = list(current)
        best_local_score = current_score
        
        t = t0
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose mutation
            mutation_type = random.random()
            
            if mutation_type < 0.3 and len(current) < 20:
                # Add an element
                max_cur = max(current)
                # Try adding near existing or slightly beyond
                candidates = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 8):
                    candidates.add(new_el)
                # Also try some random larger values
                for _ in range(5):
                    new_el = random.randint(1, max_cur + 20)
                    if new_el not in current:
                        candidates.add(new_el)
                
                if candidates:
                    new_el = random.choice(sorted(candidates))
                    new_set = sorted(current + [new_el])
            elif mutation_type < 0.5 and len(current) > 3:
                # Remove an element (not 0)
                removable = [i for i in range(len(current)) if current[i] != 0]
                if removable:
                    idx = random.choice(removable)
                    new_set = current[:idx] + current[idx+1:]
                    new_set = sorted(set(new_set))
                else:
                    continue
            elif mutation_type < 0.7:
                # Replace an element
                if len(current) < 3:
                    continue
                replaceable = [i for i in range(len(current)) if current[i] != 0]
                if not replaceable:
                    continue
                idx = random.choice(replaceable)
                old_el = current[idx]
                max_cur = max(current)
                # Try nearby or random replacement
                if random.random() < 0.5:
                    new_el = random.randint(max(1, old_el - 5), min(max_cur + 10, old_el + 6))
                else:
                    new_el = random.randint(1, max_cur + 15)
                
                if new_el == old_el or new_el in current:
                    continue
                
                new_set = list(current)
                new_set[idx] = new_el
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            else:
                # Swap two elements (neither is 0)
                if len(current) < 4:
                    continue
                non_zero = [i for i in range(len(current)) if current[i] != 0]
                if len(non_zero) < 2:
                    continue
                i, j = random.sample(non_zero, 2)
                # Generate new values
                max_cur = max(current)
                new_i = random.randint(1, max_cur + 10)
                new_j = random.randint(1, max_cur + 10)
                
                if new_i in current or new_j in current or new_i == new_j:
                    continue
                
                new_set = list(current)
                new_set[i] = new_i
                new_set[j] = new_j
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) != len(current):
                    continue
            
            new_score = get_score(new_set)
            
            # Accept or reject
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if new_score > best_local_score:
                    best_local = list(new_set)
                    best_local_score = new_score
            else:
                # Accept with probability based on temperature
                delta = current_score - new_score
                if delta < 0:
                    accept_prob = 1.0
                else:
                    if t > 1e-10:
                        accept_prob = math.exp(-delta / t)
                    else:
                        accept_prob = 0.0
                
                if random.random() < accept_prob:
                    current = new_set
                    current_score = new_score
            
            t *= cooling
        
        return best_local, best_local_score
    
    # Run simulated annealing from multiple initializations
    num_sa_runs = 15
    for i in range(num_sa_runs):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if i == 0:
            init = list(best_set)
        elif i == 1:
            # Golomb ruler
            init = [0, 1, 4, 10, 18, 23, 25, 34]
        elif i == 2:
            # Sparse set with large gaps
            init = [0, 1, 3, 7, 15, 31]
        elif i == 3:
            # Medium sparse
            init = [0, 1, 4, 9, 16, 25]
        elif i == 4:
            # Random sparse
            max_val = random.randint(20, 60)
            size = random.randint(5, 10)
            init = [0] + sorted(random.sample(range(1, max_val + 1), size - 1))
        elif i == 5:
            # Powers-like
            init = [0, 1, 2, 5, 11, 23]
        elif i == 6:
            # Fibonacci-like
            init = [0, 1, 2, 3, 5, 8, 13, 21]
        elif i == 7:
            # Mixed
            init = [0, 1, 3, 6, 11, 18, 27]
        elif i == 8:
            # Dense small
            init = [0, 1, 2, 4, 7, 11]
        elif i == 9:
            # Very sparse
            init = [0, 1, 5, 13, 29]
        elif i == 10:
            # Medium
            init = [0, 1, 4, 8, 14, 22, 31]
        elif i == 11:
            # Random
            max_val = random.randint(15, 50)
            size = random.randint(4, 9)
            init = [0] + sorted(random.sample(range(1, max_val + 1), size - 1))
        elif i == 12:
            # Structured
            init = [0, 1, 4, 10, 19, 31, 46]
        elif i == 13:
            # Another random
            max_val = random.randint(20, 70)
            size = random.randint(5, 12)
            init = [0] + sorted(random.sample(range(1, max_val + 1), size - 1))
        else:
            init = list(best_set)
        
        local_best, local_score = simulate_anneal(init, max_iters=3000, t0=0.5, cooling=0.9995)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_best
    
    # Final local refinement of best set
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        no_improve = 0
        max_no_improve = 200
        
        while no_improve < max_no_improve and time.time() < deadline - 3:
            improved = False
            
            # Try adding
            if len(current) < 18:
                max_cur = max(current)
                candidates = set()
                for el in current:
                    for delta in range(-2, 3):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 6):
                    candidates.add(new_el)
                
                for new_el in sorted(candidates):
                    cand = sorted(current + [new_el])
                    s = get_score(cand)
                    if s > current_score + 1e-12:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            # Try replacing
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    for new_el in range(max(1, old_el - 4), min(max_cur + 6, old_el + 5)):
                        if new_el == old_el or new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != len(current):
                            continue
                        s = get_score(cand)
                        if s > current_score + 1e-12:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing
            if len(current) > 4:
                removable = [i for i in range(len(current)) if current[i] != 0]
                for idx in removable:
                    cand = current[:idx] + current[idx+1:]
                    s = get_score(cand)
                    if s > current_score + 1e-12:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)

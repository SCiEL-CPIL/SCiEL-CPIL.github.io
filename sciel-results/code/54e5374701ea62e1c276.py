import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N_ELEMENTS = 129  # [0, 128]
    
    def calc_mult(A_list):
        """Compute multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        n = len(A_list)
        A_arr = np.array(A_list)
        for i in range(n):
            a = A_arr[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for j > i
            for j in range(i + 1, n):
                b = A_arr[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    def calc_score(A):
        """Returns (gaps, peak) where gaps is number of uncovered sums, peak is max multiplicity."""
        M = calc_mult(A)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)
    
    def fast_calc_mult(A_set):
        """Compute multiplicity using set membership for speed."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    def get_pairs_for_sum(A_set, target):
        """Get all unordered pairs (a,b) with a<=b, a+b=target, a,b in A_set."""
        pairs = []
        for a in A_set:
            b = target - a
            if b >= a and b in A_set:
                pairs.append((a, b))
        return pairs
    
    def try_remove_element(A_set, idx, deadline):
        """Try removing element at idx. Returns (new_A, new_M, score) if valid, else None."""
        if time.time() > deadline:
            return None
        A_list = sorted(A_set)
        elem = A_list[idx]
        new_A_set = set(A_set)
        new_A_set.discard(elem)
        if not new_A_set:
            return None
        new_M = fast_calc_mult(new_A_set)
        gaps = int(np.sum(new_M == 0))
        if gaps > 0:
            return None
        new_peak = int(np.max(new_M))
        return (new_A_set, new_M, (0, new_peak))
    
    def try_swap_element(A_set, idx, new_elem, deadline):
        """Try swapping element at idx with new_elem. Returns (new_A, new_M, score) if valid, else None."""
        if time.time() > deadline:
            return None
        A_list = sorted(A_set)
        if idx >= len(A_list):
            return None
        old_elem = A_list[idx]
        if new_elem == old_elem:
            return None
        new_A_set = set(A_set)
        new_A_set.discard(old_elem)
        new_A_set.add(new_elem)
        new_M = fast_calc_mult(new_A_set)
        gaps = int(np.sum(new_M == 0))
        if gaps > 0:
            return None
        new_peak = int(np.max(new_M))
        return (new_A_set, new_M, (0, new_peak))
    
    def local_search(A_set, M, deadline, max_iters=1000):
        """Local search to minimize peak multiplicity."""
        A_set = set(A_set)
        M = fast_calc_mult(A_set)
        current_score = calc_score(A_set)
        
        if current_score == (0, 1):
            return current_score, A_set, M
        
        improved = True
        iters = 0
        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1
            
            A_list = sorted(A_set)
            current_peak = int(np.max(M))
            
            # Find sums with peak multiplicity
            peak_sums = np.where(M == current_peak)[0] + MIN_SUM
            
            # Try to reduce peak by removing or swapping elements that contribute to peak sums
            best_move = None
            best_score = current_score
            
            # Strategy 1: Try removing elements
            for idx in range(len(A_list)):
                if time.time() > deadline:
                    break
                result = try_remove_element(A_set, idx, deadline)
                if result is not None:
                    new_A, new_M, new_score = result
                    if new_score < best_score:
                        best_score = new_score
                        best_move = ('remove', idx, new_A, new_M)
                        if best_score == (0, 1):
                            break
            
            # Strategy 2: Try swapping elements
            if best_move is None or best_score[1] > 1:
                available = [x for x in range(N_ELEMENTS) if x not in A_set]
                if available:
                    # Try random swaps
                    n_swaps = min(20, len(available))
                    for _ in range(n_swaps):
                        if time.time() > deadline:
                            break
                        idx = random.randint(0, len(A_list) - 1)
                        new_elem = random.choice(available)
                        result = try_swap_element(A_set, idx, new_elem, deadline)
                        if result is not None:
                            new_A, new_M, new_score = result
                            if new_score < best_score:
                                best_score = new_score
                                best_move = ('swap', idx, new_A, new_M)
                                if best_score == (0, 1):
                                    break
            
            if best_move is not None and best_score < current_score:
                current_score = best_score
                if best_move[0] == 'remove':
                    A_set = best_move[2]
                    M = best_move[3]
                else:
                    A_set = best_move[2]
                    M = best_move[3]
                improved = True
                if current_score == (0, 1):
                    break
        
        return current_score, A_set, M
    
    def simulated_annealing(A_set, M, deadline, start_temp=10.0, end_temp=0.1, max_iters=5000):
        """Simulated annealing to escape local minima."""
        A_set = set(A_set)
        M = fast_calc_mult(A_set)
        current_score = calc_score(A_set)
        best_score = current_score
        best_A = set(A_set)
        best_M = M.copy()
        
        if current_score == (0, 1):
            return current_score, A_set, M
        
        T = start_temp
        iters = 0
        while time.time() < deadline and iters < max_iters:
            iters += 1
            T = max(end_temp, T * 0.999)
            
            A_list = sorted(A_set)
            current_peak = int(np.max(M)) if M.size > 0 else 0
            
            # Generate a random move
            move_type = random.random()
            new_A_set = set(A_set)
            
            if move_type < 0.3:
                # Remove an element
                if len(A_list) > 1:
                    idx = random.randint(0, len(A_list) - 1)
                    new_A_set.discard(A_list[idx])
            elif move_type < 0.6:
                # Add an element
                available = [x for x in range(N_ELEMENTS) if x not in A_set]
                if available:
                    new_A_set.add(random.choice(available))
            else:
                # Swap
                if len(A_list) > 0:
                    idx = random.randint(0, len(A_list) - 1)
                    available = [x for x in range(N_ELEMENTS) if x not in A_set]
                    if available:
                        new_A_set.discard(A_list[idx])
                        new_A_set.add(random.choice(available))
            
            if new_A_set == A_set:
                continue
            
            new_M = fast_calc_mult(new_A_set)
            new_score = calc_score(new_A_set)
            
            # Accept if better, or with probability based on temperature
            if new_score < current_score:
                current_score = new_score
                A_set = new_A_set
                M = new_M
                if new_score < best_score:
                    best_score = new_score
                    best_A = set(A_set)
                    best_M = new_M.copy()
                if best_score == (0, 1):
                    break
            else:
                # Compute delta in a comparable way
                # If both valid, compare peaks
                if current_score[0] == 0 and new_score[0] == 0:
                    delta = new_score[1] - current_score[1]
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_score = new_score
                        A_set = new_A_set
                        M = new_M
                        if new_score < best_score:
                            best_score = new_score
                            best_A = set(A_set)
                            best_M = new_M.copy()
                        if best_score == (0, 1):
                            break
        
        return best_score, best_A, best_M
    
    def targeted_polish(A_set, M, deadline, max_rounds=100):
        """Targeted polish: focus on reducing peak multiplicity sums."""
        A_set = set(A_set)
        M = fast_calc_mult(A_set)
        score = calc_score(A_set)
        
        if score[0] > 0:
            return score, A_set, M
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            A_list = sorted(A_set)
            current_peak = int(np.max(M))
            
            if current_peak <= 1:
                break
            
            # Find sums with peak multiplicity
            peak_sums = np.where(M == current_peak)[0] + MIN_SUM
            
            # For each peak sum, try to reduce its multiplicity
            improved_this_round = False
            for target in peak_sums:
                if time.time() > deadline:
                    break
                
                pairs = get_pairs_for_sum(A_set, target)
                if len(pairs) < 2:
                    continue
                
                # Try removing one element from a pair
                for a, b in pairs:
                    if time.time() > deadline:
                        break
                    
                    # Try removing a
                    if a in A_set:
                        new_A_set = set(A_set)
                        new_A_set.discard(a)
                        if new_A_set:
                            new_M = fast_calc_mult(new_A_set)
                            if np.all(new_M >= 1):
                                new_peak = int(np.max(new_M))
                                if new_peak < current_peak:
                                    A_set = new_A_set
                                    M = new_M
                                    current_peak = new_peak
                                    improved_this_round = True
                                    break
                    
                    # Try removing b (if different from a)
                    if b != a and b in A_set:
                        new_A_set = set(A_set)
                        new_A_set.discard(b)
                        if new_A_set:
                            new_M = fast_calc_mult(new_A_set)
                            if np.all(new_M >= 1):
                                new_peak = int(np.max(new_M))
                                if new_peak < current_peak:
                                    A_set = new_A_set
                                    M = new_M
                                    current_peak = new_peak
                                    improved_this_round = True
                                    break
                
                if improved_this_round:
                    break
            
            if not improved_this_round:
                # Try swaps
                available = [x for x in range(N_ELEMENTS) if x not in A_set]
                if available and A_list:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        idx = random.randint(0, len(A_list) - 1)
                        new_elem = random.choice(available)
                        new_A_set = set(A_set)
                        new_A_set.discard(A_list[idx])
                        new_A_set.add(new_elem)
                        new_M = fast_calc_mult(new_A_set)
                        if np.all(new_M >= 1):
                            new_peak = int(np.max(new_M))
                            if new_peak < current_peak:
                                A_set = new_A_set
                                M = new_M
                                current_peak = new_peak
                                improved_this_round = True
                                break
            
            if not improved_this_round:
                break
        
        score = calc_score(A_set)
        return score, A_set, M
    
    def greedy_removal(A_set, M, deadline, max_rounds=500):
        """Greedy removal: remove elements while maintaining validity."""
        A_set = set(A_set)
        M = fast_calc_mult(A_set)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            if len(A_set) <= 1:
                break
            
            A_list = sorted(A_set)
            current_peak = int(np.max(M))
            
            best_removal = None
            best_peak = current_peak
            
            for idx in range(len(A_list)):
                if time.time() > deadline:
                    break
                elem = A_list[idx]
                new_A_set = set(A_set)
                new_A_set.discard(elem)
                if not new_A_set:
                    continue
                new_M = fast_calc_mult(new_A_set)
                if np.all(new_M >= 1):
                    new_peak = int(np.max(new_M))
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_removal = elem
                        if best_peak == 1:
                            break
            
            if best_removal is not None:
                A_set.discard(best_removal)
                M = fast_calc_mult(A_set)
            else:
                break
        
        score = calc_score(A_set)
        return score, A_set, M
    
    # Main search
    best_score = (float('inf'), float('inf'))
    best_A = None
    best_M = None
    
    # Phase 1: Try structured initial solutions
    structured_starts = [
        # All even numbers
        set(range(0, 129, 2)),
        # All numbers divisible by 2
        set(range(0, 129, 2)),
        # Numbers in [64, 128]
        set(range(64, 129)),
        # Numbers in [0, 64]
        set(range(0, 65)),
        # Alternating
        set(range(0, 129, 2)),
        # Random dense sets
    ]
    
    for start_set in structured_starts:
        if time.time() > deadline:
            break
        if not start_set:
            continue
        M = fast_calc_mult(start_set)
        score = calc_score(start_set)
        if score[0] == 0 and score < best_score:
            best_score = score
            best_A = set(start_set)
            best_M = M.copy()
    
    # Phase 2: Random restarts with local search
    for k in range(20, 129, 5):
        if time.time() > deadline:
            break
        
        n_restarts = max(3, int(10 * (1 - (k - 20) / 100)))
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            A = set(random.sample(range(N_ELEMENTS), k))
            M = fast_calc_mult(A)
            score = calc_score(A)
            
            if score[0] == 0:
                # Local search
                score, A, M = local_search(A, M, deadline, max_iters=200)
                # Targeted polish
                score, A, M = targeted_polish(A, M, deadline, max_rounds=50)
                # Greedy removal
                score, A, M = greedy_removal(A, M, deadline, max_rounds=200)
                
                if score < best_score:
                    best_score = score
                    best_A = set(A)
                    best_M = M.copy()
                
                if best_score == (0, 1):
                    return sorted(best_A)
    
    # Phase 3: Simulated annealing from best solution
    if best_A is not None and best_score[0] == 0:
        if time.time() < deadline - 10:
            sa_score, sa_A, sa_M = simulated_annealing(best_A, best_M, deadline - 10, 
                                                         start_temp=5.0, end_temp=0.05, 
                                                         max_iters=3000)
            if sa_score < best_score:
                best_score = sa_score
                best_A = sa_A
                best_M = sa_M
    
    # Phase 4: Final polish
    if best_A is not None and best_score[0] == 0 and time.time() < deadline - 5:
        polish_score, polish_A, polish_M = targeted_polish(best_A, best_M, deadline - 5, 
                                                            max_rounds=100)
        if polish_score < best_score:
            best_score = polish_score
            best_A = polish_A
            best_M = polish_M
        
        # Greedy removal
        final_score, final_A, final_M = greedy_removal(best_A, best_M, deadline, 
                                                        max_rounds=500)
        if final_score < best_score:
            best_score = final_score
            best_A = final_A
            best_M = final_M
    
    if best_A is None:
        # Fallback: return a valid solution
        return list(range(64, 129))
    
    return sorted(best_A)

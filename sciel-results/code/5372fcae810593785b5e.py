import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = list(range(target_min, target_max + 1))
    
    def calculate_counts(A_mask):
        """Vectorized count computation using outer product approach."""
        A_indices = np.where(A_mask)[0]
        if len(A_indices) == 0:
            return np.zeros(257, dtype=np.int32)
        # Use broadcasting: all pairs (a_i, a_j) with i <= j
        # For unordered pairs with equal summands allowed:
        # Count[s] = number of pairs (a,b) in A, a<=b, a+b=s
        # We can compute this as: for each s, count = #{a in A : a <= s-a and s-a in A}
        # More efficiently: use the fact that for each a in A, it contributes to sums a+a, a+(a+1), ..., a+128
        # Let's use a different approach: create a 2D indicator and count along diagonals
        # Actually, let's just do it with numpy for speed
        N = np.zeros(257, dtype=np.int32)
        m = len(A_indices)
        if m > 0:
            # Create all unordered pairs
            i_idx, j_idx = np.triu_indices(m)
            sums = A_indices[i_idx] + A_indices[j_idx]
            # Only count sums in target range
            valid = (sums >= target_min) & (sums <= target_max)
            if np.any(valid):
                np.add.at(N, sums[valid], 1)
        return N

    def calculate_counts_fast(A_mask):
        """Faster count using element-wise contributions."""
        N = np.zeros(257, dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        m = len(A_indices)
        if m == 0:
            return N
        # For each element a in A, it forms pairs (a, b) for all b in A with b >= a
        # Contribution to sum s from element a: 1 if (s-a) in A and (s-a) >= a
        # Let's compute differently: for each a, the sums it contributes to are a+a, a+(next), ...
        # Actually, let's just use the triu approach but vectorized
        i_idx, j_idx = np.triu_indices(m)
        sums = A_indices[i_idx] + A_indices[j_idx]
        valid = (sums >= target_min) & (sums <= target_max)
        if np.any(valid):
            np.add.at(N, sums[valid], 1)
        return N

    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = int(np.max(target_N))
        sum_sq = int(np.sum(target_N**2))
        return (0, peak, sum_sq)

    def greedy_init(k_min, k_max):
        A_mask = np.zeros(129, dtype=bool)
        k = random.randint(k_min, k_max)
        init_set = set(random.sample(range(129), k))
        for x in init_set:
            A_mask[x] = True
        
        N = calculate_counts_fast(A_mask)
        
        # Repair: add elements to cover missing sums
        while True:
            missing_sums = [s for s in target_sums if N[s] == 0]
            if not missing_sums:
                break
            
            best_x = None
            best_score = -1
            best_new_N = None
            
            candidates = [x for x in range(129) if not A_mask[x]]
            random.shuffle(candidates)
            
            A_indices = np.where(A_mask)[0]
            
            for x in candidates:
                # Compute new counts if we add x
                new_N = N.copy()
                for a in A_indices:
                    s = x + a
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                if 2*x in target_sums:
                    new_N[2*x] += 1
                
                covered = sum(1 for s in missing_sums if new_N[s] > 0)
                
                if covered > best_score:
                    best_score = covered
                    best_x = x
                    best_new_N = new_N
            
            if best_x is None or best_score == 0:
                break
                
            A_mask[best_x] = True
            N = best_new_N
            
        return A_mask, N

    def simulate_annealing(A_mask, N, time_limit):
        """Simulated annealing with peak-aware moves."""
        A_mask = A_mask.copy()
        N = N.copy()
        current_obj = get_objective(N)
        current_peak = current_obj[1]
        current_sum_sq = current_obj[2]
        
        if current_peak == float('inf'):
            return A_mask, N, current_obj
            
        best_A = A_mask.copy()
        best_N = N.copy()
        best_peak = current_peak
        best_sum_sq = current_sum_sq
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        T = T_start
        
        # Track no-improvement counter for adaptive cooling
        no_improve = 0
        patience = 50
        
        A_indices_list = list(np.where(A_mask)[0])
        
        while time.time() < time_limit:
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Swap: remove one element, add another
                if len(A_indices_list) == 0:
                    continue
                a = random.choice(A_indices_list)
                b = random.randint(0, 128)
                if A_mask[b]:
                    continue
                
                # Compute delta in counts
                # Remove a: subtract contributions of pairs (a, u) for u in A, u != a, plus (a,a)
                # Add b: add contributions of pairs (b, u) for u in A\{a}, plus (b,b)
                new_N = N.copy()
                
                # Remove a
                for u in A_indices_list:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                s = 2*a
                if target_min <= s <= target_max:
                    new_N[s] -= 1
                    
                # Add b
                for u in A_indices_list:
                    if u == a:
                        continue
                    s = b + u
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                s = 2*b
                if target_min <= s <= target_max:
                    new_N[s] += 1
                    
                new_obj = get_objective(new_N)
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
                
            elif move_type < 0.7:
                # Add an element
                candidates = [x for x in range(129) if not A_mask[x]]
                if not candidates:
                    continue
                b = random.choice(candidates)
                
                new_N = N.copy()
                for u in A_indices_list:
                    s = b + u
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                s = 2*b
                if target_min <= s <= target_max:
                    new_N[s] += 1
                    
                new_obj = get_objective(new_N)
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
                
            else:
                # Remove an element (only if we have enough elements)
                if len(A_indices_list) <= 10:
                    continue
                a = random.choice(A_indices_list)
                
                new_N = N.copy()
                for u in A_indices_list:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                s = 2*a
                if target_min <= s <= target_max:
                    new_N[s] -= 1
                    
                new_obj = get_objective(new_N)
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
            
            # Accept/reject based on simulated annealing
            if new_obj[0] > 0:
                # Invalid: only accept with very low probability
                if random.random() < 0.01:
                    # Accept invalid (to escape)
                    A_mask, N = _apply_move(A_mask, N, A_indices_list, a if move_type < 0.7 else b, b if move_type < 0.7 else a, move_type)
                    A_indices_list = list(np.where(A_mask)[0])
                    current_obj = get_objective(N)
                    current_peak = current_obj[1]
                    current_sum_sq = current_obj[2]
                continue
            
            # Valid move
            if new_peak < current_peak or (new_peak == current_peak and new_sum_sq < current_sum_sq):
                # Improvement: always accept
                A_mask, N = _apply_move(A_mask, N, A_indices_list, a if move_type < 0.7 else b, b if move_type < 0.7 else a, move_type)
                A_indices_list = list(np.where(A_mask)[0])
                current_obj = get_objective(N)
                current_peak = current_obj[1]
                current_sum_sq = current_obj[2]
                no_improve = 0
                
                if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                    best_peak = new_peak
                    best_sum_sq = new_sum_sq
                    best_A = A_mask.copy()
                    best_N = N.copy()
                    
            else:
                # Worse: accept with probability exp(-delta/T)
                delta = (new_peak - current_peak) * 100 + (new_sum_sq - current_sum_sq)
                if delta < 0:
                    delta = 0
                accept_prob = np.exp(-delta / max(T, 1e-10))
                if random.random() < accept_prob:
                    A_mask, N = _apply_move(A_mask, N, A_indices_list, a if move_type < 0.7 else b, b if move_type < 0.7 else a, move_type)
                    A_indices_list = list(np.where(A_mask)[0])
                    current_obj = get_objective(N)
                    current_peak = current_obj[1]
                    current_sum_sq = current_obj[2]
                no_improve += 1
            
            # Cool down
            if no_improve >= patience:
                T = max(T * 0.5, T_end)
                no_improve = 0
            else:
                T = max(T * 0.995, T_end)
        
        return best_A, best_N, (0, best_peak, best_sum_sq)

    def _apply_move(A_mask, N, A_indices_list, remove_elem, add_elem, move_type):
        """Apply a move to A_mask and N."""
        A_mask = A_mask.copy()
        N = N.copy()
        
        if move_type < 0.4:
            # Swap
            if remove_elem is not None and A_mask[remove_elem]:
                A_mask[remove_elem] = False
            if add_elem is not None:
                A_mask[add_elem] = True
        elif move_type < 0.7:
            # Add
            if add_elem is not None:
                A_mask[add_elem] = True
        else:
            # Remove
            if remove_elem is not None:
                A_mask[remove_elem] = False
        
        N = calculate_counts_fast(A_mask)
        return A_mask, N

    def peak_balanced_repair(A_mask, N, time_limit):
        """Repair by targeting the most congested sums."""
        A_mask = A_mask.copy()
        N = N.copy()
        current_obj = get_objective(N)
        current_peak = current_obj[1]
        current_sum_sq = current_obj[2]
        
        if current_peak == float('inf'):
            return A_mask, N, current_obj
            
        best_A = A_mask.copy()
        best_N = N.copy()
        best_peak = current_peak
        best_sum_sq = current_sum_sq
        
        A_indices_list = list(np.where(A_mask)[0])
        
        while time.time() < time_limit:
            # Find the most congested sums
            target_N = N[target_min:target_max+1]
            if np.max(target_N) == 0:
                break
            
            # Get the sums with highest count
            sorted_sums = np.argsort(-target_N)[:5]  # Top 5 most congested
            congested_sums = [target_min + s for s in sorted_sums]
            
            # Try to reduce congestion by swapping elements
            improved = False
            for _ in range(20):  # Try a few random swaps
                if len(A_indices_list) == 0:
                    break
                a = random.choice(A_indices_list)
                b = random.randint(0, 128)
                if A_mask[b]:
                    continue
                
                new_N = N.copy()
                
                # Remove a
                for u in A_indices_list:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                s = 2*a
                if target_min <= s <= target_max:
                    new_N[s] -= 1
                    
                # Add b
                for u in A_indices_list:
                    if u == a:
                        continue
                    s = b + u
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                s = 2*b
                if target_min <= s <= target_max:
                    new_N[s] += 1
                    
                new_obj = get_objective(new_N)
                if new_obj[0] > 0:
                    continue
                
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
                
                # Check if this reduces congestion on the most congested sums
                old_congested = sum(N[s] for s in congested_sums)
                new_congested = sum(new_N[s] for s in congested_sums)
                
                if new_peak < best_peak or (new_peak == best_peak and new_sum_sq < best_sum_sq):
                    A_mask[a] = False
                    A_mask[b] = True
                    N = new_N
                    A_indices_list = list(np.where(A_mask)[0])
                    best_peak = new_peak
                    best_sum_sq = new_sum_sq
                    best_A = A_mask.copy()
                    best_N = N.copy()
                    improved = True
                    break
            
            if not improved:
                # Try adding an element that helps balance
                candidates = [x for x in range(129) if not A_mask[x]]
                if candidates:
                    # Choose candidate that minimizes peak increase
                    best_candidate = None
                    best_candidate_peak = float('inf')
                    random.shuffle(candidates)
                    for x in candidates[:10]:  # Only check a few
                        new_N = N.copy()
                        for u in A_indices_list:
                            s = x + u
                            if target_min <= s <= target_max:
                                new_N[s] += 1
                        s = 2*x
                        if target_min <= s <= target_max:
                            new_N[s] += 1
                        new_obj = get_objective(new_N)
                        if new_obj[0] == 0 and new_obj[1] < best_candidate_peak:
                            best_candidate_peak = new_obj[1]
                            best_candidate = x
                        if time.time() > time_limit:
                            break
                    if best_candidate is not None and best_candidate_peak <= best_peak:
                        A_mask[best_candidate] = True
                        N = calculate_counts_fast(A_mask)
                        A_indices_list = list(np.where(A_mask)[0])
                        obj = get_objective(N)
                        if obj[1] < best_peak or (obj[1] == best_peak and obj[2] < best_sum_sq):
                            best_peak = obj[1]
                            best_sum_sq = obj[2]
                            best_A = A_mask.copy()
                            best_N = N.copy()
        
        return best_A, best_N, (0, best_peak, best_sum_sq)

    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')

    # Main Loop
    # Try different initial sizes
    for k_min, k_max in [(20, 30), (25, 35), (30, 40)]:
        for i in range(8):
            if time.time() > deadline:
                break
                
            A_mask, N = greedy_init(k_min, k_max)
            
            obj = get_objective(N)
            if obj[0] > 0:
                continue
                
            peak = obj[1]
            sum_sq = obj[2]
            
            if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
                best_peak = peak
                best_sum_sq = sum_sq
                best_A_mask = A_mask.copy()
                
            time_left = deadline - time.time()
            if time_left > 2:
                # Alternate between simulated annealing and peak-balanced repair
                if i % 2 == 0:
                    A_opt, N_opt, obj_opt = simulate_annealing(A_mask, N, time_left - 1)
                else:
                    A_opt, N_opt, obj_opt = peak_balanced_repair(A_mask, N, time_left - 1)
                
                if obj_opt[0] == 0:
                    if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                        best_peak = obj_opt[1]
                        best_sum_sq = obj_opt[2]
                        best_A_mask = A_opt.copy()
                        
            if time.time() > deadline - 5:
                break

    if best_A_mask is None:
        # Fallback: try to construct a valid set
        A_mask = np.zeros(129, dtype=bool)
        for x in range(64, 129):
            A_mask[x] = True
        N = calculate_counts_fast(A_mask)
        obj = get_objective(N)
        if obj[0] == 0:
            best_peak = obj[1]
            best_sum_sq = obj[2]
            best_A_mask = A_mask.copy()
        else:
            # Last resort: return a large set
            return list(range(129))
        
    # Convert mask to list
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list

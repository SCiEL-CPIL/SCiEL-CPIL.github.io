import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors
    neighbors = []
    weights = [q ** (n - 1 - i) for i in range(n)]
    
    for idx in range(total_words):
        w = [0] * n
        temp = idx
        for d in range(n-1, -1, -1):
            w[d] = temp % q
            temp //= q
            
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                new_idx = idx + (new_val - w[i]) * weights[i]
                neigh.append(new_idx)
        neighbors.append(np.array(neigh, dtype=np.int32))

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    def greedy_construct(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        remaining_count = total_words
        
        # M matrix for fast gain
        M = np.zeros((total_words, total_words), dtype=np.int8)
        for idx in range(total_words):
            M[idx, neighbors[idx]] = 1
            
        while remaining_count > 0:
            gains = (M @ uncovered).astype(np.int32)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[np.random.randint(len(candidates))]
            code.append(chosen_idx)
            
            nb_idx = neighbors[chosen_idx]
            num_covered = np.sum(uncovered[nb_idx])
            uncovered[nb_idx] = 0
            remaining_count -= num_covered
            
        return code, M

    def simulated_annealing(initial_code, M, time_limit):
        start_time = time.time()
        
        # State
        code_set = set(initial_code)
        code_list = list(initial_code)
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            coverage[neighbors[idx]] += 1
            
        uncovered_count = np.sum(coverage == 0)
        
        # Cost function
        # We want to minimize len(code).
        # If uncovered_count > 0, we add a large penalty to discourage invalid solutions,
        # but allow them to escape local minima.
        # Penalty should be high enough that valid solutions are always preferred if available,
        # but low enough to allow temporary invalidity if it leads to a much smaller code size later.
        # A common choice is Penalty = (total_words / 100) or similar.
        # Let's use a dynamic penalty or a fixed one.
        # If we use a fixed penalty P, cost = len(code) + P * uncovered_count.
        # We want P to be > 1, so that removing a word (cost -1) that covers P points (cost +P) is bad.
        # But if removing a word covers 0 points, cost -1 is good.
        # If adding a word covers P points, cost +1 - P is good if P > 1.
        
        P = 10 # Penalty per uncovered point
        
        def get_cost():
            return len(code_list) + P * uncovered_count
        
        current_cost = get_cost()
        best_code = code_list.copy()
        best_cost = len(best_code) # Only track valid solutions as best
        
        T = 50.0 # Initial temperature
        cooling_rate = 0.995
        
        while time.time() - start_time < time_limit:
            if T < 1e-4:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3 and len(code_list) > 0:
                # Remove a random codeword
                idx_to_remove = random.choice(code_list)
                code_list.remove(idx_to_remove)
                code_set.discard(idx_to_remove)
                
                # Update coverage
                nb = neighbors[idx_to_remove]
                old_cov = coverage[nb]
                coverage[nb] -= 1
                new_cov = coverage[nb]
                
                # Update uncovered_count
                # Points that were 1 and became 0 are now uncovered
                # Points that were >1 and became >=1 are still covered
                # delta_uncovered = count(old_cov == 1)
                delta_unc = np.sum(old_cov == 1)
                uncovered_count += delta_unc
                
            elif move_type < 0.6:
                # Add a random word
                # To be efficient, we can pick from the whole space
                new_idx = np.random.randint(total_words)
                if new_idx not in code_set:
                    code_list.append(new_idx)
                    code_set.add(new_idx)
                    
                    nb = neighbors[new_idx]
                    old_cov = coverage[nb]
                    coverage[nb] += 1
                    new_cov = coverage[nb]
                    
                    # Points that were 0 and became 1 are no longer uncovered
                    delta_unc = -np.sum(old_cov == 0)
                    uncovered_count += delta_unc
            else:
                # Swap: Remove one, Add one
                if len(code_list) > 0:
                    idx_to_remove = random.choice(code_list)
                    new_idx = np.random.randint(total_words)
                    
                    if new_idx not in code_set or new_idx == idx_to_remove:
                         if new_idx == idx_to_remove:
                             pass # No-op
                         else:
                             # Remove
                             code_list.remove(idx_to_remove)
                             code_set.discard(idx_to_remove)
                             nb_rem = neighbors[idx_to_remove]
                             old_cov_rem = coverage[nb_rem]
                             coverage[nb_rem] -= 1
                             delta_unc_rem = np.sum(old_cov_rem == 1)
                             
                             # Add
                             code_list.append(new_idx)
                             code_set.add(new_idx)
                             nb_add = neighbors[new_idx]
                             old_cov_add = coverage[nb_add]
                             coverage[nb_add] += 1
                             delta_unc_add = -np.sum(old_cov_add == 0)
                             
                             uncovered_count += delta_unc_rem + delta_unc_add

            new_cost = get_cost()
            
            # Acceptance
            if new_cost < current_cost:
                current_cost = new_cost
                if uncovered_count == 0:
                    if len(code_list) < best_cost:
                        best_cost = len(code_list)
                        best_code = code_list.copy()
            else:
                delta = new_cost - current_cost
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    if uncovered_count == 0:
                        if len(code_list) < best_cost:
                            best_cost = len(code_list)
                            best_code = code_list.copy()
                else:
                    # Reject: Undo the move
                    # This is tricky because we modified state in-place.
                    # We need to store the previous state or undo it.
                    # Given the complexity of undoing, it's better to simulate the move first,
                    # calculate cost, and only apply if accepted.
                    
                    # REFACTOR: Let's rewrite the loop to calculate delta cost without applying.
                    # This is safer.
                    pass # Placeholder, see below for correct implementation

            T *= cooling_rate

        return best_code

    # Corrected SA Implementation with Undo logic or Dry-Run
    
    def simulated_annealing_v2(initial_code, M, time_limit):
        start_time = time.time()
        
        code_list = initial_code.copy()
        code_set = set(code_list)
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            coverage[neighbors[idx]] += 1
        uncovered_count = np.sum(coverage == 0)
        
        P = 10
        def calc_cost(l, u):
            return l + P * u
            
        current_cost = calc_cost(len(code_list), uncovered_count)
        best_code = code_list.copy()
        best_cost = len(best_code)
        
        T = 50.0
        cooling_rate = 0.995
        
        while time.time() - start_time < time_limit and T > 1e-4:
            move_type = random.random()
            
            # Store state to undo if rejected
            # We can just store the specific changes
            
            if move_type < 0.3 and len(code_list) > 0:
                # Remove
                idx_to_remove = random.choice(code_list)
                nb = neighbors[idx_to_remove]
                old_cov = coverage[nb].copy()
                
                # Simulate
                new_cov = old_cov - 1
                delta_unc = np.sum(old_cov == 1) # 1->0 becomes uncovered
                new_unc = uncovered_count + delta_unc
                new_len = len(code_list) - 1
                new_cost = calc_cost(new_len, new_unc)
                
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply
                    code_list.remove(idx_to_remove)
                    code_set.discard(idx_to_remove)
                    coverage[nb] = new_cov
                    uncovered_count = new_unc
                    current_cost = new_cost
                    
                    if uncovered_count == 0 and new_len < best_cost:
                        best_cost = new_len
                        best_code = code_list.copy()
                        
            elif move_type < 0.6:
                # Add
                new_idx = np.random.randint(total_words)
                if new_idx not in code_set:
                    nb = neighbors[new_idx]
                    old_cov = coverage[nb].copy()
                    
                    # Simulate
                    new_cov = old_cov + 1
                    delta_unc = -np.sum(old_cov == 0) # 0->1 becomes covered
                    new_unc = uncovered_count + delta_unc
                    new_len = len(code_list) + 1
                    new_cost = calc_cost(new_len, new_unc)
                    
                    delta = new_cost - current_cost
                    
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Apply
                        code_list.append(new_idx)
                        code_set.add(new_idx)
                        coverage[nb] = new_cov
                        uncovered_count = new_unc
                        current_cost = new_cost
                        
                        if uncovered_count == 0 and new_len < best_cost:
                            best_cost = new_len
                            best_code = code_list.copy()
            else:
                # Swap
                if len(code_list) > 0:
                    idx_to_remove = random.choice(code_list)
                    new_idx = np.random.randint(total_words)
                    
                    if new_idx not in code_set:
                        nb_rem = neighbors[idx_to_remove]
                        nb_add = neighbors[new_idx]
                        
                        old_cov_rem = coverage[nb_rem].copy()
                        old_cov_add = coverage[nb_add].copy()
                        
                        # Simulate
                        new_cov_rem = old_cov_rem - 1
                        new_cov_add = old_cov_add + 1
                        
                        # Note: If nb_rem and nb_add overlap, we need to be careful.
                        # But since we are copying arrays, we can just compute the final coverage
                        # for the union of affected points?
                        # Actually, simpler:
                        # The change in coverage is:
                        # -1 for points in nb_rem
                        # +1 for points in nb_add
                        
                        # Let's just calculate the new uncovered count directly for the affected points.
                        # Affected points are union(nb_rem, nb_add).
                        # But copying full arrays is slow.
                        
                        # Optimization:
                        # delta_unc_rem = count(old_cov_rem == 1)
                        # delta_unc_add = -count(old_cov_add == 0)
                        # This is only valid if the sets are disjoint.
                        # If they overlap, a point in intersection:
                        # Old: c
                        # New: c - 1 + 1 = c. No change.
                        # So the simple delta calculation works IF we handle overlaps correctly?
                        # No.
                        # If p is in intersection:
                        # old_cov_rem[p] = c
                        # old_cov_add[p] = c
                        # new_cov[p] = c
                        # My delta logic:
                        # delta_unc_rem: if c==1, +1.
                        # delta_unc_add: if c==0, -1.
                        # If c==1: +1 - 0 = +1. But actual change is 0.
                        # So we MUST handle overlaps.
                        
                        # To handle overlaps efficiently:
                        # We can just update the coverage array in place for the simulation?
                        # No, we need to undo.
                        
                        # Alternative:
                        # Just apply, check, and undo if rejected.
                        # Since Swap is complex to simulate without applying, let's apply and undo.
                        
                        # Apply
                        code_list.remove(idx_to_remove)
                        code_set.discard(idx_to_remove)
                        coverage[nb_rem] -= 1
                        
                        code_list.append(new_idx)
                        code_set.add(new_idx)
                        coverage[nb_add] += 1
                        
                        new_unc = np.sum(coverage == 0)
                        new_len = len(code_list)
                        new_cost = calc_cost(new_len, new_unc)
                        
                        delta = new_cost - current_cost
                        
                        if delta < 0 or random.random() < np.exp(-delta / T):
                            current_cost = new_cost
                            uncovered_count = new_unc
                            if uncovered_count == 0 and new_len < best_cost:
                                best_cost = new_len
                                best_code = code_list.copy()
                        else:
                            # Undo
                            code_list.remove(new_idx)
                            code_set.discard(new_idx)
                            coverage[nb_add] -= 1
                            
                            code_list.append(idx_to_remove)
                            code_set.add(idx_to_remove)
                            coverage[nb_rem] += 1
                            # uncovered_count and current_cost remain unchanged

            T *= cooling_rate

        return best_code

    # Main
    start_time = time.time()
    best_code = None
    best_size = float('inf')
    
    # Run a few SA iterations with different seeds/initializations
    for i in range(5):
        if time.time() - start_time > 110:
            break
            
        code, M = greedy_construct(i)
        
        # Run SA for remaining time divided by iterations
        time_per_iter = (110 - (time.time() - start_time)) / (5 - i)
        
        sa_code = simulated_annealing_v2(code, M, time_per_iter)
        
        if len(sa_code) < best_size:
            best_size = len(sa_code)
            best_code = sa_code.copy()

    if best_code is None:
        code, _ = greedy_construct(0)
        best_code = code

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result

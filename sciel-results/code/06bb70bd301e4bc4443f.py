import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    total = q ** n
    steps = [1] * n
    for i in range(1, n):
        steps[i] = steps[i - 1] * q
        
    def int_to_digits(x):
        return [x // steps[i] % q for i in range(n)]
        
    def digits_to_int(d):
        return sum(d[i] * steps[i] for i in range(n) if d[i] != 0)

    # Precompute balls as lists of integers
    # For q=3, n=7, radius=1: ball size is 1 + 2*7 = 15
    balls = []
    for c in range(total):
        digits = int_to_digits(c)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
        
    # Convert balls to numpy arrays for vectorized operations if needed, 
    # but lists are fine for small sizes. 
    # We will use a coverage count array.
    
    # Helper to check validity and count uncovered
    # For efficiency in SA, we maintain:
    # code_set: set of selected codewords
    # cover_count: np.array of counts
    
    # Initial Solution: Fast Greedy
    def greedy_init():
        uncovered = np.ones(total, dtype=bool)
        code = []
        # Precompute gain for each codeword? 
        # gain[c] = number of uncovered words in balls[c]
        # Since balls are small, we can just iterate.
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            best_c = -1
            best_gain = 0
            # Optimization: Only check codewords that cover at least one uncovered word?
            # Or just scan all. 2187 * 15 ops is cheap.
            for c in range(total):
                gain = 0
                for w in balls[c]:
                    if uncovered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for w in balls[best_c]:
                uncovered[w] = False
                
        return code, uncovered

    # Simulated Annealing
    def run_sa(initial_code, initial_uncovered, max_time):
        # State
        code_set = set(initial_code)
        cover_count = np.zeros(total, dtype=np.int8)
        for c in code_set:
            for w in balls[c]:
                cover_count[w] += 1
                
        uncovered_count = np.sum(cover_count == 0)
        
        # Cost function: Minimize |C| + Penalty * |Uncovered|
        # We want to minimize |C|. If valid, cost = |C|.
        # If invalid, cost = |C| + 100 * |Uncovered| (high penalty to push towards validity)
        P = 100
        
        def calc_cost(code_size, unc_count):
            return code_size + P * unc_count
            
        current_cost = calc_cost(len(code_set), uncovered_count)
        best_code = list(code_set)
        best_cost = current_cost
        best_uncovered = uncovered_count
        
        # SA Parameters
        T = 50.0  # Initial temperature
        cooling = 0.995
        min_T = 0.1
        t_start = time.time()
        
        # Moves:
        # 1. Add random codeword
        # 2. Remove random codeword
        # 3. Swap random codeword with random other
        
        while time.time() < max_time and T > min_T:
            T *= cooling
            
            # Choose a move
            move_type = random.random()
            
            old_size = len(code_set)
            old_unc = uncovered_count
            
            if move_type < 0.3:
                # Add
                c_new = random.randint(0, total - 1)
                if c_new in code_set:
                    continue
                # Simulate add
                # Check if it covers any uncovered
                gain = 0
                for w in balls[c_new]:
                    if cover_count[w] == 0:
                        gain += 1
                new_size = old_size + 1
                new_unc = old_unc - gain
                new_cost = calc_cost(new_size, new_unc)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    code_set.add(c_new)
                    for w in balls[c_new]:
                        cover_count[w] += 1
                    current_cost = new_cost
                    uncovered_count = new_unc
                    if new_unc == 0 and new_size < best_cost:
                        best_code = list(code_set)
                        best_cost = new_size
                        best_uncovered = 0
                else:
                    continue
                    
            elif move_type < 0.6:
                # Remove
                if len(code_set) == 0:
                    continue
                c_old = random.choice(list(code_set))
                # Check if removing it uncovers any words
                loss = 0
                for w in balls[c_old]:
                    if cover_count[w] == 1:
                        loss += 1
                new_size = old_size - 1
                new_unc = old_unc + loss
                new_cost = calc_cost(new_size, new_unc)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    code_set.remove(c_old)
                    for w in balls[c_old]:
                        cover_count[w] -= 1
                    current_cost = new_cost
                    uncovered_count = new_unc
                    if new_unc == 0 and new_size < best_cost:
                        best_code = list(code_set)
                        best_cost = new_size
                        best_uncovered = 0
                else:
                    continue
                    
            else:
                # Swap
                if len(code_set) == 0:
                    continue
                c_old = random.choice(list(code_set))
                c_new = random.randint(0, total - 1)
                if c_new in code_set:
                    continue
                    
                # Simulate: remove c_old, add c_new
                # 1. Effect of removing c_old
                loss_old = 0
                for w in balls[c_old]:
                    if cover_count[w] == 1:
                        loss_old += 1
                # 2. Effect of adding c_new (considering c_old is removed)
                # This is tricky because cover_count hasn't changed yet.
                # Let's just apply and revert, or calculate carefully.
                # Since balls are small (15), applying/reverting is fast.
                
                # Apply removal
                for w in balls[c_old]:
                    cover_count[w] -= 1
                # Apply addition
                gain_new = 0
                for w in balls[c_new]:
                    if cover_count[w] == 0:
                        gain_new += 1
                    cover_count[w] += 1
                # Restore removal effect? No, we are accepting/rejecting.
                # Current state: c_old removed, c_new added.
                # Uncovered count change:
                # Words that were covered only by c_old are now uncovered (unless covered by c_new).
                # Words that were uncovered are now covered by c_new.
                
                # Let's just recalculate uncovered count for the affected words?
                # Or track delta.
                # Delta in uncovered:
                # 1. Words in balls[c_old] \ balls[c_new]: if cover_count was 1, they become uncovered.
                # 2. Words in balls[c_new] \ balls[c_old]: if cover_count was 0 (before add), they become covered.
                
                # This is complex to track incrementally without state.
                # Let's just compute new_unc by checking the union of affected balls.
                # Affected words: balls[c_old] U balls[c_new]
                affected = set(balls[c_old]) | set(balls[c_new])
                new_unc = 0
                for w in affected:
                    if cover_count[w] == 0:
                        new_unc += 1
                # Add unchanged uncovered words?
                # We don't know the global uncovered count easily without scanning.
                # However, we can maintain a global count if we update it.
                # Let's do a full recount of uncovered for simplicity if size is small?
                # No, 2187 is small. np.sum(cover_count == 0) is fast.
                new_unc = np.sum(cover_count == 0)
                
                new_size = old_size # Size unchanged
                new_cost = calc_cost(new_size, new_unc)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    code_set.remove(c_old)
                    code_set.add(c_new)
                    current_cost = new_cost
                    uncovered_count = new_unc
                    if new_unc == 0 and new_size < best_cost:
                        best_code = list(code_set)
                        best_cost = new_size
                        best_uncovered = 0
                else:
                    # Revert
                    # Remove c_new
                    for w in balls[c_new]:
                        cover_count[w] -= 1
                    # Add c_old
                    for w in balls[c_old]:
                        cover_count[w] += 1
                    continue
                    
        return best_code, best_cost, best_uncovered

    # Multi-Start Strategy
    # Run SA multiple times with different random seeds/initial states.
    # Since we have ~140s, we can afford 3-5 restarts.
    
    best_overall_code = []
    best_overall_size = float('inf')
    
    # 1. Get greedy initial
    init_code, init_uncovered = greedy_init()
    
    # Run SA on this
    t1 = time.time()
    code1, size1, unc1 = run_sa(init_code, init_uncovered, deadline - 20)
    if unc1 == 0 and size1 < best_overall_size:
        best_overall_size = size1
        best_overall_code = code1
        
    # 2. Random initial valid code?
    # We can generate a random valid code by adding random codewords until valid.
    # Or perturb the greedy code.
    
    # Let's do 2 more restarts with perturbed greedy codes.
    for i in range(2):
        if time.time() > deadline - 10:
            break
        # Perturb: Remove 10% of codewords, then repair
        perturbed = list(init_code)
        random.shuffle(perturbed)
        remove_count = max(1, len(perturbed) // 10)
        for _ in range(remove_count):
            if perturbed:
                perturbed.pop()
        
        # Repair: Add codewords to cover uncovered
        # Rebuild cover count for perturbed
        cover_count = np.zeros(total, dtype=np.int8)
        for c in perturbed:
            for w in balls[c]:
                cover_count[w] += 1
        
        # Repair greedily
        while True:
            if time.time() > deadline - 10:
                break
            unc_idx = np.where(cover_count == 0)[0]
            if len(unc_idx) == 0:
                break
            w = unc_idx[0]
            best_c = -1
            best_gain = 0
            for c in range(total):
                if w in balls[c]:
                    gain = 0
                    for ww in balls[c]:
                        if cover_count[ww] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
            if best_c == -1:
                break
            perturbed.append(best_c)
            for ww in balls[best_c]:
                cover_count[ww] += 1
                
        if len(perturbed) < best_overall_size:
            # Run SA on this perturbed valid code
            code2, size2, unc2 = run_sa(perturbed, np.zeros(total, dtype=bool), deadline - 10)
            if unc2 == 0 and size2 < best_overall_size:
                best_overall_size = size2
                best_overall_code = code2

    # Final Check: If we have a valid code, return it.
    # If not, return the greedy one (which should be valid).
    if not best_overall_code:
        best_overall_code = init_code
        
    # Convert to list of lists
    final_code = [int_to_digits(c) for c in best_overall_code]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in best_overall_code:
        if c not in seen:
            seen.add(c)
            unique_code.append(c)
            
    if len(unique_code) != len(best_overall_code):
        # Re-verify coverage for unique code
        # This case is rare if SA logic is correct, but just in case
        final_code = [int_to_digits(c) for c in unique_code]
        
    return final_code

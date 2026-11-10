import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp_bits[a] has bit b set iff a and b are comparable (a != b, one is proper subset of other)
    comp_bits = [0] * num_masks
    
    for a in range(num_masks):
        bits = 0
        if a != 0:
            # Proper subsets of a
            s = (a - 1) & a
            while True:
                bits |= (1 << s)
                if s == 0:
                    break
                s = (s - 1) & a
        # Proper supersets of a
        free = (~a) & full_mask
        if free != 0:
            s = (free - 1) & free
            while True:
                sup = a | s
                bits |= (1 << sup)
                if s == 0:
                    break
                s = (s - 1) & free
        comp_bits[a] = bits
    
    def greedy_init():
        # Start with a chain: 0, 1, 3, 7, ..., full_mask
        chain = [0]
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            chain.append(cur)
        selected = list(chain)
        selected_set = set(selected)
        selected_bits = 0
        for s in selected:
            selected_bits |= (1 << s)
        
        # Greedily add m-13 more sets
        for _ in range(m - 13):
            best_gain = -1
            best_mask = -1
            for candidate in range(num_masks):
                if candidate in selected_set:
                    continue
                gain = (comp_bits[candidate] & selected_bits).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_mask = candidate
            if best_mask == -1:
                break
            selected.append(best_mask)
            selected_set.add(best_mask)
            selected_bits |= (1 << best_mask)
        return selected
    
    def local_search(initial_masks, max_steps=200000, stuck_limit=200):
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        # Calculate initial score
        cur_score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (comp_bits[current[i]] >> current[j]) & 1:
                    cur_score += 1
        
        best_score = cur_score
        best_solution = list(current)
        
        stuck = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            idx = random.randint(0, m - 1)
            old = current[idx]
            
            # Propose new mask
            op = random.random()
            new = old
            if op < 0.35:
                # Add a bit
                zeros = [i for i in range(n) if not (old & (1 << i))]
                if zeros:
                    bit = random.choice(zeros)
                    new = old | (1 << bit)
            elif op < 0.70:
                # Remove a bit
                ones = [i for i in range(n) if (old & (1 << i))]
                if ones:
                    bit = random.choice(ones)
                    new = old ^ (1 << bit)
            elif op < 0.85:
                # Swap a bit
                ones = [i for i in range(n) if (old & (1 << i))]
                zeros = [i for i in range(n) if not (old & (1 << i))]
                if ones and zeros:
                    bo = random.choice(ones)
                    bz = random.choice(zeros)
                    new = (old ^ (1 << bo)) | (1 << bz)
            else:
                # Random jump
                new = random.randint(0, full_mask)
            
            if new == old or new in current_set:
                continue
            
            # Fast delta calculation using precomputed bitsets
            others = selected_bits & ~(1 << old)
            delta = (comp_bits[new] & others).bit_count() - (comp_bits[old] & others).bit_count()
            
            if delta > 0:
                current_set.remove(old)
                current_set.add(new)
                selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                current[idx] = new
                cur_score += delta
                stuck = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_solution = list(current)
            else:
                stuck += 1
                if stuck > stuck_limit:
                    # Random kick
                    idx = random.randint(0, m - 1)
                    old_kick = current[idx]
                    while True:
                        new_kick = random.randint(0, full_mask)
                        if new_kick not in current_set:
                            break
                    current_set.remove(old_kick)
                    current_set.add(new_kick)
                    selected_bits = (selected_bits & ~(1 << old_kick)) | (1 << new_kick)
                    current[idx] = new_kick
                    # Recalculate score
                    cur_score = 0
                    for i in range(m):
                        for j in range(i + 1, m):
                            if (comp_bits[current[i]] >> current[j]) & 1:
                                cur_score += 1
                    stuck = 0
        
        return best_score, best_solution
    
    def two_opt_search(initial_masks, max_steps=50000):
        """Try pairwise swaps to improve the solution."""
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        cur_score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (comp_bits[current[i]] >> current[j]) & 1:
                    cur_score += 1
        
        best_score = cur_score
        best_solution = list(current)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline - 5.0:
                        break
                    old_i = current[i]
                    old_j = current[j]
                    others = selected_bits & ~(1 << old_i) & ~(1 << old_j)
                    
                    # Try swapping with random candidates
                    for _ in range(20):
                        new_i = random.randint(0, full_mask)
                        new_j = random.randint(0, full_mask)
                        if new_i == old_i or new_j == old_j or new_i == new_j:
                            continue
                        if new_i in current_set or new_j in current_set:
                            continue
                        
                        # Calculate delta
                        gain_i = (comp_bits[new_i] & others).bit_count()
                        gain_j = (comp_bits[new_j] & others).bit_count()
                        gain_ij = 1 if ((comp_bits[new_i] >> new_j) & 1) or ((comp_bits[new_j] >> new_i) & 1) else 0
                        
                        loss_i = (comp_bits[old_i] & others).bit_count()
                        loss_j = (comp_bits[old_j] & others).bit_count()
                        loss_ij = 1 if ((comp_bits[old_i] >> old_j) & 1) or ((comp_bits[old_j] >> old_i) & 1) else 0
                        
                        delta = gain_i + gain_j + gain_ij - loss_i - loss_j - loss_ij
                        
                        if delta > 0:
                            # Accept the swap
                            current_set.remove(old_i)
                            current_set.remove(old_j)
                            current_set.add(new_i)
                            current_set.add(new_j)
                            selected_bits = (selected_bits & ~(1 << old_i) & ~(1 << old_j)) | (1 << new_i) | (1 << new_j)
                            current[i] = new_i
                            current[j] = new_j
                            cur_score += delta
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_solution = list(current)
                            break
        
        return best_score, best_solution
    
    def simulated_annealing(initial_masks, max_steps=300000):
        """Simulated annealing with bitset-based delta calculation."""
        current = list(initial_masks)
        current_set = set(current)
        selected_bits = 0
        for s in current:
            selected_bits |= (1 << s)
        
        cur_score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (comp_bits[current[i]] >> current[j]) & 1:
                    cur_score += 1
        
        best_score = cur_score
        best_solution = list(current)
        
        T = 10.0
        cooling_rate = 0.9999
        min_T = 0.001
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            if T < min_T:
                break
            
            idx = random.randint(0, m - 1)
            old = current[idx]
            
            # Generate neighbor
            r = random.random()
            new = old
            if r < 0.5:
                # Flip one bit
                bit = random.randint(0, n - 1)
                new = old ^ (1 << bit)
            elif r < 0.8:
                # Flip two bits
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new = old ^ (1 << b1) ^ (1 << b2)
            else:
                # Random mask
                new = random.randint(0, full_mask)
            
            if new == old or new in current_set:
                continue
            
            # Fast delta calculation
            others = selected_bits & ~(1 << old)
            delta = (comp_bits[new] & others).bit_count() - (comp_bits[old] & others).bit_count()
            
            if delta > 0:
                # Accept
                current_set.remove(old)
                current_set.add(new)
                selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                current[idx] = new
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_solution = list(current)
            else:
                # Accept with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    current_set.remove(old)
                    current_set.add(new)
                    selected_bits = (selected_bits & ~(1 << old)) | (1 << new)
                    current[idx] = new
                    cur_score += delta
            
            T *= cooling_rate
        
        return best_score, best_solution
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    # Multiple different initialization strategies
    init_strategies = []
    
    # Strategy 1: Chain-based greedy
    greedy_solution = greedy_init()
    init_strategies.append(greedy_solution)
    
    # Strategy 2: Random with some structure
    for r in range(3):
        random_solution = random.sample(range(num_masks), m)
        init_strategies.append(random_solution)
    
    # Strategy 3: Concentrated around certain sizes
    for target_size in [3, 4, 5]:
        candidates = [mask for mask in range(num_masks) if bin(mask).count('1') == target_size]
        if len(candidates) >= m:
            random.shuffle(candidates)
            init_strategies.append(candidates[:m])
    
    # Run local search and simulated annealing on each initialization
    for init_sol in init_strategies:
        if time.time() > deadline - 20.0:
            break
        
        # Local search
        ls_score, ls_sol = local_search(init_sol)
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_solution = ls_sol
        
        # Simulated annealing
        if time.time() < deadline - 10.0:
            sa_score, sa_sol = simulated_annealing(init_sol)
            if sa_score > best_global_score:
                best_global_score = sa_score
                best_global_solution = sa_sol
    
    # Run two-opt search on the best solution found so far
    if best_global_solution is not None and time.time() < deadline - 10.0:
        to_score, to_sol = two_opt_search(best_global_solution)
        if to_score > best_global_score:
            best_global_score = to_score
            best_global_solution = to_sol
    
    # Final polish: run more local search and simulated annealing steps on the best solution
    if best_global_solution is not None and time.time() < deadline - 5.0:
        ls_score, ls_sol = local_search(best_global_solution, max_steps=100000, stuck_limit=300)
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_solution = ls_sol
        
        if time.time() < deadline - 3.0:
            sa_score, sa_sol = simulated_annealing(best_global_solution, max_steps=100000)
            if sa_score > best_global_score:
                best_global_score = sa_score
                best_global_solution = sa_sol
    
    # Convert to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result

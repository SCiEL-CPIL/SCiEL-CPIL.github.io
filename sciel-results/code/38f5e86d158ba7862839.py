def construct():
    """Construct a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.
    
    Uses Simulated Annealing to maximize the size of the set.
    """
    import random
    import math
    
    required = [1, 2, 5, 11]
    N = 284
    
    def count_collisions(arr):
        """Count the number of duplicate sums."""
        sums = set()
        collisions = 0
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                if s in sums:
                    collisions += 1
                else:
                    sums.add(s)
        return collisions

    def is_valid(arr):
        return count_collisions(arr) == 0

    def greedy_init():
        """Greedy construction: add smallest valid numbers."""
        a = sorted(required)
        sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                
        for x in range(12, N + 1):
            new_sums = {x + y for y in a} | {2 * x}
            if not (new_sums & sums) and len(new_sums) == len(a) + 1:
                a.append(x)
                sums.update(new_sums)
                a.sort()
        return a

    def simulated_annealing(initial_arr, max_iters=50000, start_temp=50.0, end_temp=0.05):
        current_arr = sorted(initial_arr)
        current_collisions = count_collisions(current_arr)
        current_size = len(current_arr)
        current_cost = -1000 * current_size + current_collisions
        
        best_arr = current_arr[:]
        best_size = current_size
        best_cost = current_cost
        
        T = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            r = random.random()
            new_arr = None
            
            if r < 0.3:
                # Insert
                used = set(current_arr)
                candidates = [x for x in range(1, N+1) if x not in used]
                if candidates:
                    x = random.choice(candidates)
                    new_arr = current_arr + [x]
            elif r < 0.9:
                # Swap
                if current_arr:
                    used = set(current_arr)
                    candidates = [x for x in range(1, N+1) if x not in used]
                    if candidates:
                        idx = random.randint(0, len(current_arr) - 1)
                        x = random.choice(candidates)
                        new_arr = current_arr[:]
                        new_arr[idx] = x
            else:
                # Remove
                non_required_indices = [i for i, x in enumerate(current_arr) if x not in required]
                if non_required_indices:
                    idx = random.choice(non_required_indices)
                    new_arr = current_arr[:]
                    new_arr.pop(idx)
            
            if new_arr is None:
                continue
                
            new_collisions = count_collisions(new_arr)
            new_size = len(new_arr)
            new_cost = -1000 * new_size + new_collisions
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_arr = new_arr
                current_collisions = new_collisions
                current_size = new_size
                current_cost = new_cost
                
                if new_collisions == 0:
                    if new_size > best_size:
                        best_arr = new_arr[:]
                        best_size = new_size
                        best_cost = new_cost

            T *= cooling
            if T < end_temp:
                T = end_temp

        return best_arr

    # Run multiple SA rounds with different seeds to improve robustness
    best_overall = greedy_init()
    best_size = len(best_overall)
    
    for seed in [42, 123, 456, 789, 101]:
        random.seed(seed)
        init = greedy_init()
        res = simulated_annealing(init, max_iters=50000, start_temp=50.0, end_temp=0.05)
        if len(res) > best_size:
            best_size = len(res)
            best_overall = res
            
    return sorted(best_overall)

import time
import random

def construct():
    fixed = {1, 2, 5, 11}
    N = 284
    
    def init_sums(S):
        sums = set()
        s_list = sorted(S)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums

    def is_valid_add(S, sums, x):
        if x in S:
            return False
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    def add_to_set(S, sums, x):
        S.add(x)
        sums.add(2 * x)
        for y in S:
            if y != x:
                sums.add(x + y)
                
    def remove_from_set(S, sums, x):
        S.remove(x)
        sums.discard(2 * x)
        for y in S:
            sums.discard(x + y)

    def greedy_init(strategy, rng):
        S = set(fixed)
        sums = init_sums(S)
        
        if strategy == 'small':
            candidates = list(range(1, N + 1))
        elif strategy == 'large':
            candidates = list(range(N, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif strategy == 'greedy_density':
            # Try to pick elements that leave room for more
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
        for x in candidates:
            if is_valid_add(S, sums, x):
                add_to_set(S, sums, x)
        return S, sums

    def local_search(S, sums, best_size, best_set, deadline, rng):
        """Perform local search with simulated annealing."""
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
        
        # Temperature for simulated annealing
        temp = 10.0
        cooling = 0.995
        min_temp = 0.1
        max_iterations = 10000
        
        iteration = 0
        stuck = 0
        
        while time.time() < deadline and iteration < max_iterations:
            iteration += 1
            
            # Try to add a random element
            x = rng.randint(1, N)
            if x not in S and is_valid_add(S, sums, x):
                add_to_set(S, sums, x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                temp = max(min_temp, temp * cooling)
                stuck = 0
                continue
            
            # Try to swap: remove one non-fixed, add another
            removable = [y for y in S if y not in fixed]
            if not removable:
                break
            
            y = rng.choice(removable)
            
            # Tentatively remove y
            remove_from_set(S, sums, y)
            
            # Try to add a new element
            new_size = 0
            best_x = None
            best_gain = 0
            
            # Try several candidates
            for _ in range(10):
                x = rng.randint(1, N)
                if x not in S and is_valid_add(S, sums, x):
                    # This swap increases size by 0 (remove 1, add 1)
                    # But might lead to better future additions
                    new_size = len(S) + 1
                    best_x = x
                    break
            
            if best_x is not None:
                add_to_set(S, sums, best_x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                temp = max(min_temp, temp * cooling)
                stuck = 0
            else:
                # Revert
                add_to_set(S, sums, y)
                stuck += 1
                
                # With some probability, accept a worse state to escape local optima
                if stuck > 50 and rng.random() < 0.1:
                    # Kick: remove a random non-fixed element
                    removable_kick = [y for y in S if y not in fixed]
                    if removable_kick:
                        y_kick = rng.choice(removable_kick)
                        remove_from_set(S, sums, y_kick)
                        stuck = 0
            
            temp = max(min_temp, temp * cooling)
            
            if stuck > 200:
                break
        
        return best_size, best_set

    def two_opt_swap(S, sums, best_size, best_set, deadline, rng):
        """Try systematic two-opt swaps to improve the solution."""
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
        
        removable = [y for y in S if y not in fixed]
        if not removable:
            return best_size, best_set
        
        max_swaps = 500
        swaps_done = 0
        
        while time.time() < deadline and swaps_done < max_swaps:
            swaps_done += 1
            
            y = rng.choice(removable)
            remove_from_set(S, sums, y)
            
            # Try to find a good replacement
            best_x = None
            for _ in range(20):
                x = rng.randint(1, N)
                if x not in S and is_valid_add(S, sums, x):
                    best_x = x
                    break
            
            if best_x is not None:
                add_to_set(S, sums, best_x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
            else:
                add_to_set(S, sums, y)
        
        return best_size, best_set

    best_set = set(fixed)
    best_size = len(fixed)
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random()
    
    # Multiple restarts with different strategies
    strategies = ['small', 'large', 'random', 'random', 'random', 'random', 'random', 'random']
    
    for strat in strategies:
        if time.time() > deadline:
            break
            
        S, sums = greedy_init(strat, rng)
        
        # Local search
        best_size, best_set = local_search(S, sums, best_size, best_set, deadline, rng)
        
        # Two-opt refinement
        if time.time() < deadline:
            # Rebuild sums from best_set
            S = set(best_set)
            sums = init_sums(S)
            best_size, best_set = two_opt_swap(S, sums, best_size, best_set, deadline, rng)
    
    return list(best_set)

import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    
    def compute_counts(S_list):
        counts = {}
        for i, x in enumerate(S_list):
            for y in S_list[i:]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def can_add(x, S_list, counts):
        # Check if adding x keeps all sums <= 2
        for y in S_list:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True
    
    def add_element(x, S_list, counts):
        S_list.append(x)
        for y in S_list[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
    
    def remove_element(x, S_list, counts):
        idx = S_list.index(x)
        S_list.pop(idx)
        for y in S_list:
            s = x + y
            if s in counts:
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
        s = 2 * x
        if s in counts:
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
    
    # Greedy initialization with density-aware ordering
    def greedy_init():
        S_list = []
        counts = {}
        candidates = list(range(1, N + 1))
        
        # Sort by a heuristic: prefer smaller elements first (they create fewer large sums)
        # but also consider "sparsity" - start with well-spaced elements
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_list, counts):
                add_element(x, S_list, counts)
        
        return S_list, counts
    
    # Simulated annealing with 2-swap moves
    def simulated_annealing(S_list, counts, time_limit):
        S_set = set(S_list)
        best_S = S_list[:]
        best_counts = dict(counts)
        best_size = len(S_list)
        
        # Temperature schedule
        T_start = 100.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        step = 0
        
        while time.time() < time_limit and T > T_end:
            step += 1
            
            # Decide move type: 1-add, 1-remove, 2-swap
            r = random.random()
            
            if r < 0.3:
                # Try to add one element
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                if candidates:
                    x = random.choice(candidates)
                    if can_add(x, S_list, counts):
                        add_element(x, S_list, counts)
                        S_set.add(x)
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_S = S_list[:]
                            best_counts = dict(counts)
                    else:
                        # Reject (or accept with probability)
                        pass
                # If we added, continue; if not, try remove
            elif r < 0.5:
                # Try to remove one element (to create space)
                if S_list:
                    x = random.choice(S_list)
                    remove_element(x, S_list, counts)
                    S_set.discard(x)
                    # Then try to add a different one
                    candidates = [y for y in range(1, N + 1) if y not in S_set]
                    if candidates:
                        y = random.choice(candidates)
                        if can_add(y, S_list, counts):
                            add_element(y, S_list, counts)
                            S_set.add(y)
                        else:
                            # Revert
                            add_element(x, S_list, counts)
                            S_set.add(x)
            else:
                # 2-swap: remove two, add two
                if len(S_list) >= 2:
                    x1, x2 = random.sample(S_list, 2)
                    remove_element(x1, S_list, counts)
                    remove_element(x2, S_list, counts)
                    S_set.discard(x1)
                    S_set.discard(x2)
                    
                    # Try to add two new elements
                    candidates = [y for y in range(1, N + 1) if y not in S_set]
                    random.shuffle(candidates)
                    
                    added = []
                    for y in candidates:
                        if len(added) == 2:
                            break
                        if can_add(y, S_list, counts):
                            add_element(y, S_list, counts)
                            S_set.add(y)
                            added.append(y)
                    
                    if len(added) < 2:
                        # Revert: re-add the removed ones
                        for y in added:
                            remove_element(y, S_list, counts)
                            S_set.discard(y)
                        add_element(x1, S_list, counts)
                        add_element(x2, S_list, counts)
                        S_set.add(x1)
                        S_set.add(x2)
                    else:
                        # Check if we improved
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_S = S_list[:]
                            best_counts = dict(counts)
            
            # Cool down
            T *= cooling_rate
            
            # Periodic restart if stuck
            if step % 500 == 0 and len(S_list) < best_size - 2:
                # Revert to best and restart
                S_list = best_S[:]
                S_set = set(S_list)
                counts = dict(best_counts)
        
        return best_S, best_counts, best_size
    
    # Main loop: multiple restarts
    best_size = 0
    best_set = []
    
    while time.time() < deadline:
        S_list, counts = greedy_init()
        
        # Local improvement: try to add more
        improved = True
        while improved:
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in set(S_list)]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S_list, counts):
                    add_element(x, S_list, counts)
                    improved = True
                    break
        
        # Simulated annealing phase
        remaining_time = deadline - time.time() - 5  # Leave 5s buffer
        if remaining_time > 2:
            S_list, counts, size = simulated_annealing(S_list, counts, time.time() + remaining_time)
        else:
            size = len(S_list)
        
        if size > best_size:
            best_size = size
            best_set = S_list[:]
        
        # Check if we have enough time for another restart
        if time.time() > deadline - 10:
            break
    
    return best_set

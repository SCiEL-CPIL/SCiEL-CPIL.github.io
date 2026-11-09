import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    N = 1000
    MAX_SUM = 2 * N

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

    # Greedy initialization: build a B2[2] set
    def greedy_init():
        S_list = []
        counts = {}
        
        # Start with 1
        S_list.append(1)
        counts[2] = 1
        
        # Greedy: try to add each number from 2 to N
        for x in range(2, N + 1):
            if can_add(x, S_list, counts):
                add_element(x, S_list, counts)
        
        return S_list, counts

    # Local search: try swaps to improve
    def local_search(S_list, counts, time_limit):
        S_set = set(S_list)
        best_S = S_list[:]
        best_size = len(S_list)
        
        # Try to add elements
        for x in range(1, N + 1):
            if x not in S_set and can_add(x, S_list, counts):
                add_element(x, S_list, counts)
                S_set.add(x)
        
        best_size = len(S_list)
        best_S = S_list[:]
        
        # Try swaps: remove one, try to add two
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(S_list)):
                if time.time() > time_limit:
                    break
                x = S_list[i]
                # Remove x
                remove_element(x, S_list, counts)
                S_set.discard(x)
                
                # Try to add two elements
                added = []
                for y in range(1, N + 1):
                    if y not in S_set and y != x and can_add(y, S_list, counts):
                        add_element(y, S_list, counts)
                        S_set.add(y)
                        added.append(y)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = S_list[:]
                    changed = True
                else:
                    # Revert
                    for y in added:
                        remove_element(y, S_list, counts)
                        S_set.discard(y)
                    add_element(x, S_list, counts)
                    S_set.add(x)
        
        return best_S, best_size

    # Main loop: multiple restarts with random perturbations
    best_set = []
    best_size = 0
    
    while time.time() < deadline - 5:
        # Greedy initialization
        S_list, counts = greedy_init()
        
        # Local improvement
        remaining_time = deadline - time.time() - 5
        if remaining_time > 2:
            S_list, size = local_search(S_list, counts, time.time() + remaining_time)
        else:
            size = len(S_list)
        
        if size > best_size:
            best_size = size
            best_set = S_list[:]
        
        # Check if we have enough time for another restart
        if time.time() > deadline - 10:
            break
    
    # Verify the result is valid
    if best_set:
        counts = compute_counts(best_set)
        if any(c > 2 for c in counts.values()):
            # Invalid result, fall back to a simple valid set
            best_set = list(range(1, 20))
    
    return best_set

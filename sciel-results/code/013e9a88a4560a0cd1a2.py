import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85  # 5s buffer for 90s limit
    
    # Pre-compute conflict masks for each element
    element_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask_bce = (1 << b) | (1 << c) | (1 << e)
            mask_ace = (1 << a) | (1 << c) | (1 << e)
            mask_abe = (1 << a) | (1 << b) | (1 << e)
            mask_abc = (1 << a) | (1 << b) | (1 << c)
            
            element_to_aps[a].append(mask_bce)
            element_to_aps[b].append(mask_ace)
            element_to_aps[c].append(mask_abe)
            element_to_aps[e].append(mask_abc)

    def can_add(x, in_set_int):
        for mask in element_to_aps[x]:
            if (in_set_int & mask) == mask:
                return False
        return True

    def get_size(in_set_int):
        return bin(in_set_int).count('1')

    def greedy_construct():
        in_set_int = 0
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, in_set_int):
                in_set_int |= (1 << x)
        return in_set_int

    def local_search(initial_set, time_limit):
        current_set = initial_set
        current_size = get_size(current_set)
        best_set = current_set
        best_size = current_size
        
        start_ls = time.time()
        
        while time.time() < time_limit:
            # Choose a move type
            # 0: 1-Kick (Remove 1, re-add 1 + random others)
            # 1: 2-Kick (Remove 2, re-add 2 + random others)
            move_type = random.randint(0, 1)
            
            in_elements = [i for i in range(N) if (current_set >> i) & 1]
            if not in_elements:
                break
                
            if move_type == 0:
                if len(in_elements) < 1: continue
                x = random.choice(in_elements)
                new_set = current_set & ~(1 << x)
                removed = [x]
            else:
                if len(in_elements) < 2: continue
                removed = random.sample(in_elements, 2)
                new_set = current_set
                for x in removed:
                    new_set &= ~(1 << x)
            
            # Try to add back removed elements
            random.shuffle(removed)
            for x in removed:
                if can_add(x, new_set):
                    new_set |= (1 << x)
            
            # Try to add some random other elements
            # Pick 10 random candidates from outside the set
            not_in_elements = [i for i in range(N) if not (new_set >> i) & 1]
            if not_in_elements:
                # Sample up to 20
                num_to_try = min(20, len(not_in_elements))
                candidates = random.sample(not_in_elements, num_to_try)
                for z in candidates:
                    if can_add(z, new_set):
                        new_set |= (1 << z)
            
            new_size = get_size(new_set)
            
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
            elif new_size == current_size and random.random() < 0.1:
                # Accept equal size moves to escape plateaus
                current_set = new_set
            else:
                # Revert
                pass # current_set is unchanged

        return best_set

    # Phase 1: Greedy Restarts
    best_overall_set = 0
    best_overall_size = 0
    top_sets = []
    
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        s = greedy_construct()
        size = get_size(s)
        if size > best_overall_size:
            best_overall_size = size
            best_overall_set = s
        top_sets.append((size, s))
        top_sets.sort(key=lambda x: x[0], reverse=True)
        if len(top_sets) > 3:
            top_sets.pop()

    # Phase 2: Local Search
    if top_sets:
        # Run local search on the best set found
        # Use remaining time minus a small buffer
        ls_deadline = deadline - 2
        best_set_ls = top_sets[0][1]
        final_set = local_search(best_set_ls, ls_deadline)
        
        final_size = get_size(final_set)
        if final_size > best_overall_size:
            best_overall_size = final_size
            best_overall_set = final_set

    # Final Polish
    for x in range(N):
        if not (best_overall_set >> x) & 1:
            if can_add(x, best_overall_set):
                best_overall_set |= (1 << x)

    result = [i + 1 for i in range(N) if (best_overall_set >> i) & 1]
    return sorted(result)

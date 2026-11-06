import time
import random

# EVOLVE-BLOCK-START

def _score_set(s):
    """
    Compute the number of distinct d > 0 such that there exist a, a+d, a+2d in s.
    """
    s_set = set(s)
    diffs_found = set()
    lst = sorted(s_set)
    n = len(lst)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if mid in s_set:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)

def _local_search(initial_set, deadline):
    """
    Perform local search starting from initial_set.
    Returns the best set found.
    """
    current_set = set(initial_set)
    current_score = _score_set(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    all_elements = set(range(1, 513))
    
    no_improvement_count = 0
    max_no_improvement = 20 # Restart if stuck
    
    while time.time() < deadline - 1.0:
        improved = False
        # Try to swap each element with any other
        # Iterate over a copy of the list to handle modifications
        elements_list = list(current_set)
        random.shuffle(elements_list)
        
        for x in elements_list:
            if time.time() > deadline - 1.0:
                break
            # x might have been removed if we improved and broke, but we break immediately
            if x not in current_set:
                continue
                
            # Candidates
            # To speed up, we can try a random subset of candidates first
            candidates = [y for y in all_elements if y not in current_set]
            random.shuffle(candidates)
            
            # Try all candidates? 
            # 512 candidates. 
            # If we find an improvement, we break.
            for y in candidates:
                new_set = current_set - {x} | {y}
                new_score = _score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    break # Restart the element loop
            
            if improved:
                no_improvement_count = 0
                break # Restart the element loop
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break
                
    return best_set

def _construct_greedy():
    """
    Greedy construction.
    """
    current_set = set()
    all_elements = set(range(1, 513))
    
    # Start with a small random seed
    seed_size = 5
    current_set = set(random.sample(range(1, 513), seed_size))
    
    for _ in range(32 - seed_size):
        best_elem = None
        best_score = -1
        
        candidates = list(all_elements - current_set)
        random.shuffle(candidates)
        
        for elem in candidates:
            new_set = current_set | {elem}
            score = _score_set(new_set)
            if score > best_score:
                best_score = score
                best_elem = elem
        
        if best_elem is not None:
            current_set.add(best_elem)
            
    return list(current_set)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    random.seed(42)
    
    best_set = None
    best_score = -1
    
    # Run multiple restarts
    # 1. Greedy + LS
    for i in range(3):
        if time.time() > deadline - 15.0:
            break
        random.seed(100 + i)
        g_set = _construct_greedy()
        g_score = _score_set(g_set)
        
        if g_score > best_score:
            best_score = g_score
            best_set = g_set
            
        # Refine with local search
        ls_deadline = deadline - 5.0
        if time.time() < ls_deadline - 5.0:
            l_set = _local_search(g_set, ls_deadline)
            l_score = _score_set(l_set)
            if l_score > best_score:
                best_score = l_score
                best_set = l_set
        
    # 2. Random + LS
    for i in range(3):
        if time.time() > deadline - 15.0:
            break
        random.seed(200 + i)
        r_set = set(random.sample(range(1, 513), 32))
        r_score = _score_set(r_set)
        
        if r_score > best_score:
            best_score = r_score
            best_set = r_set
            
        if time.time() < deadline - 10.0:
            l_set = _local_search(r_set, deadline - 5.0)
            l_score = _score_set(l_set)
            if l_score > best_score:
                best_score = l_score
                best_set = l_set

    # 3. Interval + LS
    if time.time() < deadline - 10.0:
        int_set = list(range(1, 33))
        int_score = _score_set(int_set)
        if int_score > best_score:
            best_score = int_score
            best_set = int_set
            
        l_set = _local_search(int_set, deadline - 5.0)
        l_score = _score_set(l_set)
        if l_score > best_score:
            best_score = l_score
            best_set = l_set

    if best_set is None:
        best_set = list(range(1, 33))
        
    return best_set

# EVOLVE-BLOCK-END

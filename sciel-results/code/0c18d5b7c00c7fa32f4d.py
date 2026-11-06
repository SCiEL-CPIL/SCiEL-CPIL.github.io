import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    # Pre-compute all 4-APs
    # AP is (a, b, c, d) such that b-a = c-b = d-c = diff
    aps = []
    element_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            element_to_aps[a].append(ap_idx)
            element_to_aps[b].append(ap_idx)
            element_to_aps[c].append(ap_idx)
            element_to_aps[e].append(ap_idx)
            
    num_aps = len(aps)
    
    # Helper functions for state management
    # in_set: numpy bool array
    # ap_count: numpy int array, count of elements from AP currently in set
    
    def init_state():
        in_set = np.zeros(N, dtype=np.bool_)
        ap_count = np.zeros(num_aps, dtype=np.int32)
        return in_set, ap_count

    def add_element(x, in_set, ap_count):
        # Check if adding x creates a 4-AP
        for ap_idx in element_to_aps[x]:
            if ap_count[ap_idx] == 3:
                return False
        # Safe to add
        in_set[x] = True
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] += 1
        return True

    def remove_element(x, in_set, ap_count):
        in_set[x] = False
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] -= 1

    def get_size(in_set):
        return np.sum(in_set)

    # 1. Initial Greedy Pass to get a good starting point
    # We will run multiple greedy restarts to find the best initial set
    best_in_set = None
    best_ap_count = None
    best_size = 0
    
    # Run a few greedy passes
    for _ in range(20):
        in_set, ap_count = init_state()
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if add_element(x, in_set, ap_count):
                pass
        
        size = get_size(in_set)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            best_ap_count = ap_count.copy()
            
    # 2. Local Search with Kicks
    current_in_set = best_in_set
    current_ap_count = best_ap_count
    current_size = best_size
    
    # Kick parameters
    kick_sizes = [1, 2, 3]
    
    while time.time() < deadline:
        # Try to add any available element (greedy improvement)
        # This is fast if we just check a few random candidates
        # But to be thorough, we can scan all not-in-set elements occasionally
        # For now, let's rely on the kick to unblock elements
        
        # Perform a kick
        # Choose a random kick size
        k = random.choice(kick_sizes)
        
        # Get list of elements in set
        current_elements = np.where(current_in_set)[0]
        
        if len(current_elements) < k:
            # Set is too small, just add random elements if possible
            for x in range(N):
                if not current_in_set[x]:
                    if add_element(x, current_in_set, current_ap_count):
                        pass
            continue
            
        # Select k random elements to remove
        removed_elements = random.sample(current_elements.tolist(), k)
        
        # Store state before kick to revert if bad
        old_in_set = current_in_set.copy()
        old_ap_count = current_ap_count.copy()
        
        # Remove elements
        for x in removed_elements:
            remove_element(x, current_in_set, current_ap_count)
            
        # Try to add back elements
        # We should try to add the removed elements first? No, that's useless.
        # We should try to add elements from outside the set.
        # To be efficient, we can just iterate through all elements not in set
        # and try to add them.
        
        # Optimization: Only try to add elements that were "blocked" by the removed ones?
        # Simpler: Just iterate through all N elements.
        # Since N=512, this is fast.
        
        for x in range(N):
            if not current_in_set[x]:
                if add_element(x, current_in_set, current_ap_count):
                    pass
        
        new_size = get_size(current_in_set)
        
        # Acceptance
        if new_size >= current_size:
            # Accept (including same size to explore)
            current_size = new_size
        else:
            # Revert
            current_in_set = old_in_set
            current_ap_count = old_ap_count
            # Optionally, we could accept with probability exp(-delta/T)
            # But for simplicity and safety, we only accept non-worsening moves
            # This makes it a Hill Climber. To escape local optima, the "kick" 
            # with k>1 is the key. If k=1, it's just swapping.
            # If k=2, we remove 2, might add 3.
            
        # Occasionally, if we are stuck (no improvement in last X steps), 
        # we might want to do a larger kick or restart.
        # But the random kick size helps.
        
        # Check for completion
        if current_size > 250: # Arbitrary high threshold
             break

    # Final polish: try to add any remaining elements
    for x in range(N):
        if not current_in_set[x]:
            if add_element(x, current_in_set, current_ap_count):
                pass

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if current_in_set[i]]
    return sorted(result)

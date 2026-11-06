import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    # Precompute: for each x, list of (other1, other2, other3) for each 4-AP containing x
    element_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            element_to_aps[a].append((b, c, e))
            element_to_aps[b].append((a, c, e))
            element_to_aps[c].append((a, b, e))
            element_to_aps[e].append((a, b, c))
    
    # Precompute bitmasks for each element's APs
    ap_masks = []
    for x in range(N):
        masks = []
        for (o1, o2, o3) in element_to_aps[x]:
            mask = (1 << o1) | (1 << o2) | (1 << o3)
            masks.append(mask)
        ap_masks.append(masks)
    
    def can_add(x, in_set_int):
        for mask in ap_masks[x]:
            if (in_set_int & mask) == mask:
                return False
        return True
    
    def add_to_int(x, in_set_int):
        return in_set_int | (1 << x)
    
    def remove_from_int(x, in_set_int):
        return in_set_int & ~(1 << x)
    
    def get_size_int(in_set_int):
        return bin(in_set_int).count('1')
    
    def fill_greedy(in_set_int):
        """Fill in_set_int with all elements that can be added, in order."""
        for x in range(N):
            if not (in_set_int >> x) & 1:
                if can_add(x, in_set_int):
                    in_set_int = add_to_int(x, in_set_int)
        return in_set_int
    
    def fill_greedy_random(in_set_int):
        """Fill in_set_int with all elements that can be added, in random order."""
        candidates = [x for x in range(N) if not (in_set_int >> x) & 1]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, in_set_int):
                in_set_int = add_to_int(x, in_set_int)
        return in_set_int
    
    # 1. Initial Greedy Pass with bitset
    best_in_set_int = 0
    best_size = 0
    
    num_restarts = 300
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        in_set_int = 0
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, in_set_int):
                in_set_int = add_to_int(x, in_set_int)
        
        size = get_size_int(in_set_int)
        if size > best_size:
            best_size = size
            best_in_set_int = in_set_int
    
    # 2. Local Search with Kicks using bitset
    current_in_set_int = best_in_set_int
    current_size = best_size
    
    kick_sizes = [1, 2, 3, 4]
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        k = random.choice(kick_sizes)
        
        current_elements = [i for i in range(N) if (current_in_set_int >> i) & 1]
        
        if len(current_elements) < k:
            current_in_set_int = fill_greedy(current_in_set_int)
            continue
            
        removed_elements = random.sample(current_elements, k)
        old_in_set_int = current_in_set_int
        
        for x in removed_elements:
            current_in_set_int = remove_from_int(x, current_in_set_int)
            
        # Try multiple fill strategies
        best_local = current_in_set_int
        best_local_size = get_size_int(current_in_set_int)
        
        # Strategy 1: sequential fill
        s1 = fill_greedy(current_in_set_int)
        sz1 = get_size_int(s1)
        if sz1 > best_local_size:
            best_local = s1
            best_local_size = sz1
        
        # Strategy 2: random fill
        s2 = fill_greedy_random(current_in_set_int)
        sz2 = get_size_int(s2)
        if sz2 > best_local_size:
            best_local = s2
            best_local_size = sz2
        
        current_in_set_int = best_local
        new_size = best_local_size
        
        if new_size > current_size:
            current_size = new_size
            no_improve_count = 0
        elif new_size == current_size:
            no_improve_count += 1
        else:
            current_in_set_int = old_in_set_int
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Reset to best and try different kick sizes
            current_in_set_int = best_in_set_int
            current_size = best_size
            no_improve_count = 0
            max_no_improve = random.randint(30, 80)
    
    # 3. Branch-and-bound refinement on the border
    # Identify elements not in the set that conflict with the set
    # For each such element, determine which elements in the set it conflicts with
    # Then try to find a better configuration by swapping
    
    # For each element not in set, find which elements in set block it
    def get_conflicts(x, in_set_int):
        """Return list of elements in in_set_int that conflict with x."""
        conflicts = []
        for mask in ap_masks[x]:
            # Find which bits in mask are set in in_set_int
            m = in_set_int & mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                conflicts.append(idx)
                m ^= lb
        return list(set(conflicts))
    
    # Try to improve by finding elements that can be added if we remove some blockers
    for iteration in range(50):
        if time.time() > deadline - 1:
            break
        
        current_elements = [i for i in range(N) if (current_in_set_int >> i) & 1]
        not_in_set = [i for i in range(N) if not (current_in_set_int >> i) & 1]
        
        # For each element not in set, find its blockers
        # Group by number of blockers
        candidates_by_blockers = {}
        for x in not_in_set:
            conflicts = get_conflicts(x, current_in_set_int)
            if len(conflicts) <= 3:
                if len(conflicts) not in candidates_by_blockers:
                    candidates_by_blockers[len(conflicts)] = []
                candidates_by_blockers[len(conflicts)].append((x, conflicts))
        
        # Try to add elements with fewest blockers by removing those blockers
        improved = False
        for num_blockers in sorted(candidates_by_blockers.keys()):
            if num_blockers > 3:
                break
            for x, conflicts in candidates_by_blockers[num_blockers]:
                # Try removing the conflicts and adding x
                old_state = current_in_set_int
                new_state = current_in_set_int
                for c in conflicts:
                    new_state = remove_from_int(c, new_state)
                new_state = add_to_int(x, new_state)
                new_state = fill_greedy(new_state)
                new_size = get_size_int(new_state)
                
                if new_size > current_size:
                    current_in_set_int = new_state
                    current_size = new_size
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Final polish
    current_in_set_int = fill_greedy(current_in_set_int)
    final_size = get_size_int(current_in_set_int)
    if final_size > best_size:
        best_size = final_size
        best_in_set_int = current_in_set_int
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (best_in_set_int >> i) & 1]
    return sorted(result)

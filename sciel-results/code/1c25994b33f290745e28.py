import random
import time

N = 256

def _is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S):
    """Check if S is an inclusion-maximal Sidon set in [1, N]."""
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if S U {x} is Sidon
        candidate = list(S) + [x]
        if not _is_sidon(candidate):
            continue
        return False
    return True

def _greedy_maximal_sidon(start=1, end=N, reverse=False, seed=None):
    """Generate a maximal Sidon set using a greedy strategy."""
    if seed is not None:
        random.seed(seed)
    
    selected = []
    sums = set()
    
    # Create list of candidates
    candidates = list(range(start, end + 1))
    if reverse:
        candidates.reverse()
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        # Check if x can be added
        added_sums = [x + a for a in selected] + [2 * x]
        # Check for duplicates within added_sums
        if len(set(added_sums)) != len(added_sums):
            continue
        # Check against existing sums
        if sums.intersection(added_sums):
            continue
        
        selected.append(x)
        sums.update(added_sums)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    def _get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def _can_add_with_sums(S, sums_S, x):
        new_sums = [2*x]
        for s in S:
            new_sums.append(x + s)
        if len(set(new_sums)) != len(new_sums):
            return False
        if sums_S.intersection(new_sums):
            return False
        return True

    def _is_maximal_fast(S, sums_S):
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if _can_add_with_sums(S, sums_S, x):
                return False
        return True

    def _find_redundant_elements(S, sums_S):
        """Find ALL elements whose removal still leaves S maximal."""
        redundant = []
        S_set = set(S)
        for i in range(len(S)):
            z = S[i]
            new_set = S[:i] + S[i+1:]
            if not new_set:
                continue
            new_sums = _get_sums(new_set)
            if _is_maximal_fast(new_set, new_sums):
                redundant.append(i)
        return redundant

    def _remove_redundant(S, sums_S):
        """Iteratively remove redundant elements, removing all possible at each step."""
        changed = True
        while changed:
            changed = False
            redundant = _find_redundant_elements(S, sums_S)
            if redundant:
                # Remove ALL redundant elements found in this pass
                for idx in sorted(redundant, reverse=True):
                    S = S[:idx] + S[idx+1:]
                sums_S = _get_sums(S)
                changed = True
        return S, sums_S

    def _local_search(S, sums_S, deadline):
        """Local search with removal and swap moves, tracking best solution."""
        current_set = S[:]
        current_sums = sums_S
        best_set = current_set[:]
        best_sums = current_sums
        best_size = len(current_set)
        
        while time.time() < deadline:
            current_size = len(current_set)
            if current_size <= 1:
                break
            
            # Track best
            if current_size < best_size:
                best_size = current_size
                best_set = current_set[:]
                best_sums = current_sums
            
            # Try removal first (most likely to improve)
            if current_size > 1:
                # Try all removals, collect all valid ones
                removals = []
                for i in range(current_size):
                    z = current_set[i]
                    new_set = current_set[:i] + current_set[i+1:]
                    if not new_set:
                        continue
                    new_sums = _get_sums(new_set)
                    if _is_maximal_fast(new_set, new_sums):
                        removals.append(i)
                
                if removals:
                    # Remove ALL valid redundant elements at once
                    for idx in sorted(removals, reverse=True):
                        current_set = current_set[:idx] + current_set[idx+1:]
                    current_sums = _get_sums(current_set)
                    continue
            
            # Try swaps
            if current_size > 0:
                z_idx = random.randint(0, current_size - 1)
                z = current_set[z_idx]
                
                # Try a few random x values
                found_swap = False
                for _ in range(30):
                    x = random.randint(1, N)
                    if x in current_set:
                        continue
                    
                    temp_set = current_set[:z_idx] + current_set[z_idx+1:]
                    temp_sums = _get_sums(temp_set)
                    
                    if _can_add_with_sums(temp_set, temp_sums, x):
                        new_set = temp_set + [x]
                        new_sums = temp_sums.union([x+s for s in temp_set] + [2*x])
                        
                        if _is_maximal_fast(new_set, new_sums):
                            current_set = new_set
                            current_sums = new_sums
                            found_swap = True
                            break
                
                if found_swap:
                    continue
            
            # If stuck, try adding an element to escape
            x = random.randint(1, N)
            if x not in current_set:
                if _can_add_with_sums(current_set, current_sums, x):
                    new_set = current_set + [x]
                    new_sums = current_sums.union([x+s for s in current_set] + [2*x])
                    current_set = new_set
                    current_sums = new_sums
                    continue
            
            break
        
        return best_set, best_sums

    best_size = N + 1
    best_set = []
    
    # Phase 1: Try structured sets and greedy to get initial solutions
    structured_sets = [
        [1, 3, 7, 15, 31, 63, 127],
        [1, 3, 7, 15, 31, 63],
        [1, 3, 7, 15, 31],
        [1, 3, 7, 15],
        [1, 3, 7],
        [1, 3],
        [1],
        [2, 5, 11, 23, 47, 95, 191],
        [2, 5, 11, 23, 47, 95],
        [2, 5, 11, 23, 47],
        [2, 5, 11, 23],
        [2, 5, 11],
        [2, 5],
        [4, 9, 21, 43, 87, 175],
        [4, 9, 21, 43, 87],
        [4, 9, 21, 43],
        [4, 9, 21],
        [4, 9],
        [6, 13, 27, 55, 111, 223],
        [6, 13, 27, 55, 111],
        [6, 13, 27, 55],
        [6, 13, 27],
        [6, 13],
        [8, 17, 35, 71, 143],
        [8, 17, 35, 71],
        [8, 17, 35],
        [8, 17],
        [10, 21, 43, 87, 175],
        [10, 21, 43, 87],
        [10, 21, 43],
        [10, 21],
        [12, 25, 51, 103, 207],
        [12, 25, 51, 103],
        [12, 25, 51],
        [12, 25],
    ]
    
    for s in structured_sets:
        if _is_sidon(s):
            s_sums = _get_sums(s)
            if _is_maximal_fast(s, s_sums):
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
    
    # Try multiple greedy seeds
    for seed in range(15):
        if time.time() > deadline:
            break
        s = _greedy_maximal_sidon(seed=seed)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Phase 2: Local search from the best initial solution
    if best_set:
        best_sums = _get_sums(best_set)
        
        # First, try to remove redundant elements
        best_set, best_sums = _remove_redundant(best_set, best_sums)
        best_size = len(best_set)
        
        # Then do local search
        deadline_local = min(deadline, start_time + 52.0)
        best_set, best_sums = _local_search(best_set, best_sums, deadline_local)
        best_size = len(best_set)
        
        # Try local search from a few other starting points
        for seed in range(8):
            if time.time() > deadline_local:
                break
            s = _greedy_maximal_sidon(seed=seed + 100)
            if len(s) < best_size + 3:  # Only if close to best
                s_sums = _get_sums(s)
                s, s_sums = _remove_redundant(s, s_sums)
                s, s_sums = _local_search(s, s_sums, deadline_local)
                if len(s) < best_size:
                    best_size = len(s)
                    best_set = s[:]
                    best_sums = s_sums
    
    # Phase 3: Final aggressive search if time remains
    if time.time() < deadline - 3.0:
        # Try random restarts with local search
        for attempt in range(30):
            if time.time() > deadline - 1.0:
                break
            
            # Generate a random small Sidon set
            random.seed(attempt + 500)
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            max_size = random.choice([4, 5, 6, 7, 8])
            for x in candidates:
                added_sums = [x + a for a in selected] + [2 * x]
                if len(set(added_sums)) != len(added_sums):
                    continue
                if sums.intersection(added_sums):
                    continue
                selected.append(x)
                sums.update(added_sums)
                if len(selected) >= max_size:
                    break
            
            if not selected:
                continue
            
            s_sums = _get_sums(selected)
            if not _is_maximal_fast(selected, s_sums):
                continue
            
            if len(selected) < best_size:
                selected, s_sums = _remove_redundant(selected, s_sums)
                selected, s_sums = _local_search(selected, s_sums, deadline - 1.0)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected[:]
                    best_sums = s_sums
    
    return best_set

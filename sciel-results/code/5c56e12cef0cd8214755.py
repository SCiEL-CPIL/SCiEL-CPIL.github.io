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
        # We need to check if adding x creates any duplicate sums.
        # Existing sums are from S.
        # New sums are x+s for s in S, and 2*x.
        # We don't need to recompute all sums of S if we cached them, 
        # but for simplicity in the check, let's just verify collisions.
        
        # Optimization: 
        # A collision occurs if:
        # 1. 2*x == s1 + s2 for some s1, s2 in S
        # 2. x + s1 == s2 + s3 for some s1, s2, s3 in S
        # 3. x + s1 == 2*s2 is not a collision type, it's just a sum.
        # Wait, the definition is: all a+b with a<=b are distinct.
        # So we check if any new sum (involving x) equals any existing sum (in S) or another new sum.
        
        # New sums: {2*x} U {x+s | s in S}
        # Existing sums: {s1+s2 | s1, s2 in S, s1<=s2}
        
        # To be efficient, we can precompute the set of sums of S.
        # But _is_maximal is called frequently.
        
        # Let's just use the simple O(|S|^2) check for the candidate set S U {x}.
        # Since |S| is small (< 20), this is fast.
        candidate = list(S) + [x]
        if not _is_sidon(candidate):
            continue
        # If it IS Sidon, then S is NOT maximal (because we could add x).
        # Wait, the logic is:
        # S is maximal iff for ALL x not in S, S U {x} is NOT Sidon.
        # So if we find ANY x such that S U {x} IS Sidon, then S is NOT maximal.
        # So _is_maximal returns False if we find such an x.
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
        
    # Ensure it's maximal? Greedy usually produces maximal sets if we iterate through all candidates?
    # Not necessarily. If we skip x because it conflicts, it might become addable later? No, adding more elements only creates more conflicts.
    # So if x is not addable now, it will never be addable later.
    # Thus, the greedy set is maximal.
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Leave 5s margin for 30s limit
    
    best_size = N + 1
    best_set = []
    
    # Initial solution
    # Try a few greedy starts
    for i in range(5):
        s = _greedy_maximal_sidon(seed=i)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Local Search
    # We try to minimize size by removing elements or swapping.
    # State: S
    # We maintain the current best S.
    
    # To speed up maximality check, we can cache the sums of S.
    # But _is_maximal is O(N * |S|^2). With |S|~15, N=256, it's ~256*225 ~ 57k ops.
    # 1000 checks = 57M ops. Python is slow, maybe 1-2 seconds per 1000 checks?
    # So we can do ~10,000-50,000 steps in 25s.
    
    def _get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def _can_add_with_sums(S, sums_S, x):
        # Check if x + s or 2x collides with sums_S or each other
        new_sums = [2*x]
        for s in S:
            new_sums.append(x + s)
            
        # Check duplicates in new_sums
        if len(set(new_sums)) != len(new_sums):
            return False
            
        # Check intersection with sums_S
        if sums_S.intersection(new_sums):
            return False
            
        return True

    def _is_maximal_fast(S, sums_S):
        # Check if any x can be added
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if _can_add_with_sums(S, sums_S, x):
                return False
        return True

    # Main Loop
    current_set = best_set[:]
    current_sums = _get_sums(current_set)
    
    # If initial isn't maximal (should be, but just in case), fix it?
    # Greedy produces maximal sets.
    
    while time.time() < deadline:
        current_size = len(current_set)
        
        # 1. Try to remove an element
        if current_size > 1:
            z_idx = random.randint(0, current_size - 1)
            z = current_set[z_idx]
            
            # Test removal
            new_set = current_set[:z_idx] + current_set[z_idx+1:]
            new_sums = _get_sums(new_set)
            
            if _is_maximal_fast(new_set, new_sums):
                # Accept
                current_set = new_set
                current_sums = new_sums
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                continue # Restart loop with new state
        
        # 2. Try to swap
        if current_size > 0:
            z_idx = random.randint(0, current_size - 1)
            z = current_set[z_idx]
            
            # Pick a random x not in current_set
            # To avoid bias, pick from range(1, N+1)
            x = random.randint(1, N)
            if x in current_set:
                continue
                
            # Test swap: remove z, add x
            # First, check if x can be added to (S \ {z})
            temp_set = current_set[:z_idx] + current_set[z_idx+1:]
            temp_sums = _get_sums(temp_set)
            
            if _can_add_with_sums(temp_set, temp_sums, x):
                # Form new set
                new_set = temp_set + [x]
                new_sums = temp_sums.union([x+s for s in temp_set] + [2*x])
                
                # Check if new set is maximal
                if _is_maximal_fast(new_set, new_sums):
                    current_set = new_set
                    current_sums = new_sums
                    # Size is same, so no best_size update unless we track "quality" differently
                    # But we want to minimize size, so swap is just for exploration
                    continue
        
        # 3. If stuck, try adding an element to escape local minimum?
        # Or restart?
        # Let's try adding a random element if it's Sidon-valid.
        # This increases size, but might allow subsequent removals.
        x = random.randint(1, N)
        if x not in current_set:
            if _can_add_with_sums(current_set, current_sums, x):
                new_set = current_set + [x]
                new_sums = current_sums.union([x+s for s in current_set] + [2*x])
                current_set = new_set
                current_sums = new_sums
                continue
        
        # If no move was possible (very unlikely with random x), restart
        # Just break and try a new seed
        break

    return best_set

# Run a few times to ensure we get a good result?
# The function construct() is called once.
# We should use the full time budget inside construct().

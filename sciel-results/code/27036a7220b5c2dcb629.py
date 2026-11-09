"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _is_valid_sidon(a):
    """Check if list a is a valid Sidon set (all pairwise sums a[i]+a[j] for i<=j are distinct)."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s = a[i] + a[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _pairwise_sums_set(a):
    """Return the set of all pairwise sums a[i]+a[j] for i<=j."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    return sums

def _try_add_element(a, sums, x):
    """Try to add x to Sidon set a with sum set sums. Return (success, new_sums)."""
    if x in a:
        return False, sums
    # Check all new sums: x + y for y in a, and 2*x
    new_sums = {x + y for y in a}
    new_sums.add(2 * x)
    if new_sums & sums:
        return False, sums
    return True, sums | new_sums

def _greedy_build(prescribed, max_val=284, rng=None):
    """Greedy construction starting from prescribed elements."""
    if rng is None:
        rng = random.Random()
    
    a = list(prescribed)
    a.sort()
    sums = _pairwise_sums_set(a)
    
    # Try adding elements in various orders
    candidates = list(range(12, max_val + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        ok, new_sums = _try_add_element(a, sums, x)
        if ok:
            a.append(x)
            sums = new_sums
    
    a.sort()
    return a

def _local_search(a, max_val=284, time_limit=30.0, rng=None):
    """Local search: try to improve the Sidon set by swapping, adding, or removing elements."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    a = list(a)
    a.sort()
    best_a = list(a)
    best_size = len(a)
    
    def _evaluate(arr):
        return len(arr)
    
    # Hill climbing with occasional random restarts within the neighborhood
    current_a = list(a)
    current_sums = _pairwise_sums_set(current_a)
    
    while time.time() - start_time < time_limit:
        # Try various moves
        improved = False
        
        # Move 1: Try adding a random element
        if rng.random() < 0.3:
            x = rng.randint(1, max_val)
            if x not in current_a:
                ok, new_sums = _try_add_element(current_a, current_sums, x)
                if ok:
                    current_a.append(x)
                    current_a.sort()
                    current_sums = new_sums
                    if len(current_a) > best_size:
                        best_a = list(current_a)
                        best_size = len(current_a)
                    improved = True
                    continue
        
        # Move 2: Try swapping an element with another
        if len(current_a) > 4 and rng.random() < 0.5:
            idx = rng.randint(0, len(current_a) - 1)
            # Don't swap prescribed elements
            if current_a[idx] not in (1, 2, 5, 11):
                old_x = current_a[idx]
                new_x = rng.randint(1, max_val)
                if new_x != old_x and new_x not in current_a:
                    # Try the swap
                    trial = current_a[:]
                    trial[idx] = new_x
                    trial.sort()
                    if _is_valid_sidon(trial):
                        current_a = trial
                        current_sums = _pairwise_sums_set(current_a)
                        if len(current_a) > best_size:
                            best_a = list(current_a)
                            best_size = len(current_a)
                        improved = True
                        continue
        
        # Move 3: Try removing a non-prescribed element and adding a different one
        if len(current_a) > 4 and rng.random() < 0.2:
            # Remove a random non-prescribed element
            non_prescribed = [i for i, x in enumerate(current_a) if x not in (1, 2, 5, 11)]
            if non_prescribed:
                idx = non_prescribed[rng.randint(0, len(non_prescribed) - 1)]
                removed = current_a[idx]
                trial = current_a[:]
                del trial[idx]
                trial.sort()
                trial_sums = _pairwise_sums_set(trial)
                
                # Try to add a new element
                added = False
                for _ in range(10):
                    new_x = rng.randint(1, max_val)
                    if new_x not in trial:
                        ok, new_trial_sums = _try_add_element(trial, trial_sums, new_x)
                        if ok:
                            trial.append(new_x)
                            trial.sort()
                            if len(trial) >= len(current_a):
                                current_a = trial
                                current_sums = new_trial_sums
                                if len(current_a) > best_size:
                                    best_a = list(current_a)
                                    best_size = len(current_a)
                                added = True
                            break
                if added:
                    improved = True
                    continue
        
        if not improved:
            # Stuck: try a more aggressive move - replace a random non-prescribed element
            if len(current_a) > 4:
                non_prescribed = [i for i, x in enumerate(current_a) if x not in (1, 2, 5, 11)]
                if non_prescribed:
                    idx = non_prescribed[rng.randint(0, len(non_prescribed) - 1)]
                    # Try replacing with several random values
                    for _ in range(20):
                        new_x = rng.randint(1, max_val)
                        if new_x not in current_a:
                            trial = current_a[:]
                            trial[idx] = new_x
                            trial.sort()
                            if _is_valid_sidon(trial):
                                current_a = trial
                                current_sums = _pairwise_sums_set(current_a)
                                if len(current_a) > best_size:
                                    best_a = list(current_a)
                                    best_size = len(current_a)
                                break
    
    return best_a

def construct():
    """Construct a large Sidon set in [1, 284] containing 1, 2, 5, 11."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    
    best_result = list(prescribed)
    best_size = len(prescribed)
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for seed in range(50):
        if time.time() > deadline - 20.0:
            break
        rng = random.Random(seed)
        a = _greedy_build(prescribed, max_val, rng)
        if len(a) > best_size:
            best_result = a
            best_size = len(a)
    
    # Phase 2: Local search on the best greedy result
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        improved = _local_search(best_result, max_val, time_limit=min(30.0, deadline - time.time() - 5.0), rng=rng)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
    
    # Phase 3: More local search iterations if time permits
    while time.time() < deadline - 2.0:
        rng = random.Random(int(time.time() * 1000) % 100000)
        improved = _local_search(best_result, max_val, time_limit=min(10.0, deadline - time.time() - 2.0), rng=rng)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
        else:
            # Try from a fresh greedy construction
            a = _greedy_build(prescribed, max_val, rng)
            if len(a) >= best_size:
                improved = _local_search(a, max_val, time_limit=min(10.0, deadline - time.time() - 2.0), rng=rng)
                if len(improved) > best_size:
                    best_result = improved
                    best_size = len(improved)
    
    return best_result
# EVOLVE-BLOCK-END

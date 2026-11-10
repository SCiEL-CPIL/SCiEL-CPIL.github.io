"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START

def _is_maximal(sidon_set, sums):
    """Check if the Sidon set is inclusion-maximal in [1, N]."""
    s = set(sidon_set)
    for x in range(1, N + 1):
        if x in s:
            continue
        # Try adding x: check all new sums x+a for a in sidon_set, and 2*x
        new_sums = [x + a for a in sidon_set] + [2 * x]
        # Check for duplicates within new_sums
        if len(set(new_sums)) != len(new_sums):
            continue
        # Check against existing sums
        if sums.intersection(new_sums):
            continue
        # If we get here, x can be added -> not maximal
        return False
    return True


def _greedy_construct(order, time_limit=55.0):
    """
    Greedily construct a maximal Sidon set by processing elements in `order`.
    Returns (sidon_list, sums_set) if maximal, else (None, None).
    """
    start_time = time.time()
    selected = []
    sums = set()
    for x in order:
        if time.time() - start_time > time_limit:
            return None, None
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    # Check maximality
    if _is_maximal(selected, sums):
        return selected, sums
    return None, None


def _remove_element(sidon_list, sums, x):
    """Remove x from sidon set, return new sums set."""
    new_sums = set()
    for a in sidon_list:
        if a == x:
            continue
        for b in sidon_list:
            if b == x or (a > b):
                continue
            if a == b:
                new_sums.add(2 * a)
            else:
                new_sums.add(a + b)
    # Actually, let's just recompute sums properly
    new_sums = set()
    remaining = [a for a in sidon_list if a != x]
    for i, a in enumerate(remaining):
        new_sums.add(2 * a)
        for b in remaining[i+1:]:
            new_sums.add(a + b)
    return new_sums


def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, safety margin before 60s limit
    
    # First try the ascending greedy to get a baseline
    best_set = None
    best_size = N + 1
    
    # Ascending greedy
    selected, sums = _greedy_construct(list(range(1, N + 1)), time_limit=1.0)
    if selected is not None:
        best_set = list(selected)
        best_size = len(selected)
    
    if time.time() - start_time > deadline - 5.0:
        return best_set if best_set else list(range(1, N + 1))
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() - start_time < deadline - 3.0:
        attempts += 1
        # Try different random orderings
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_construct(order, time_limit=1.0)
        if selected is not None and len(selected) < best_size:
            best_set = list(selected)
            best_size = len(selected)
        
        # Also try some structured orderings
        if attempts % 5 == 0:
            # Descending
            order = list(range(N, 0, -1))
            selected, sums = _greedy_construct(order, time_limit=1.0)
            if selected is not None and len(selected) < best_size:
                best_set = list(selected)
                best_size = len(selected)
        
        if attempts % 10 == 0:
            # Random with bias toward larger numbers first
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
            # Sort with random perturbation favoring larger
            order.sort(key=lambda x: -x + rng.uniform(-10, 10))
            selected, sums = _greedy_construct(order, time_limit=1.0)
            if selected is not None and len(selected) < best_size:
                best_set = list(selected)
                best_size = len(selected)
    
    # Local improvement: try removing elements from the best set
    if best_set is not None and time.time() - start_time < deadline - 1.0:
        current = list(best_set)
        current_sums = set()
        for i, a in enumerate(current):
            current_sums.add(2 * a)
            for b in current[i+1:]:
                current_sums.add(a + b)
        
        improved = True
        while improved and time.time() - start_time < deadline - 0.5:
            improved = False
            for idx in range(len(current) - 1, -1, -1):
                x = current[idx]
                # Try removing x
                new_current = current[:idx] + current[idx+1:]
                new_sums = set()
                for i, a in enumerate(new_current):
                    new_sums.add(2 * a)
                    for b in new_current[i+1:]:
                        new_sums.add(a + b)
                
                # Check if still maximal
                if _is_maximal(new_current, new_sums):
                    current = new_current
                    current_sums = new_sums
                    improved = True
                    break
                if time.time() - start_time > deadline - 0.5:
                    break
    
    if best_set is not None:
        # Use the potentially improved version
        if len(current) < len(best_set):
            best_set = current
    
    return best_set if best_set else list(range(1, N + 1))

# EVOLVE-BLOCK-END

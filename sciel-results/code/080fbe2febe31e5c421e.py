import time
import random
import numpy as np

def _compute_sum_counts(v, max_val=1000):
    """Compute the number of representations a+b=s with a<=b for each s.
    v is a boolean/int array of length max_val+1, v[i]=1 if i in set.
    Returns array count[s] = number of pairs (a,b) with a<=b, a+b=s, a,b in set.
    """
    # conv[s] = number of ordered pairs (a,b) with a+b=s
    conv = np.convolve(v, v[::-1], mode='full')
    # conv has length 2*max_val+1, conv[s] for s in 0..2*max_val
    # For s in 1..2*max_val-1, conv[s] gives ordered pair count
    # diag: for even s, if s/2 in set, then (s/2, s/2) is one ordered pair
    # count_a_le_b[s] = (conv[s] + (1 if s even and v[s//2] else 0)) // 2
    n = len(v)
    count = np.zeros(2 * max_val + 1, dtype=np.int32)
    for s in range(1, 2 * max_val + 1):
        ordered = conv[s] if s < len(conv) else 0
        if s % 2 == 0:
            half = s // 2
            diag = v[half] if 0 <= half < n else 0
        else:
            diag = 0
        count[s] = (ordered + diag) // 2
    return count

def _is_valid(v, max_val=1000):
    """Check if v represents a valid B2[2] set."""
    count = _compute_sum_counts(v, max_val)
    # We only care about s from 2 to 2*max_val
    return np.all(count[2:2*max_val+1] <= 2)

def _count_violations(v, max_val=1000):
    """Return number of sums with more than 2 representations, and max excess."""
    count = _compute_sum_counts(v, max_val)
    violations = count[2:2*max_val+1] - 2
    violations = np.maximum(violations, 0)
    return int(np.sum(violations)), int(np.max(violations)) if len(violations) > 0 else 0

def _try_add(v, x, max_val=1000):
    """Check if adding x to the set represented by v is valid.
    Returns True if valid, and the updated v if valid."""
    if v[x]:
        return False, v
    # Compute new sums: x + y for all y in set, and 2*x
    # We need to check if any sum would exceed 2
    # Get current counts
    count = _compute_sum_counts(v, max_val)
    # New sums to check: for each y in set, x+y; and 2*x
    # Find all y in set
    ys = np.where(v)[0]
    new_sums = ys + x
    # Also 2*x
    # Check each new sum
    for s in new_sums:
        if s > 2 * max_val:
            continue
        if count[s] >= 2:
            return False, v
    if 2 * x <= 2 * max_val:
        if count[2 * x] >= 2:
            return False, v
    # All good, add x
    v_new = v.copy()
    v_new[x] = 1
    return True, v_new

def _try_remove(v, x, max_val=1000):
    """Remove x from the set. Always valid after removal."""
    v_new = v.copy()
    v_new[x] = 0
    return v_new

def _local_search_dense(v_init, deadline, max_val=1000, max_iter=5000):
    """Local search starting from dense initialization.
    Iteratively try to fix violations by removing elements, then add back."""
    v = v_init.copy()
    current_size = int(np.sum(v))
    best_v = v.copy()
    best_size = current_size
    
    elements_in_set = list(np.where(v)[0])
    all_elements = list(range(1, max_val + 1))
    candidates = [x for x in all_elements if not v[x]]
    
    rng = random.Random(12345)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check current validity
        viol_count, max_excess = _count_violations(v, max_val)
        
        if viol_count == 0:
            # Valid! Try to add more elements
            if candidates:
                # Try adding random candidates
                for _ in range(min(10, len(candidates))):
                    x = rng.choice(candidates)
                    valid, v_new = _try_add(v, x, max_val)
                    if valid:
                        v = v_new
                        current_size += 1
                        elements_in_set.append(x)
                        candidates.remove(x)
                        if current_size > best_size:
                            best_size = current_size
                            best_v = v.copy()
                        break
            else:
                break
        else:
            # Has violations. Find the most violated sums and try removing elements
            count = _compute_sum_counts(v, max_val)
            # Find sums with count > 2
            violated_sums = np.where(count[2:2*max_val+1] > 2)[0] + 2
            if len(violated_sums) == 0:
                break
            
            # Pick a random violated sum
            s = rng.choice(violated_sums.tolist())
            
            # Find pairs (a,b) with a<=b, a+b=s, both in set
            pairs = []
            for a in range(1, s // 2 + 1):
                b = s - a
                if b > max_val or b < 1:
                    continue
                if v[a] and v[b]:
                    pairs.append((a, b))
            
            if not pairs:
                continue
            
            # Remove one of the elements involved in a violated pair
            # Try removing each element in the pairs and see if it helps
            candidates_to_remove = set()
            for a, b in pairs:
                candidates_to_remove.add(a)
                candidates_to_remove.add(b)
            
            if not candidates_to_remove:
                continue
            
            x = rng.choice(list(candidates_to_remove))
            
            # Remove x
            v[x] = 0
            current_size -= 1
            if x in elements_in_set:
                elements_in_set.remove(x)
            candidates.append(x)
            candidates.sort()
    
    return best_v, best_size

def _construct_greedy(order, deadline, max_val=1000):
    """Greedy construction."""
    v = np.zeros(max_val + 1, dtype=np.int8)
    chosen = []
    
    for x in order:
        if time.time() > deadline:
            break
        valid, v_new = _try_add(v, x, max_val)
        if valid:
            v = v_new
            chosen.append(x)
    
    return v, chosen

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    max_val = 1000
    
    best_v = None
    best_size = 0
    
    # Phase 1: Greedy constructions with different orders
    orders = [
        list(range(1, max_val + 1)),
        list(range(max_val, 0, -1)),
    ]
    
    rng = random.Random(42)
    for _ in range(5):
        orders.append(list(rng.sample(range(1, max_val + 1), max_val)))
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        v, chosen = _construct_greedy(order, deadline, max_val)
        size = len(chosen)
        if size > best_size:
            best_size = size
            best_v = v.copy()
    
    # Phase 2: Dense random initialization + local search
    # Try several dense random initializations
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random dense initialization: pick ~600-700 random elements
        density = 0.6 + 0.1 * (trial % 5)  # 60% to 100%
        n_init = int(max_val * density)
        init_set = set(rng.sample(range(1, max_val + 1), n_init))
        v_init = np.zeros(max_val + 1, dtype=np.int8)
        for x in init_set:
            v_init[x] = 1
        
        # Local search
        v_result, size = _local_search_dense(v_init, deadline, max_val, max_iter=3000)
        if size > best_size:
            best_size = size
            best_v = v_result.copy()
    
    # Phase 3: Refine the best solution with more local search
    if best_v is not None and time.time() < deadline - 5:
        v_result, size = _local_search_dense(best_v, deadline, max_val, max_iter=5000)
        if size > best_size:
            best_size = size
            best_v = v_result.copy()
    
    # Final: try to add more elements to the best solution
    if best_v is not None and time.time() < deadline - 2:
        candidates = [x for x in range(1, max_val + 1) if not best_v[x]]
        rng2 = random.Random(999)
        rng2.shuffle(candidates)
        v = best_v.copy()
        size = int(np.sum(v))
        for x in candidates:
            if time.time() > deadline - 1:
                break
            valid, v_new = _try_add(v, x, max_val)
            if valid:
                v = v_new
                size += 1
        if size > best_size:
            best_size = size
            best_v = v.copy()
    
    # Return the best set
    if best_v is None:
        return []
    
    result = list(np.where(best_v)[0])
    return result


"""Combined program: precomputed pair-sum conflicts + greedy/swap local search."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 128
DEADLINE = 82.0  # safety margin before 90s kill

def _precompute():
    """
    Precompute:
    - pair_sum_ok[a][b] = whether a+b <= N (for a < b)
    - For each pair (a,b) with a<b and a+b<=N, the bitmask of c > b such that
      a+c <= N and b+c <= N. If (a,b,a+b) are all in S, then any c in that mask
      creates a forbidden triple.
    """
    # forbidden_c[a][b] = bitmask of c values (c > b) that would create a violation
    # given a, b, a+b are all in S
    forbidden_c = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s > N:
                break
            # c must be > b, a+c <= N, b+c <= N
            max_c = min(N - a, N - b)
            if max_c <= b:
                continue
            mask = 0
            for c in range(b + 1, max_c + 1):
                mask |= (1 << c)
            if mask:
                forbidden_c[(a, b)] = mask
    return forbidden_c

def _check_valid(mask, forbidden_c):
    """Check if mask has no forbidden triple. Returns True if valid."""
    for (a, b), c_mask in forbidden_c.items():
        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> (a + b)) & 1):
            if (mask & c_mask) != 0:
                return False
    return True

def _greedy_add(mask, forbidden_c, candidates):
    """
    Try to add elements from candidates (shuffled) to mask in greedy fashion.
    Returns new_mask after adding all valid candidates.
    """
    new_mask = mask
    for x in candidates:
        if (new_mask >> x) & 1:
            continue
        trial = new_mask | (1 << x)
        if _check_valid(trial, forbidden_c):
            new_mask = trial
    return new_mask

def _try_add_single(mask, x, forbidden_c):
    """Check if adding x to mask keeps it valid. Return new mask or None."""
    if (mask >> x) & 1:
        return mask
    trial = mask | (1 << x)
    if _check_valid(trial, forbidden_c):
        return trial
    return None

def _try_swap(mask, x_remove, x_add, forbidden_c):
    """Check if removing x_remove and adding x_add keeps mask valid and same size."""
    base = mask & ~(1 << x_remove)
    trial = base | (1 << x_add)
    if _check_valid(trial, forbidden_c):
        return trial
    return None

def _try_add2_remove1(mask, x_remove, x_add1, x_add2, forbidden_c):
    """Remove one, add two. Size increases by 1."""
    base = mask & ~(1 << x_remove)
    trial = base | (1 << x_add1) | (1 << x_add2)
    if _check_valid(trial, forbidden_c):
        return trial
    return None

def construct():
    start_time = time.time()
    random.seed(42)
    
    forbidden_c = _precompute()
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Initial solution: upper half {65..128}
    mask = 0
    for i in range(65, N + 1):
        mask |= (1 << i)
    
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Phase 1: Greedy add from lower half
    lower_candidates = list(range(1, 65))
    random.shuffle(lower_candidates)
    mask = _greedy_add(mask, forbidden_c, lower_candidates)
    size = bin(mask).count('1')
    if size > best_size:
        best_mask = mask
        best_size = size
    
    # Phase 2: Local search with swaps and 2-for-1 moves
    T = 3.0
    T_min = 0.05
    cooling = 0.9997
    
    while T > T_min and time_ok():
        elems_in = [i for i in range(1, N + 1) if (mask >> i) & 1]
        elems_out = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
        
        if not elems_in:
            break
        
        move_type = random.random()
        new_mask = None
        delta_size = 0
        
        if move_type < 0.35 and elems_out:
            # Add a random element
            x = random.choice(elems_out)
            new_mask = _try_add_single(mask, x, forbidden_c)
            if new_mask is not None:
                delta_size = 1
            else:
                continue
        elif move_type < 0.65 and elems_in and elems_out:
            # Swap: remove one, add one
            x_rem = random.choice(elems_in)
            x_add = random.choice(elems_out)
            new_mask = _try_swap(mask, x_rem, x_add, forbidden_c)
            if new_mask is not None:
                delta_size = 0
            else:
                continue
        elif elems_in and len(elems_out) >= 2:
            # 2-for-1: remove one, add two (size +1)
            x_rem = random.choice(elems_in)
            x_adds = random.sample(elems_out, 2)
            new_mask = _try_add2_remove1(mask, x_rem, x_adds[0], x_adds[1], forbidden_c)
            if new_mask is not None:
                delta_size = 1
            else:
                # Try simple swap as fallback
                x_add = random.choice(elems_out)
                new_mask = _try_swap(mask, x_rem, x_add, forbidden_c)
                if new_mask is not None:
                    delta_size = 0
                else:
                    continue
        else:
            continue
        
        new_size = bin(new_mask).count('1')
        
        # Acceptance: prefer larger valid sets
        if new_size >= best_size:
            # Always accept if it improves or ties best
            mask = new_mask
            if new_size > best_size:
                best_mask = mask
                best_size = new_size
        else:
            # Simulated annealing acceptance for smaller sets
            delta_obj = (best_size - new_size)  # positive means worse
            if random.random() < math.exp(-delta_obj / T):
                mask = new_mask
        
        T *= cooling
    
    # Phase 3: Final greedy pass
    if time_ok():
        final_candidates = [i for i in range(1, N + 1) if not ((best_mask >> i) & 1)]
        random.shuffle(final_candidates)
        trial = _greedy_add(best_mask, forbidden_c, final_candidates)
        trial_size = bin(trial).count('1')
        if trial_size > best_size:
            best_mask = trial
            best_size = trial_size
    
    # Return best valid set
    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return list(result)

# EVOLVE-BLOCK-END

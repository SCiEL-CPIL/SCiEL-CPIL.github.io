import time
import random
import math
import numpy as np

# Precompute FFT setup for fast 3-AP difference counting
_MAX_VAL = 512
_NFFT = 1024  # next power of 2 >= 2*512+1
_FFT_SIZE = 2 * _NFFT  # for linear convolution via FFT

def _score_diffs_bitset(s):
    """
    Fast count of distinct d>0 where a, a+d, a+2d all in s.
    Uses FFT convolution: (indicator * indicator)[2d] counts pairs (a, a+2d) both in s.
    Then a 3-AP with difference d exists iff there's some a with a, a+d, a+2d in s.
    We compute: for each d, check if sum over a of I(a)*I(a+d)*I(a+2d) > 0.
    
    Actually, the number of 3-APs with difference d is:
      sum_a I(a) * I(a+d) * I(a+2d)
    
    We can compute this via: let c = I * I (convolution). Then c[2d] = sum_a I(a)*I(a+2d).
    But that's not quite what we need. We need sum_a I(a)*I(a+d)*I(a+2d).
    
    Alternative: For each d, the count is the inner product of I shifted by 0, d, 2d.
    
    Better approach using FFT:
    Let f = indicator vector of S (length _NFFT, with f[x]=1 if x in S).
    Compute F = FFT(f).
    Then the 3-AP count for difference d is:
      sum_a f[a] * f[a+d] * f[a+2d]
    
    This can be computed as: let g_d[a] = f[a] * f[a+d] * f[a+2d].
    Sum over a.
    
    For all d simultaneously, we can use:
    h = IFFT(|F|^2) gives the autocorrelation: h[k] = sum_a f[a]*f[a+k] (circular).
    But we need triple product, not double.
    
    Actually, let me think differently. For a 3-AP (a, a+d, a+2d), all three must be in S.
    The number of such triples with difference d is:
      T(d) = sum_a f[a] * f[a+d] * f[a+2d]
    
    We can compute T(d) for all d using the following trick:
    Let's define for each element x in S, and each other element y in S with y > x,
    the midpoint m = (x+y)/2. If m is also in S and (y-x) is even, then we have a 3-AP.
    
    But for counting distinct d values, we just need T(d) > 0.
    
    Fast method: Use the fact that T(d) = sum_a f[a]*f[a+d]*f[a+2d].
    We can compute this for all d at once using FFT if we're clever, but it's a triple product.
    
    Simpler fast method for 32 elements: 
    For each pair (x, y) in S with y > x, let d = (y - x) // 2 if (y-x) even, else skip.
    Actually no, that's for the midpoint.
    
    Let me just use a bitset approach that's fast enough for 32 elements:
    For each d from 1 to 255:
      Check if there exists a such that a, a+d, a+2d all in S.
    
    With 32 elements, we can do this very fast by iterating over pairs.
    """
    s_set = set(s)
    # For each d, we need to check if there's a 3-AP.
    # A 3-AP (a, a+d, a+2d) requires a, a+d, a+2d all in s_set.
    # Equivalently: for some a, a in s_set, a+d in s_set, a+2d in s_set.
    # 
    # Fast approach: build a bitset, then for each d, check (bits & (bits >> d) & (bits >> 2*d)) != 0
    max_val = max(s_set) if s_set else 0
    if max_val < 3:
        return 0
    
    # Build bitset as Python int
    bits = 0
    for x in s_set:
        bits |= (1 << x)
    
    count = 0
    # d can range from 1 to (max_val - 1) // 2
    max_d = (max_val - 1) // 2
    for d in range(1, max_d + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    
    return count


def _score_diffs_fft(s):
    """
    Alternative: use numpy for potentially faster scoring.
    For 32 elements in 1..512, the bitset approach is already very fast.
    Let's benchmark and use whichever is faster.
    """
    s_set = set(s)
    if len(s_set) < 3:
        return 0
    
    max_val = max(s_set)
    if max_val < 3:
        return 0
    
    # Create indicator array
    f = np.zeros(_NFFT, dtype=np.float64)
    for x in s_set:
        if 1 <= x <= _NFFT - 1:
            f[x] = 1.0
    
    # Compute autocorrelation via FFT
    F = np.fft.rfft(f)
    # Autocorrelation: sum_a f[a]*f[a+k]
    # We need triple product, so let's use a different approach.
    # 
    # Actually, for the 3-AP count with difference d:
    # T(d) = sum_a f[a] * f[a+d] * f[a+2d]
    # 
    # This is a convolution-like operation but with three shifts.
    # Let's just use the bitset method which is proven fast for small sets.
    # 
    # For now, fall back to bitset
    return _score_diffs_bitset(s)


def _score(s):
    """Main scoring function - uses the fastest method."""
    return _score_diffs_bitset(s)


def _initial_sets():
    """Generate diverse initial sets for local search."""
    sets = []
    
    # Dense interval
    sets.append(list(range(1, 33)))
    
    # Two blocks
    sets.append(list(range(1, 17)) + list(range(257, 273)))
    
    # Four blocks
    sets.append(list(range(1, 9)) + list(range(129, 137)) + list(range(257, 265)) + list(range(385, 393)))
    
    # Eight blocks
    sets.append(list(range(1, 5)) + list(range(65, 69)) + list(range(129, 133)) + list(range(193, 197)) +
                list(range(257, 261)) + list(range(321, 325)) + list(range(385, 389)) + list(range(449, 453)))
    
    # Structured: multiples
    sets.append(list(range(1, 513, 16))[:32])
    
    # Structured: powers and multiples
    base = list(range(1, 17))
    sets.append(base + [x * 2 for x in base if x * 2 <= 512][:16])
    
    # Random sets
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        rng = random.Random(seed)
        s = set()
        while len(s) < 32:
            s.add(rng.randint(1, 512))
        sets.append(list(s))
    
    return sets


def _local_search(initial_set, deadline, rng):
    """
    Local search with multiple move types and adaptive temperature.
    Returns (best_set, best_score).
    """
    current = set(initial_set)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    T = 3.0
    T_min = 0.05
    cooling = 0.9998
    
    # Precompute all possible elements
    all_elements = list(range(1, 513))
    
    # For faster sampling, maintain a list of current elements
    current_list = list(current)
    
    while time.time() < deadline and T > T_min:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Single element swap
            idx = rng.randrange(len(current_list))
            old_elem = current_list[idx]
            
            # Try a few random new elements
            for _ in range(3):
                new_elem = rng.choice(all_elements)
                if new_elem == old_elem:
                    continue
                
                # Quick check: will this help?
                # Remove old, add new
                current.remove(old_elem)
                current_list[idx] = new_elem
                current.add(new_elem)
                
                new_score = _score(current)
                
                if new_score >= current_score or rng.random() < math.exp(-(current_score - new_score) / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = list(current)
                    break
                else:
                    # Revert
                    current.remove(new_elem)
                    current.add(old_elem)
                    current_list[idx] = old_elem
            
        elif move_type < 0.7:
            # Double swap
            if len(current_list) >= 2:
                i, j = rng.sample(range(len(current_list)), 2)
                old1, old2 = current_list[i], current_list[j]
                
                for _ in range(3):
                    new1 = rng.choice(all_elements)
                    new2 = rng.choice(all_elements)
                    if new1 == old1 or new1 == old2 or new2 == old1 or new2 == old2:
                        continue
                    if new1 == new2:
                        continue
                    
                    current.remove(old1)
                    current.remove(old2)
                    current_list[i] = new1
                    current_list[j] = new2
                    current.add(new1)
                    current.add(new2)
                    
                    new_score = _score(current)
                    
                    if new_score >= current_score or rng.random() < math.exp(-(current_score - new_score) / T):
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best = list(current)
                        break
                    else:
                        # Revert
                        current.remove(new1)
                        current.remove(new2)
                        current.add(old1)
                        current.add(old2)
                        current_list[i] = old1
                        current_list[j] = old2
                        
        else:
            # Perturbation: replace 3-4 random elements
            n_replace = rng.choice([3, 4])
            if len(current_list) >= n_replace:
                indices = rng.sample(range(len(current_list)), n_replace)
                old_elems = [current_list[i] for i in indices]
                
                current.difference_update(old_elems)
                for i, idx in enumerate(indices):
                    current_list[idx] = None  # placeholder
                
                for _ in range(5):
                    new_elems = []
                    valid = True
                    for _ in range(n_replace):
                        ne = rng.choice(all_elements)
                        if ne in current or ne in new_elems:
                            valid = False
                            break
                        new_elems.append(ne)
                    
                    if not valid:
                        continue
                    
                    current.update(new_elems)
                    for i, idx in enumerate(indices):
                        current_list[idx] = new_elems[i]
                    
                    new_score = _score(current)
                    
                    if new_score >= current_score or rng.random() < math.exp(-(current_score - new_score) / T):
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best = list(current)
                        break
                    else:
                        # Revert
                        current.difference_update(new_elems)
                        current.update(old_elems)
                        for i, idx in enumerate(indices):
                            current_list[idx] = old_elems[i]
        
        T *= cooling
    
    return best, best_score


def _greedy_construct():
    """
    Greedy construction: start with a base structure and add elements
    that maximize the number of new 3-AP differences.
    """
    # Start with a well-structured base
    # Use a Sidon-like or structured base
    base = list(range(1, 17))  # 1..16
    s = set(base)
    
    # Greedily add elements that create the most new 3-AP differences
    all_elements = list(range(17, 513))
    rng = random.Random(42)
    rng.shuffle(all_elements)
    
    current_score = _score(s)
    
    for elem in all_elements:
        if len(s) >= 32:
            break
        
        # Try adding this element
        s.add(elem)
        new_score = _score(s)
        
        if new_score > current_score:
            current_score = new_score
        else:
            s.remove(elem)
    
    # If we have more than 32, trim
    if len(s) > 32:
        s = set(list(s)[:32])
    
    # If we have fewer than 32, fill randomly
    while len(s) < 32:
        x = rng.randint(1, 512)
        if x not in s:
            s.add(x)
    
    return list(s)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_set = None
    best_score = -1
    
    # Phase 1: Try multiple initializations with local search
    inits = _initial_sets()
    
    # Also add greedy constructions
    greedy_set = _greedy_construct()
    inits.append(greedy_set)
    
    # Split time among initializations
    n_inits = len(inits)
    time_per_init = (deadline - time.time()) / max(n_inits, 1)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 2:
            break
        
        # Ensure valid set
        init_set = set(init)
        if len(init_set) != 32:
            init_set = set(list(init_set)[:32])
            while len(init_set) < 32:
                init_set.add(random.randint(1, 512))
        
        # Local search with a portion of remaining time
        remaining = deadline - time.time() - 2
        alloc = min(time_per_init, remaining * 0.5)
        if alloc < 1:
            alloc = max(remaining * 0.1, 0.5)
        
        rng = random.Random(1000 + i)
        ls_set, ls_score = _local_search(list(init_set), time.time() + alloc, rng)
        
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 2: If time remains, do more local search on the best found
    if best_set is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            rng = random.Random(9999)
            ls_set, ls_score = _local_search(best_set, time.time() + remaining, rng)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
    
    # Phase 3: Final polish - try specific structured improvements
    if best_set is not None and time.time() < deadline - 1:
        # Try replacing worst elements with structured candidates
        s = set(best_set)
        current_score = _score(s)
        
        # Candidates that often help: small numbers, multiples
        candidates = list(range(1, 33)) + list(range(1, 513, 16)) + list(range(1, 513, 32))
        candidates = [x for x in candidates if x not in s]
        
        for c in candidates:
            if time.time() > deadline - 1:
                break
            if len(s) >= 32:
                # Try swapping: remove the element whose removal hurts least
                best_swap = None
                best_swap_score = -1
                for elem in list(s):
                    s.remove(elem)
                    s.add(c)
                    sc = _score(s)
                    if sc > best_swap_score:
                        best_swap_score = sc
                        best_swap = elem
                    s.remove(c)
                    s.add(elem)
                
                if best_swap is not None and best_swap_score > current_score:
                    s.remove(best_swap)
                    s.add(c)
                    current_score = best_swap_score
                    best_set = list(s)
                    best_score = max(best_score, current_score)
            else:
                s.add(c)
                sc = _score(s)
                if sc > current_score:
                    current_score = sc
                    best_set = list(s)
                    best_score = max(best_score, current_score)
                else:
                    s.remove(c)
    
    if best_set is None:
        return list(range(1, 33))
    
    # Ensure exactly 32 distinct integers in 1..512
    result = list(set(best_set))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
    
    return result


# Quick self-test
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    score = _score(result)
    print(f"Result: {sorted(result)}")
    print(f"Score: {score}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Count: {len(result)}")
    print(f"Distinct: {len(set(result))}")
    print(f"Range ok: {all(1 <= x <= 512 for x in result)}")

"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-term AP differences."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in set s."""
    s_set = set(s)
    diffs = set()
    # For each pair (a, b) with a < b, check if a+2*(b-a) = a + 2*(b-a) is in set
    # Actually: for 3-term AP a, a+d, a+2d: we need all three in set.
    # d = b - a where b = a + d, and a + 2d = a + 2*(b-a) must be in set.
    # Iterate over all pairs as potential (first, second) terms.
    arr = sorted(s_set)
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[j] - arr[i]
            third = arr[i] + 2 * d
            if third in s_set:
                diffs.add(d)
    return len(diffs)


def _score_fast(s_set):
    """Faster scoring using set membership."""
    diffs = set()
    arr = sorted(s_set)
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[j] - arr[i]
            if arr[i] + 2 * d in s_set:
                diffs.add(d)
    return len(diffs)


def _greedy_init():
    """Density-aware greedy initialization."""
    # Start with a spread-out set, then greedily add elements that maximize score
    # Use a base structure: try to include many small differences
    # Strategy: start with multiples of small numbers, then refine
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    # Greedy: build set by adding element that gives best marginal gain
    s = set()
    # Seed with a few well-placed elements
    # Try to get a good starting point: use arithmetic progression base + perturbations
    base = []
    # Use a mix of small and medium values
    for _ in range(32):
        best_elem = None
        best_score = -1
        for c in candidates:
            if c in s:
                continue
            s.add(c)
            sc = _score_set(s)
            s.remove(c)
            if sc > best_score:
                best_score = sc
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            candidates.remove(best_elem)
        else:
            # Fallback
            for c in candidates:
                s.add(c)
                break
            if candidates:
                candidates.pop(0)
    
    return list(s)


def _triple_swap(s_list, rng):
    """Perform a triple swap: remove 3 elements, add 3 new ones."""
    s_set = set(s_list)
    # Remove 3 random elements
    removed = rng.sample(s_list, 3)
    for r in removed:
        s_set.discard(r)
    
    # Add 3 random new elements not in set
    available = [x for x in range(1, 513) if x not in s_set]
    if len(available) < 3:
        return None
    added = rng.sample(available, 3)
    for a in added:
        s_set.add(a)
    
    return list(s_set)


def _double_swap(s_list, rng):
    """Remove 2, add 2."""
    s_set = set(s_list)
    removed = rng.sample(s_list, 2)
    for r in removed:
        s_set.discard(r)
    
    available = [x for x in range(1, 513) if x not in s_set]
    if len(available) < 2:
        return None
    added = rng.sample(available, 2)
    for a in added:
        s_set.add(a)
    
    return list(s_set)


def _single_swap(s_list, rng):
    """Remove 1, add 1."""
    s_set = set(s_list)
    removed = rng.choice(s_list)
    s_set.discard(removed)
    
    available = [x for x in range(1, 513) if x not in s_set]
    if not available:
        return None
    added = rng.choice(available)
    s_set.add(added)
    
    return list(s_set)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_set = None
    best_score = -1
    
    # Multi-restart with different strategies
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Greedy density-based init
            current = _greedy_init()
        elif restart == 1:
            # Start from simple AP with perturbations
            current = list(range(1, 33))
            for _ in range(10):
                current = _triple_swap(current, rng)
        elif restart == 2:
            # Spread out initialization
            current = [1 + i * 16 for i in range(32)]
            for _ in range(5):
                current = _triple_swap(current, rng)
        elif restart == 3:
            # Random with density bias toward small numbers
            current = []
            for i in range(32):
                # Bias toward smaller numbers for more AP opportunities
                val = 1 + int(rng.random() ** 1.5 * 511)
                while val in current:
                    val = 1 + int(rng.random() ** 1.5 * 511)
                current.append(val)
        elif restart == 4:
            # Use structure: multiples of 1,2,3,4,5,6,7,8 in ranges
            current = []
            for mult in range(1, 9):
                # Pick 4 numbers that are multiples or near multiples
                base = mult * 10
                for k in range(4):
                    val = base + k * mult + rng.randint(-2, 2)
                    val = max(1, min(512, val))
                    if val not in current:
                        current.append(val)
                    else:
                        # Find another
                        for v in range(1, 513):
                            if v not in current:
                                current.append(v)
                                break
            # Ensure exactly 32
            while len(current) < 32:
                v = rng.randint(1, 512)
                if v not in current:
                    current.append(v)
            current = current[:32]
        else:
            # Fully random
            current = rng.sample(range(1, 513), 32)
        
        current_score = _score_set(current)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
        
        # Simulated annealing
        temp = 10.0
        cooling = 0.999
        min_temp = 0.01
        
        while temp > min_temp and time.time() < deadline - 10:
            # Choose move type
            move_type = rng.random()
            if move_type < 0.4:
                new_set = _triple_swap(current, rng)
            elif move_type < 0.7:
                new_set = _double_swap(current, rng)
            else:
                new_set = _single_swap(current, rng)
            
            if new_set is None:
                temp *= cooling
                continue
            
            new_score = _score_set(new_set)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < pow(2.718281828, delta / temp):
                current = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
            
            temp *= cooling
        
        # Local refinement: try all single swaps from best
        if best_set and time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(32):
                    if time.time() > deadline - 5:
                        break
                    old_val = best_set[i]
                    for new_val in range(1, 513):
                        if new_val == old_val or new_val in best_set:
                            continue
                        trial = best_set[:]
                        trial[i] = new_val
                        sc = _score_set(trial)
                        if sc > best_score:
                            best_score = sc
                            best_set = trial
                            improved = True
                            break
    
    if best_set is None:
        return list(range(1, 33))
    
    return best_set

# EVOLVE-BLOCK-END

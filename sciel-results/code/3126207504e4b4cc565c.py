import time
import random
import itertools

_DEADLINE = 85.0  # safety margin before 90s kill

def _is_sidon4(a, b, c, d):
    """Check if 4 distinct elements form a Sidon set (all pair sums incl. diagonals unique)."""
    # Pair sums including diagonals: 2a,2b,2c,2d,a+b,a+c,a+d,b+c,b+d,c+d
    s = (2*a, 2*b, 2*c, 2*d,
         a+b, a+c, a+d,
         b+c, b+d, c+d)
    return len(set(s)) == 10

def _count_sidon4_subsets(arr):
    """Count how many 4-element subsets of arr are Sidon."""
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _is_sidon4(arr[i], arr[j], arr[k], arr[l]):
                        count += 1
    return count

def _max_sidon_size(arr):
    """Find the maximum size of a Sidon subset of arr."""
    n = len(arr)
    # Check if full set is Sidon
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try size n
    if is_sidon(arr):
        return n
    # Try size n-1
    for omit in range(n):
        sub = [arr[i] for i in range(n) if i != omit]
        if is_sidon(sub):
            return n - 1
    # Try size n-2
    for omit1 in range(n):
        for omit2 in range(omit1+1, n):
            sub = [arr[i] for i in range(n) if i != omit1 and i != omit2]
            if is_sidon(sub):
                return n - 2
    # For our purposes, we know it's at most 4 for good candidates
    # Check size 4
    for quad in itertools.combinations(arr, 4):
        if is_sidon(list(quad)):
            return 4
    return 3

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_arr = list(range(12))
    best_score = _count_sidon4_subsets(best_arr)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    current = best_arr[:]
    current_score = best_score
    
    while time.time() - start_time < _DEADLINE:
        T *= cooling
        if T < T_min:
            break
        
        # Random move: pick two random positions and swap or perturb
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            new_arr = current[:]
            new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
        elif move_type < 0.8:
            # Perturb one element
            i = random.randint(0, 11)
            delta = random.choice([-2, -1, 1, 2, -3, 3])
            new_val = current[i] + delta
            if 0 <= new_val <= 128:
                new_arr = current[:]
                new_arr[i] = new_val
                # Check distinctness
                if len(set(new_arr)) != 12:
                    continue
            else:
                continue
        else:
            # Random restart with small perturbation
            new_arr = current[:]
            for _ in range(random.randint(1, 3)):
                i = random.randint(0, 11)
                delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                new_val = current[i] + delta
                if 0 <= new_val <= 128:
                    new_arr[i] = new_val
            if len(set(new_arr)) != 12:
                continue
        
        new_score = _count_sidon4_subsets(new_arr)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current = new_arr
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_arr = current[:]
    
    return best_arr

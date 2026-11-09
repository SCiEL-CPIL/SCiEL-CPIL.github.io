"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _objective_np(A):
    """
    Compute max(|A+A|, |A*A|) using NumPy for speed.
    A is a list of 20 distinct positive integers.
    """
    A_np = np.array(A, dtype=np.int64)
    # A+A: all pairs (i, j) with i <= j
    # We can compute all sums and take unique, but we must ensure we don't double count or miss constraints?
    # Actually, |A+A| is just the number of unique sums. Since addition is commutative, 
    # computing the full outer sum and taking unique counts each sum once.
    # However, the problem says "equal summands allowed", which is just standard A+A.
    # The size of the set of sums {a+b | a,b in A} is what matters.
    sums = A_np[:, None] + A_np[None, :]
    sum_size = len(np.unique(sums))
    
    prods = A_np[:, None] * A_np[None, :]
    prod_size = len(np.unique(prods))
    
    return max(sum_size, prod_size)

def _is_valid(A):
    """Check 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_smooth_candidates():
    """
    Generate candidates based on smooth numbers (2^i * 3^j * 5^k).
    These sets tend to have high collision rates in products.
    """
    candidates = []
    # Generate first 100 smooth numbers (5-smooth)
    smooths = set()
    for i in range(15):
        for j in range(12):
            for k in range(10):
                val = (2**i) * (3**j) * (5**k)
                if val <= 10000:
                    smooths.add(val)
    smooths_list = sorted(smooths)
    
    # Take the first 20
    if len(smooths_list) >= 20:
        candidates.append(smooths_list[:20])
        
    # Try 3-smooth numbers
    smooths_3 = set()
    for i in range(15):
        for j in range(15):
            val = (2**i) * (3**j)
            if val <= 10000:
                smooths_3.add(val)
    smooths_3_list = sorted(smooths_3)
    if len(smooths_3_list) >= 20:
        candidates.append(smooths_3_list[:20])
        
    return candidates

def _generate_structured_candidates():
    """
    Generate initial candidates from structured sets to exploit collisions.
    """
    candidates = []
    
    # 1. Geometric Progressions: {a * r^i}
    # Small r (2, 3) keeps numbers small.
    for a in [1, 2, 3, 5, 10]:
        for r in [2, 3, 4]:
            seq = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                seq.append(val)
                val *= r
            if len(seq) == 20:
                candidates.append(sorted(seq))
                
    # 2. Arithmetic Progressions: {a + i*d}
    # Small d keeps sum set small (size ~2n). Product set is larger.
    for a in [1, 2, 3]:
        for d in [1, 2, 3, 4, 5]:
            seq = [a + i*d for i in range(20)]
            if max(seq) <= 10000:
                candidates.append(seq)
                
    # 3. Mixed: First half geometric, second half arithmetic? Or just random small integers.
    # Let's add some random small integer sets as a baseline.
    for _ in range(5):
        # Sample 20 distinct ints from 1..100
        base = random.sample(range(1, 101), 20)
        candidates.append(sorted(base))
        
    # 4. "Prime-like" or "Smooth" numbers?
    # Just adding a few random geometric-like sets with start > 1
    for _ in range(5):
        a = random.randint(1, 10)
        r = random.choice([2, 3])
        seq = []
        val = a
        for _ in range(20):
            if val > 10000:
                # If we run out, fill with next smallest available? 
                # Or just break and discard if not 20.
                break
            seq.append(val)
            val *= r
        if len(seq) == 20:
             candidates.append(sorted(seq))
             
    # 5. Smooth number candidates
    candidates.extend(_generate_smooth_candidates())
    
    # 6. Greedy construction minimizing product set
    # Start with 1, then pick next number that minimizes |A*A|
    A_greedy = [1]
    for _ in range(19):
        best_val = None
        best_cost = float('inf')
        # Try candidates from 2 to 100
        for val in range(2, 101):
            if val in A_greedy:
                continue
            new_A = A_greedy + [val]
            # Approximate cost: just product size for speed in generation
            prods = set()
            for i in range(len(new_A)):
                for j in range(i, len(new_A)):
                    prods.add(new_A[i] * new_A[j])
            if len(prods) < best_cost:
                best_cost = len(prods)
                best_val = val
        if best_val:
            A_greedy.append(best_val)
        else:
            break
    if len(A_greedy) == 20:
        candidates.append(sorted(A_greedy))

    return candidates

def _nudge_move(A):
    """Nudge one element by a small amount."""
    idx = random.randrange(20)
    # Bias towards small changes
    delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
    new_val = A[idx] + delta
    if new_val < 1 or new_val > 10000:
        return None
    if new_val in A:
        return None
    A[idx] = new_val
    return sorted(A)

def _swap_move(A):
    """Swap one element with a random new value, biased towards small numbers."""
    idx = random.randrange(20)
    old = A[idx]
    # Bias towards small numbers as they create more collisions in sums
    while True:
        if random.random() < 0.7:
            new = random.randint(1, 50)
        else:
            new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves using vectorized objective."""
    current = list(A)
    current_cost = _objective_np(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        # Choose move type
        r = random.random()
        if r < 0.6:
            new_A = _nudge_move(list(current))
            if new_A is None:
                continue
        else:
            new_A = _swap_move(list(current))
        
        new_cost = _objective_np(new_A)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
        else:
            # Accept if exp(-delta/temp) > random
            if temp > 1e-9:
                prob = math.exp(-delta / temp)
            else:
                prob = 0.0
            accept = random.random() < prob
            
        if accept:
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        # Cool down
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 40s safety margin for 90s limit
    
    random.seed(42)
    
    # Phase 1: Generate structured initial solutions
    best_set = None
    best_cost = float('inf')
    
    candidates = _generate_structured_candidates()
    for A in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(A):
            continue
        cost = _objective_np(A)
        if cost < best_cost:
            best_cost = cost
            best_set = list(A)
            
    # If no valid candidate found (unlikely), fallback to simple range
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective_np(best_set)

    # Phase 2: Simulated annealing
    # Run multiple restarts if time permits
    # Since vectorized objective is fast, we can do more steps
    start_temp = 10.0
    end_temp = 0.01
    steps = 5000 # Increased steps due to speedup
    
    # We will run annealing from the best found, and maybe a few others if time allows
    # For simplicity and robustness, just run from the best one for now, 
    # but with a high step count.
    
    result, cost = _anneal(best_set, start_temp, end_temp, steps, deadline)
    if cost < best_cost:
        best_cost = cost
        best_set = list(result)
        
    # Phase 3: Local refinement (Hill Climbing)
    # Try to improve by single element changes
    if time.time() < deadline - 5:
        current = list(best_set)
        current_cost = best_cost
        improved = True
        # Limit iterations to avoid infinite loops
        max_iters = 1000
        iter_count = 0
        while improved and time.time() < deadline - 2 and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Shuffle indices to avoid bias
            indices = list(range(20))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline - 2:
                    break
                old = current[idx]
                # Try a few nearby values
                # Use a wider range of deltas
                deltas = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, -20]
                random.shuffle(deltas)
                for delta in deltas:
                    new_val = old + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current:
                        continue
                    current[idx] = new_val
                    new_cost = _objective_np(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_set = list(current)
                        break # Move to next index
                    else:
                        current[idx] = old
                
                # Also try a random swap with a small number
                if random.random() < 0.5:
                    new_val = random.randint(1, 100)
                    if new_val not in current and new_val != old:
                        current[idx] = new_val
                        new_cost = _objective_np(current)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_set = list(current)
                        else:
                            current[idx] = old

    return best_set

# EVOLVE-BLOCK-END

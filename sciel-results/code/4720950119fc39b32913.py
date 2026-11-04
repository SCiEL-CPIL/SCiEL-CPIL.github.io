import time
import math
import random
import numpy as np

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    
    # For small n, NumPy broadcasting is very fast
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    # Count unique elements using sort (faster than np.unique for small arrays)
    n_sums = len(np.unique(sums.ravel()))
    n_diffs = len(np.unique(diffs.ravel()))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def local_search(U, deadline, max_steps=500, step_range=5):
    """
    Local search to improve the score of U.
    Tries to move each element by a small amount.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        indices = list(range(1, n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            low = max(0, old_val - step_range)
            high = old_val + step_range
            
            best_new_val = old_val
            best_new_score = current_score
            
            candidates = set()
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                for _ in range(10):
                    candidates.add(random.randint(low, high))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < 0:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                
                s = fast_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val != old_val:
                U[i] = best_new_val
                U = sorted(U)
                current_score = best_new_score
                improved = True
        
        if not improved:
            step_range += 2
            if step_range > 20:
                break
                
    return U, current_score

def generate_random_set(n, max_val):
    """Generate a random set of size n containing 0."""
    if n == 1:
        return [0]
    if max_val < n:
        return list(range(n))
        
    S = {0}
    while len(S) < n:
        x = random.randint(1, max_val)
        S.add(x)
    return sorted(S)

def generate_geometric_set(m, b):
    """Generate {0, 1, b, b^2, ...}"""
    U = [0, 1]
    cur = 1
    for i in range(m-1):
        cur = int(cur * b)
        if cur in U:
            cur += 1
            while cur in U:
                cur += 1
        U.append(cur)
    return sorted(set(U))

def generate_interval_sequence(m, alpha):
    """
    Generate a set U as a sequence of intervals with exponentially growing gaps.
    U starts at 0. Each subsequent interval [start, start + width - 1] has
    start = prev_end + gap, where gap grows by factor alpha.
    Width of each interval also grows.
    """
    if m < 1 or alpha <= 1.0:
        return [0]
    
    U = [0]
    current_max = 0
    
    for i in range(1, m):
        # Gap grows exponentially
        gap = int((current_max + 1) * (alpha - 1.0) + 1)
        if gap < 1:
            gap = 1
        
        start = current_max + gap
        
        # Width grows linearly with i or exponentially
        width = max(1, int(i * 1.5))
        
        # Add the interval
        for j in range(width):
            val = start + j
            if val not in U:
                U.append(val)
        
        current_max = start + width - 1
    
    return sorted(set(U))

def construct():
    start = time.time()
    # Evaluator kills at 150s. Keep margin.
    deadline = start + 115
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Explore small sets with local search
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        strategy = random.random()
        
        if strategy < 0.3:
            n = random.randint(3, 8)
            max_val = random.randint(10, 100)
            U = generate_random_set(n, max_val)
        elif strategy < 0.5:
            n = random.randint(3, 7)
            b = random.uniform(1.5, 3.0)
            U = generate_geometric_set(n, b)
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0: U[i] = 0
            U = sorted(set(U))
        elif strategy < 0.7:
            # Interval sequence with tunable alpha
            m = random.randint(2, 6)
            alpha = random.uniform(1.5, 4.0)
            U = generate_interval_sequence(m, alpha)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.2:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0: U[i] = 0
            U = sorted(set(U))
        else:
            # Golomb-like
            n = random.randint(3, 8)
            U = [0, 1]
            for i in range(1, n-1):
                if len(U) >= 2:
                    idx = max(0, len(U) - 1 - U[-1])
                    new_val = U[-1] + U[idx] + 1
                else:
                    new_val = U[-1] + 1
                if new_val not in U:
                    U.append(new_val)
        
        U, score = local_search(U, deadline, max_steps=100, step_range=5)
        
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Phase 2: Simulated Annealing on the best set found
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.03
    T_min = 0.0005
    cooling = 0.998
    
    max_size = 15
    
    for iteration in range(50000):
        if time.time() > deadline - 2:
            break
            
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.45 and n > 2:
            idx = random.randint(1, n-1)
            old_val = U[idx]
            sigma = max(1, old_val * 0.15)
            new_val = max(0, int(old_val + random.gauss(0, sigma)))
            if new_val == old_val:
                continue
            new_U = U.copy()
            new_U[idx] = new_val
            if len(set(new_U)) != len(new_U):
                continue
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.75 and n < max_size:
            max_u = int(U.max())
            if random.random() < 0.3:
                bit_pos = random.randint(1, min(15, max_u.bit_length() + 2))
                new_val = 1 << bit_pos
                if random.random() < 0.5:
                    new_val |= (1 << max(0, bit_pos - 1))
            else:
                new_val = random.randint(1, max_u + 25)
            if new_val in U:
                continue
            new_U = np.append(U, new_val)
            new_U = np.sort(new_U)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        else:
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
                        
        T *= cooling

    # Final Polish
    U = list(best_set)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        n = len(U)
        for i in range(1, n):
            old_val = U[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if improved:
            best_set = U[:]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(compute_score(res))

import time
import random
import numpy as np
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (with repetition)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    # Difference set: all pairwise differences (with sign)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Vectorized scoring for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_structural_seeds():
    """Generate diverse structural seeds."""
    seeds = []
    
    # Powers of 2
    for k in range(3, 15):
        seeds.append([0] + [2**i for i in range(1, k)])
    
    # Powers of 3
    for k in range(3, 10):
        seeds.append([0] + [3**i for i in range(1, k)])
    
    # Geometric with base ~1.5-2.5
    for base in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5]:
        U = [0]
        val = 1
        seen = {0}
        for _ in range(20):
            val = int(round(val * base))
            if val <= 0:
                val = 1
            if val not in seen:
                U.append(val)
                seen.add(val)
        seeds.append(sorted(U))
    
    # Quadratic-ish: floor(k^2 * c)
    for c in [1, 2, 3, 5, 8]:
        U = [0]
        seen = {0}
        for k in range(1, 25):
            val = int(c * k * k)
            if val not in seen:
                U.append(val)
                seen.add(val)
        seeds.append(sorted(U))
    
    # Cubic-ish
    for c in [1, 2, 3]:
        U = [0]
        seen = {0}
        for k in range(1, 15):
            val = int(c * k * k * k)
            if val not in seen:
                U.append(val)
                seen.add(val)
        seeds.append(sorted(U))
    
    # Sparse random
    for _ in range(5):
        n = random.randint(5, 15)
        max_val = random.randint(100, 5000)
        U = [0] + sorted(random.sample(range(1, max_val + 1), min(n - 1, max_val)))
        seeds.append(sorted(U))
    
    # Dense small
    for _ in range(5):
        n = random.randint(5, 12)
        max_val = random.randint(20, 100)
        U = [0] + sorted(random.sample(range(1, max_val + 1), min(n - 1, max_val)))
        seeds.append(sorted(U))
    
    return seeds

def simulated_annealing(U, deadline, max_n=30, max_val=20000):
    """Simulated annealing to optimize the set."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    # Temperature schedule
    T = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    while time.time() < deadline - 5 and T > T_min:
        U = list(best_U)
        n = len(U)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add an element
            if n < max_n:
                # Pick a candidate: random in range or based on existing structure
                if random.random() < 0.5:
                    # Random in [1, max_val]
                    new_val = random.randint(1, max_val)
                else:
                    # Near existing elements
                    base = random.choice(U[1:]) if len(U) > 1 else 1
                    delta = random.randint(-max(1, base // 5), max(1, base // 5))
                    new_val = base + delta
                    if new_val < 1:
                        new_val = random.randint(1, max_val)
                if new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        current_score = new_score
                        U = new_U
                    else:
                        delta_e = current_score - new_score
                        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                            current_score = new_score
                            U = new_U
        elif move_type < 0.6:
            # Remove an element
            if n > 2:
                idx = random.randint(1, n - 1)  # Don't remove 0
                new_U = [U[j] for j in range(n) if j != idx]
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    current_score = new_score
                    U = new_U
                else:
                    delta_e = current_score - new_score
                    if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                        current_score = new_score
                        U = new_U
        else:
            # Replace an element
            if n > 1:
                idx = random.randint(1, n - 1)  # Don't replace 0
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val)
                else:
                    old_val = U[idx]
                    delta = random.randint(-max(1, old_val // 4), max(1, old_val // 4))
                    new_val = old_val + delta
                    if new_val < 1:
                        new_val = random.randint(1, max_val)
                if new_val not in U:
                    new_U = [U[j] if j != idx else new_val for j in range(n)]
                    new_U = sorted(set(new_U))
                    if len(new_U) == n:
                        new_score = score_set_fast(new_U)
                        if new_score > current_score:
                            current_score = new_score
                            U = new_U
                        else:
                            delta_e = current_score - new_score
                            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                                current_score = new_score
                                U = new_U
        
        if current_score > best_score:
            best_score = current_score
            best_U = list(U)
        
        T *= cooling
    
    return best_score, best_U

def local_polish(U, deadline):
    """Fine-grained local search to polish the best set."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 2 and iteration < 200:
        improved = False
        iteration += 1
        n = len(U)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old_val = U[i]
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_U = [U[j] if j != i else new_val for j in range(n)]
                new_U = sorted(set(new_U))
                if len(new_U) != n:
                    continue
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 25 and time.time() < deadline - 2:
            max_val = max(U)
            for new_val in range(max(1, max_val - 20), max_val + 21):
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 3 and time.time() < deadline - 2:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
    
    return best_score, best_U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate diverse seeds
    seeds = generate_structural_seeds()
    
    # Score all seeds
    best_score = -1
    best_U = [0, 1]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = score_set_fast(seed)
        if s > best_score:
            best_score = s
            best_U = list(seed)
    
    # Run simulated annealing from multiple seeds
    # Allocate time: ~60s for SA, ~40s for polishing
    sa_deadline = start_time + 70
    polish_deadline = deadline
    
    # Run SA from the best seed and a few others
    sa_seeds = [best_U]
    for seed in seeds[:5]:
        if seed != best_U and len(sa_seeds) < 4:
            sa_seeds.append(list(seed))
    
    for seed in sa_seeds:
        if time.time() > sa_deadline - 5:
            break
        sa_score, sa_U = simulated_annealing(seed, sa_deadline, max_n=28, max_val=15000)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
    
    # Polish the best
    if time.time() < polish_deadline - 2:
        polish_score, polish_U = local_polish(best_U, polish_deadline)
        if polish_score > best_score:
            best_score = polish_score
            best_U = polish_U
    
    return best_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

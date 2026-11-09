"""Combined program: structured candidates + fast evaluation + simulated annealing."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) using full pairwise numpy."""
        a = np.array(sorted(A), dtype=np.int64)
        sums = np.unique(a[:, None] + a[None, :])
        prods = np.unique(a[:, None] * a[None, :])
        return max(len(sums), len(prods))
    
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
            i += 1
        return sorted(divs)
    
    random.seed(42)
    candidates = []
    
    # Sequential
    candidates.append(list(range(1, 21)))
    
    # Arithmetic progressions
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 4, 5, 10]:
            ap = [start + i * step for i in range(n)]
            if max(ap) <= max_val:
                candidates.append(ap)
    
    # Geometric progressions
    for ratio in [2, 3]:
        gp = [ratio ** i for i in range(n)]
        if max(gp) <= max_val:
            candidates.append(gp)
    
    # Divisor sets of highly composite numbers
    highly_composite = [720, 2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400,
                        55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
                        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
                        3603600, 4324320, 6486480, 7207200, 10810800, 14414400]
    for N in highly_composite:
        if time.time() > deadline:
            break
        divs = [d for d in get_divisors(N) if d <= max_val]
        if len(divs) < n:
            continue
        candidates.append(divs[:n])
        candidates.append(divs[-n:])
        if len(divs) >= 40:
            mid = len(divs) // 2
            candidates.append(divs[mid-n//2:mid+n//2])
            candidates.append(divs[:10] + divs[-10:])
    
    # Multiplicative lattice (2^a * 3^b * 5^c * 7^d)
    lattice_vals = set()
    for a in range(0, 14):
        p2 = 2 ** a
        if p2 > max_val:
            break
        for b in range(0, 10):
            p23 = p2 * (3 ** b)
            if p23 > max_val:
                break
            for c in range(0, 7):
                p235 = p23 * (5 ** c)
                if p235 > max_val:
                    break
                for d in range(0, 5):
                    v = p235 * (7 ** d)
                    if v > max_val:
                        break
                    lattice_vals.add(v)
    lattice_sorted = sorted(lattice_vals)
    if len(lattice_sorted) >= n:
        candidates.append(lattice_sorted[:n])
        step = len(lattice_sorted) // n
        candidates.append(lattice_sorted[::step][:n])
    
    # Smooth numbers (only primes 2,3,5,7)
    smooth_nums = []
    for x in range(1, max_val + 1):
        t = x
        while t % 2 == 0:
            t //= 2
        while t % 3 == 0:
            t //= 3
        while t % 5 == 0:
            t //= 5
        while t % 7 == 0:
            t //= 7
        if t == 1:
            smooth_nums.append(x)
    if len(smooth_nums) >= n:
        candidates.append(smooth_nums[:n])
        step = len(smooth_nums) // n
        candidates.append(smooth_nums[::step][:n])
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != n or min(cand) < 1 or max(cand) > max_val:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
    
    # Simulated annealing local search
    current = best_set[:]
    current_score = best_score
    current_set = set(current)
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(300000):
        if time.time() > deadline:
            break
        if temp < 0.1:
            temp = 1.0
            if random.random() < 0.5:
                current = list(range(1, 21))
            else:
                current = [random.randint(1, 1000) for _ in range(n)]
                while len(set(current)) < n:
                    i = random.randint(0, n-1)
                    current[i] = random.randint(1, 1000)
            current = sorted(current)
            current_set = set(current)
            current_score = evaluate(current)
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
        
        i = random.randint(0, n-1)
        old_val = current[i]
        
        if random.random() < 0.7:
            delta = random.randint(-50, 50)
            new_val = old_val + delta
            new_val = max(1, min(max_val, new_val))
        else:
            new_val = random.randint(1, max_val)
        
        if new_val in current_set:
            continue
        
        current[i] = new_val
        current_set.discard(old_val)
        current_set.add(new_val)
        current.sort()
        
        new_score = evaluate(current)
        delta_score = new_score - current_score
        
        if delta_score < 0 or (delta_score == 0 and random.random() < 0.5) or \
           (delta_score > 0 and random.random() < np.exp(-delta_score / temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            current[i] = old_val
            current_set.discard(new_val)
            current_set.add(old_val)
            current.sort()
        
        temp *= cooling
    
    return best_set
# EVOLVE-BLOCK-END

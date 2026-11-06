"""Ambient integer sets with small largest Sidon subset."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _pair_sum_freqs(S):
    """Return dict: sum -> count of unordered pairs (i<=j) with S[i]+S[j]=sum."""
    n = len(S)
    freq = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            freq[s] = freq.get(s, 0) + 1
    return freq

def _max_sum_freq(S):
    """Maximum frequency of any pair sum (lower bound on min Sidon set size... 
    actually upper bound on max Sidon subset: each sum can appear at most once in Sidon)."""
    n = len(S)
    freq = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            freq[s] = freq.get(s, 0) + 1
    return max(freq.values()) if freq else 0

def _max_sidon_size_exact(S):
    """Exact largest Sidon subset size via backtracking with pruning."""
    n = len(S)
    if n <= 1:
        return n
    # Sort for consistency
    S_sorted = sorted(S)
    best = [0]
    
    def backtrack(start, current, sums_used):
        if len(current) > best[0]:
            best[0] = len(current)
        if start >= n:
            return
        # Upper bound: can't exceed current + (n - start)
        if len(current) + (n - start) <= best[0]:
            return
        for i in range(start, n):
            val = S_sorted[i]
            # Check if val can be added: all pair sums with existing elements must be unique
            # and val+val must not conflict
            ok = True
            new_sums = []
            for j in range(len(current)):
                s = val + current[j]
                if s in sums_used:
                    ok = False
                    break
                new_sums.append(s)
            if ok:
                s2 = val + val
                if s2 in sums_used:
                    ok = False
                else:
                    new_sums.append(s2)
            if ok:
                # Also check: the new sums among themselves must be unique
                # (they are all involving val, so val+x for different x are distinct since x distinct)
                # and val+val is distinct from val+x for x != val
                # Since current has distinct elements, val+x for x in current are all distinct
                # and val+val is distinct from val+x unless x=val which can't happen
                # So we just need them not in sums_used
                for s in new_sums:
                    sums_used.add(s)
                backtrack(i + 1, current + [val], sums_used)
                for s in new_sums:
                    sums_used.discard(s)
    
    # Try to build Sidon set
    backtrack(0, [], set())
    return best[0]

def _max_sidon_size_fast(S, timeout=2.0):
    """Fast approximation of max Sidon subset size using greedy + local search."""
    n = len(S)
    if n <= 1:
        return n
    S_sorted = sorted(S)
    
    # Greedy: try to build largest Sidon set
    def greedy_build(order):
        current = []
        sums_used = set()
        for val in order:
            ok = True
            new_sums = []
            for x in current:
                s = val + x
                if s in sums_used:
                    ok = False
                    break
                new_sums.append(s)
            if ok:
                s2 = val + val
                if s2 in sums_used:
                    ok = False
                else:
                    new_sums.append(s2)
            if ok:
                for s in new_sums:
                    sums_used.add(s)
                current.append(val)
        return len(current)
    
    # Try multiple orderings
    best = 0
    # Natural order
    best = max(best, greedy_build(S_sorted))
    # Reverse
    best = max(best, greedy_build(S_sorted[::-1]))
    
    # Random orderings
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < timeout:
        order = S_sorted[:]
        random.shuffle(order)
        sz = greedy_build(order)
        if sz > best:
            best = sz
        attempts += 1
        if attempts > 500:
            break
    
    return best

def _is_valid_sidon(S):
    """Check if S is a Sidon set (all pair sums including diagonals are unique)."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Initial candidate: dense arithmetic progression
    # {0, 2, 4, ..., 22} has many collisions: 0+6=2+4, etc.
    # Let's start with a structured candidate
    # Try: multiples of 2 from 0 to 22
    initial = list(range(0, 24, 2))  # [0,2,4,...,22], 12 elements
    
    # Evaluate initial
    current_best = initial[:]
    current_best_sidon = _max_sidon_size_fast(initial, timeout=3.0)
    
    # Also try some structured candidates
    candidates = [
        list(range(0, 24, 2)),       # even numbers 0-22
        list(range(0, 25, 2)),       # 0,2,...,24
        list(range(1, 25, 2)),       # 1,3,...,23
        list(range(0, 128, 10)),     # 0,10,20,...,110 (only 12 if we pick right)
    ]
    # Fix candidates to have exactly 12 elements in [0,128]
    candidates = []
    # Dense APs
    for d in [1, 2, 3, 4, 5]:
        for start in range(0, 128 - 11*d + 1):
            cand = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
    
    # Evaluate structured candidates
    for cand in candidates[:200]:  # Limit to avoid timeout
        if time.time() > deadline - 10:
            break
        sz = _max_sidon_size_fast(cand, timeout=1.0)
        if sz < current_best_sidon:
            current_best_sidon = sz
            current_best = cand[:]
    
    # Local search: perturb and improve
    def neighbor(S):
        """Generate a neighbor by changing one or two elements."""
        new_S = S[:]
        r = random.random()
        if r < 0.7:
            # Change one element
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val not in new_S:
                new_S[idx] = new_val
        else:
            # Swap two elements with random values
            idx1, idx2 = random.sample(range(12), 2)
            new_val1 = random.randint(0, 128)
            new_val2 = random.randint(0, 128)
            if new_val1 not in new_S and new_val2 not in new_S and new_val1 != new_val2:
                new_S[idx1] = new_val1
                new_S[idx2] = new_val2
        return new_S
    
    # Simulated annealing
    temperature = 5.0
    cooling = 0.995
    best_ever = current_best_sidon
    best_ever_set = current_best[:]
    
    # Use fast evaluation (upper bound via max sum freq is too loose)
    # Use greedy-based estimate for speed
    iterations = 0
    while time.time() < deadline - 5.0:
        new_S = neighbor(current_best)
        if len(set(new_S)) != 12:
            continue
        if not all(0 <= x <= 128 for x in new_S):
            continue
        
        new_sidon = _max_sidon_size_fast(new_S, timeout=0.5)
        
        if new_sidon <= current_best_sidon:
            current_best = new_S[:]
            current_best_sidon = new_sidon
            if new_sidon < best_ever:
                best_ever = new_sidon
                best_ever_set = new_S[:]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_sidon - current_best_sidon
            if delta < 10:  # Don't accept huge jumps
                prob = math.exp(-delta / max(temperature, 0.1))
                if random.random() < prob:
                    current_best = new_S[:]
                    current_best_sidon = new_sidon
        
        iterations += 1
        if iterations % 50 == 0:
            temperature *= cooling
            if temperature < 0.5:
                temperature = 0.5
        
        # Occasionally try larger moves
        if iterations % 200 == 0 and time.time() < deadline - 15:
            # Try a random restart from a structured candidate
            if random.random() < 0.3:
                cand = [random.randint(0, 128) for _ in range(12)]
                if len(set(cand)) == 12:
                    sz = _max_sidon_size_fast(cand, timeout=1.0)
                    if sz < current_best_sidon:
                        current_best = cand[:]
                        current_best_sidon = sz
                        if sz < best_ever:
                            best_ever = sz
                            best_ever_set = cand[:]
    
    # Final polish: try exact evaluation on best candidates
    if time.time() < deadline - 3.0:
        exact_sz = _max_sidon_size_exact(best_ever_set)
        if exact_sz < best_ever:
            best_ever = exact_sz
    
    return best_ever_set

# EVOLVE-BLOCK-END

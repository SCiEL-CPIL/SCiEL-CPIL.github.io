"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _largest_sidon_subset(s):
    """Find the largest subset of s with all pair sums (including diagonals) unique."""
    s = sorted(s)
    n = len(s)
    best = 0
    # Try all subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        ok = False
                        break
                    sums.add(sm)
                if not ok:
                    break
            if ok:
                best = size
                found = True
                break
        if found:
            break
    return best

def _evaluate(s):
    """Return the largest Sidon subset size for set s."""
    return _largest_sidon_subset(s)

def _conflict_score(subset):
    """Count total sum collisions: for each sum value appearing c times, add C(c,2)."""
    sum_counts = defaultdict(int)
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            sm = subset[i] + subset[j]
            sum_counts[sm] += 1
    conflicts = 0
    for c in sum_counts.values():
        if c >= 2:
            conflicts += c * (c - 1) // 2
    return conflicts

def _random_candidate():
    return sorted(random.sample(range(129), 12))

def _local_search(current, deadline, max_iter=500):
    """Hill-climbing: try swapping one element with another from [0,128].
    Uses conflict score as proxy, evaluates true objective periodically."""
    current = sorted(current)
    best_true_score = _evaluate(current)
    best_set = list(current)
    best_conflict = _conflict_score(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        improved = False
        positions = list(range(12))
        random.shuffle(positions)
        for i in positions:
            if time.time() > deadline:
                break
            candidates = list(all_vals - in_set)
            random.shuffle(candidates)
            best_new_conflict = best_conflict
            best_new_val = None
            for new_val in candidates:
                candidate = list(current)
                candidate[i] = new_val
                candidate = sorted(candidate)
                cs = _conflict_score(candidate)
                if cs > best_new_conflict:
                    best_new_conflict = cs
                    best_new_val = new_val
            if best_new_val is not None:
                old_val = current[i]
                in_set.discard(old_val)
                in_set.add(best_new_val)
                current = list(best_set)
                current[i] = best_new_val
                current = sorted(current)
                best_conflict = best_new_conflict
                best_set = current
                # Evaluate true objective periodically
                if iter_count % 10 == 0 or best_conflict > 50:
                    ts = _evaluate(current)
                    if ts < best_true_score:
                        best_true_score = ts
                improved = True
        if not improved:
            break
    
    # Final evaluation
    final_score = _evaluate(best_set)
    return best_set, final_score

def _local_search_limited(current, deadline):
    """Limited local search with fewer iterations."""
    return _local_search(current, deadline, max_iter=100)

def _simulated_annealing(current, deadline, max_iter=2000):
    """Simulated annealing using conflict score as proxy."""
    current = sorted(current)
    best_set = list(current)
    best_true_score = _evaluate(current)
    best_conflict = _conflict_score(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    temp = 2.0
    cooling = 0.995
    
    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        if time.time() > deadline:
            break
        
        i = random.randint(0, 11)
        old_val = current[i]
        new_val = random.choice(list(all_vals - in_set))
        
        candidate = list(current)
        candidate[i] = new_val
        candidate = sorted(candidate)
        
        new_conflict = _conflict_score(candidate)
        delta = new_conflict - best_conflict
        
        if delta >= 0 or random.random() < pow(2.718, delta / temp):
            # Accept
            in_set.discard(old_val)
            in_set.add(new_val)
            current = candidate
            if new_conflict > best_conflict:
                best_conflict = new_conflict
                best_set = list(current)
                # Evaluate true objective periodically
                if iter_count % 20 == 0:
                    ts = _evaluate(current)
                    if ts < best_true_score:
                        best_true_score = ts
        
        temp *= cooling
    
    final_score = _evaluate(best_set)
    return best_set, final_score

def _structured_candidate():
    """Generate a structured candidate with many potential sum collisions."""
    # Arithmetic progression with small difference
    start = random.randint(0, 100)
    diff = random.choice([1, 2, 3])
    candidate = []
    for i in range(12):
        v = start + i * diff
        if v <= 128:
            candidate.append(v)
    while len(candidate) < 12:
        candidate.append(random.randint(0, 128))
    candidate = sorted(set(candidate))
    while len(candidate) < 12:
        candidate.append(random.randint(0, 128))
    candidate = sorted(set(candidate))
    return candidate

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s safety margin before 60s limit
    
    random.seed(42)
    
    # Start with structured candidate
    best_set = _structured_candidate()
    best_score = _evaluate(best_set)
    
    # Try several random and structured restarts with local search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts % 3 == 0:
            candidate = _structured_candidate()
        else:
            candidate = _random_candidate()
        score = _evaluate(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate
            # Local search from this candidate
            best_set, best_score = _local_search_limited(best_set, deadline)
        
        # Also try simulated annealing from current best periodically
        if restarts % 5 == 0 and time.time() < deadline:
            sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iter=300)
            if sa_score < best_score:
                best_score = sa_score
                best_set = sa_set
    
    # Final polish: try more aggressive local search
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iter=200)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END

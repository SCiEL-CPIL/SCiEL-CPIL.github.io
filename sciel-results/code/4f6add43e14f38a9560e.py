"""Combined program: graph-based collision maximization with structured candidates."""
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

def _collision_graph(s):
    """Build a 12-vertex graph where edge (i,j) weight = number of other pairs (k,l)
    with k<l, {k,l} != {i,j}, and s[k]+s[l] == s[i]+s[j].
    Higher edge weights mean more collisions, which limits Sidon subsets."""
    n = len(s)
    # Map each sum to list of pairs (i,j) with i<=j
    sum_map = defaultdict(list)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            sum_map[sm].append((i, j))
    
    # Edge weight: for pair (i,j), count how many OTHER pairs have same sum
    edge_weight = [[0] * n for _ in range(n)]
    for sm, pairs in sum_map.items():
        cnt = len(pairs)
        if cnt > 1:
            for (i, j) in pairs:
                edge_weight[i][j] = cnt - 1
                edge_weight[j][i] = cnt - 1
    
    # Total collision score: sum of all edge weights
    total = 0
    for i in range(n):
        for j in range(i, n):
            total += edge_weight[i][j]
    return total, edge_weight

def _conflict_score(s):
    """Count total sum collisions: for each sum value appearing c times, add C(c,2)."""
    sum_counts = defaultdict(int)
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            sum_counts[sm] += 1
    conflicts = 0
    for c in sum_counts.values():
        if c >= 2:
            conflicts += c * (c - 1) // 2
    return conflicts

def _random_candidate():
    return sorted(random.sample(range(129), 12))

def _structured_candidate():
    """Generate structured candidates with high collision potential."""
    candidates = []
    
    # Symmetric pairs around various centers
    for center in range(20, 110):
        for spread in range(1, 30):
            c = []
            for k in range(6):
                a = center - spread + k * 2
                b = center + spread - k * 2
                if 0 <= a <= 128 and 0 <= b <= 128 and a != b:
                    c.append(a)
                    c.append(b)
            if len(c) == 12 and len(set(c)) == 12:
                candidates.append(sorted(set(c)))
    
    # Arithmetic progressions
    for a in range(0, 80):
        for d in range(1, 15):
            c = [a + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in c):
                candidates.append(sorted(c))
    
    # Clusters: groups of close numbers
    for _ in range(20):
        base = random.randint(0, 60)
        c = []
        for g in range(3):
            gbase = base + g * random.randint(20, 40)
            for k in range(4):
                v = gbase + random.randint(0, 5)
                if 0 <= v <= 128:
                    c.append(v)
        c = list(set(c))
        while len(c) < 12:
            c.append(random.randint(0, 128))
        c = sorted(set(c))
        while len(c) < 12:
            c.append(random.randint(0, 128))
        c = sorted(set(c))
        if len(c) == 12:
            candidates.append(c)
    
    if candidates:
        return random.choice(candidates)
    return _random_candidate()

def _local_search(current, deadline, max_iter=300):
    """Hill-climbing on collision graph score, with periodic true evaluation."""
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
            for new_val in candidates[:30]:  # Limit candidates for speed
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
                if iter_count % 15 == 0:
                    ts = _evaluate(current)
                    if ts < best_true_score:
                        best_true_score = ts
                improved = True
        if not improved:
            break
    
    final_score = _evaluate(best_set)
    return best_set, final_score

def _simulated_annealing(current, deadline, max_iter=1500):
    """Simulated annealing maximizing collision score."""
    current = sorted(current)
    best_set = list(current)
    best_true_score = _evaluate(current)
    best_conflict = _conflict_score(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    temp = 3.0
    cooling = 0.997
    
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
            in_set.discard(old_val)
            in_set.add(new_val)
            current = candidate
            if new_conflict > best_conflict:
                best_conflict = new_conflict
                best_set = list(current)
                if iter_count % 25 == 0:
                    ts = _evaluate(current)
                    if ts < best_true_score:
                        best_true_score = ts
        
        temp *= cooling
    
    final_score = _evaluate(best_set)
    return best_set, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin before 90s kill
    
    random.seed(42)
    
    # Generate diverse initial candidates
    initial_candidates = []
    initial_candidates.append(_structured_candidate())
    for _ in range(10):
        initial_candidates.append(_structured_candidate())
    for _ in range(10):
        initial_candidates.append(_random_candidate())
    
    # Evaluate all initial candidates
    best_set = None
    best_score = 12
    best_conflict = -1
    
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        cand = sorted(set(cand))
        if len(cand) != 12:
            continue
        cs = _conflict_score(cand)
        if cs > best_conflict:
            best_conflict = cs
            score = _evaluate(cand)
            if score < best_score:
                best_score = score
                best_set = cand
    
    if best_set is None:
        best_set = sorted(random.sample(range(129), 12))
        best_score = _evaluate(best_set)
    
    # Run local search and simulated annealing from best candidate
    if time.time() < deadline:
        ls_set, ls_score = _local_search(best_set, deadline, max_iter=200)
        if ls_score < best_score:
            best_score = ls_score
            best_set = ls_set
    
    if time.time() < deadline:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iter=1000)
        if sa_score < best_score:
            best_score = sa_score
            best_set = sa_set
    
    # Try more restarts if time permits
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        if restarts % 3 == 0:
            candidate = _structured_candidate()
        else:
            candidate = _random_candidate()
        
        cs = _conflict_score(candidate)
        if cs > best_conflict * 0.8:  # Only evaluate promising candidates
            score = _evaluate(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate
                best_conflict = cs
                # Local search from this candidate
                if time.time() < deadline - 5:
                    ls_set, ls_score = _local_search(best_set, deadline, max_iter=100)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_set = ls_set
    
    # Final polish
    if time.time() < deadline - 2:
        final_set, final_score = _local_search(best_set, deadline, max_iter=100)
        if final_score < best_score:
            best_score = final_score
            best_set = final_set
    
    return sorted(best_set)

# EVOLVE-BLOCK-END

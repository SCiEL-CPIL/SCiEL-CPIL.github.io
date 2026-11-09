"""Improved: bitmask K(2,2,2) constraints + multi-strategy greedy + enhanced local search."""
import time
import random
import itertools
import math

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute K(2,2,2) configs as bitmasks of the 8 transversal triples.
CONFIG_MASKS = []
TRIPLE_CFGS = [[] for _ in range(N_TRIPLES)]

def _precompute():
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    for pts in six_subsets:
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)],
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                transversals = [
                    (a, b, c), (a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                    (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p),
                ]
                mask = 0
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is not None:
                        mask |= (1 << idx)
                cid = len(CONFIG_MASKS)
                CONFIG_MASKS.append(mask)
                m = mask
                while m:
                    low = m & (-m)
                    idx = low.bit_length() - 1
                    TRIPLE_CFGS[idx].append(cid)
                    m ^= low

_precompute()
N_CONFIGS = len(CONFIG_MASKS)

def _is_valid_addition(triple_idx, in_set, config_counts):
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CFGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def _update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CFGS[triple_idx]:
        config_counts[cfg_id] += delta

def _init_counts(current_set):
    config_counts = [0] * N_CONFIGS
    for t_idx in current_set:
        _update_counts(t_idx, 1, config_counts)
    return config_counts

def _greedy_construct(seed, deadline):
    """Basic random-order greedy."""
    random.seed(seed)
    order = list(range(N_TRIPLES))
    random.shuffle(order)
    current_set = set()
    config_counts = [0] * N_CONFIGS
    for t_idx in order:
        if time.time() > deadline:
            break
        if _is_valid_addition(t_idx, current_set, config_counts):
            current_set.add(t_idx)
            _update_counts(t_idx, 1, config_counts)
    return current_set, config_counts

def _greedy_construct_priority(seed, deadline):
    """Greedy that picks the most constrained valid triple each step."""
    random.seed(seed)
    current_set = set()
    config_counts = [0] * N_CONFIGS
    remaining = set(range(N_TRIPLES))
    
    while remaining and time.time() < deadline:
        valid_candidates = []
        for t_idx in remaining:
            if _is_valid_addition(t_idx, current_set, config_counts):
                # How many configs are close to saturation
                tightness = sum(1 for cid in TRIPLE_CFGS[t_idx] if config_counts[cid] >= 5)
                valid_candidates.append((tightness, random.random(), t_idx))
        
        if not valid_candidates:
            break
        
        valid_candidates.sort(key=lambda x: (-x[0], x[1]))
        t_idx = valid_candidates[0][2]
        current_set.add(t_idx)
        remaining.discard(t_idx)
        _update_counts(t_idx, 1, config_counts)
    
    return current_set, config_counts

def _greedy_construct_least_conflict(seed, deadline):
    """Greedy that picks the triple with fewest total config conflicts."""
    random.seed(seed)
    current_set = set()
    config_counts = [0] * N_CONFIGS
    remaining = set(range(N_TRIPLES))
    
    while remaining and time.time() < deadline:
        best_t = None
        best_score = float('inf')
        best_rand = None
        for t_idx in remaining:
            if _is_valid_addition(t_idx, current_set, config_counts):
                # Fewer configs with high count = less conflict
                score = sum(config_counts[cid] for cid in TRIPLE_CFGS[t_idx])
                if score < best_score - 1e-9 or (abs(score - best_score) < 1e-9 and best_rand is None):
                    best_score = score
                    best_t = t_idx
                    best_rand = random.random()
                elif abs(score - best_score) < 1e-9:
                    if random.random() < 0.5:
                        best_t = t_idx
                        best_rand = random.random()
        
        if best_t is None:
            break
        
        current_set.add(best_t)
        remaining.discard(best_t)
        _update_counts(best_t, 1, config_counts)
    
    return current_set, config_counts

def _local_search_add_priority(current_set, config_counts, deadline):
    """Add triples in order of constraint tightness."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = []
        for t_idx in range(N_TRIPLES):
            if t_idx not in current_set and _is_valid_addition(t_idx, current_set, config_counts):
                tightness = sum(1 for cid in TRIPLE_CFGS[t_idx] if config_counts[cid] >= 4)
                candidates.append((tightness, t_idx))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: -x[0])
        for _, t_idx in candidates:
            if time.time() > deadline:
                break
            if t_idx not in current_set and _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _local_search_swap(current_set, config_counts, deadline, max_iters=50000):
    """1-out / 1-in swaps, keep best."""
    best_size = len(current_set)
    best_set = current_set.copy()
    best_counts = config_counts.copy()
    iters = 0
    while iters < max_iters and time.time() < deadline:
        iters += 1
        if not current_set:
            break
        t_out = random.choice(tuple(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)

        t_in = random.randrange(N_TRIPLES)
        if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            current_set, config_counts = _local_search_add_priority(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
    return best_set, best_counts

def _local_search_exchange(current_set, config_counts, deadline, max_iters=30000):
    """1-out / 2-in to gain net +1."""
    best_size = len(current_set)
    best_set = current_set.copy()
    best_counts = config_counts.copy()
    iters = 0
    while iters < max_iters and time.time() < deadline:
        iters += 1
        if not current_set:
            break
        t_out = random.choice(tuple(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)

        added = []
        for _ in range(2):
            t_in = random.randrange(N_TRIPLES)
            if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
                current_set.add(t_in)
                _update_counts(t_in, 1, config_counts)
                added.append(t_in)
        if len(added) == 2:
            current_set, config_counts = _local_search_add_priority(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            for t in added:
                current_set.discard(t)
                _update_counts(t, -1, config_counts)
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
    return best_set, best_counts

def _simulated_annealing(current_set, config_counts, deadline):
    """SA with swap moves and fill after each accepted move."""
    best_set = current_set.copy()
    best_size = len(current_set)
    best_counts = config_counts.copy()

    current_set, config_counts = _local_search_add_priority(current_set, config_counts, deadline)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set.copy()
        best_counts = config_counts.copy()

    temp = 10.0
    cooling = 0.998
    min_size_threshold = max(1, best_size - 5)  # allow dropping to this size
    while time.time() < deadline and temp > 0.1:
        if not current_set:
            break
        t_out = random.choice(tuple(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)
        t_in = random.randrange(N_TRIPLES)
        if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            current_set, config_counts = _local_search_add_priority(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
        temp *= cooling
    return best_set, best_counts

def _final_polish(current_set, config_counts, deadline):
    """Final exhaustive single-addition polish."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = []
        for t_idx in range(N_TRIPLES):
            if t_idx not in current_set:
                if _is_valid_addition(t_idx, current_set, config_counts):
                    tightness = sum(1 for cid in TRIPLE_CFGS[t_idx] if config_counts[cid] >= 3)
                    candidates.append((tightness, t_idx))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: -x[0])
        for _, t_idx in candidates:
            if time.time() > deadline:
                break
            if t_idx not in current_set and _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _multi_swap_restart(current_set, config_counts, deadline):
    """Try multiple restarts of local search from slightly perturbed states."""
    best_set = current_set.copy()
    best_size = len(current_set)
    best_counts = config_counts.copy()
    
    remaining_time = deadline - time.time()
    if remaining_time < 5:
        return best_set, best_counts
    
    num_restarts = min(5, int(remaining_time / 15))
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        # Perturb: remove 3-5 random triples
        perturbed_set = current_set.copy()
        perturbed_counts = config_counts.copy()
        n_remove = random.randint(3, 5)
        to_remove = random.sample(list(perturbed_set), min(n_remove, len(perturbed_set)))
        for t in to_remove:
            perturbed_set.remove(t)
            _update_counts(t, -1, perturbed_counts)
        
        # Fill and local search
        perturbed_set, perturbed_counts = _local_search_add_priority(perturbed_set, perturbed_counts, deadline)
        if len(perturbed_set) > best_size:
            best_size = len(perturbed_set)
            best_set = perturbed_set.copy()
            best_counts = perturbed_counts.copy()
        current_set = perturbed_set
        config_counts = perturbed_counts
    
    return best_set, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    best_result = set()
    best_size = 0
    best_counts = [0] * N_CONFIGS

    # Phase 1: Multiple greedy strategies with local search
    num_greedy = 40
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        seed = i * 104729 + 7
        if i % 4 == 0:
            current_set, config_counts = _greedy_construct(seed, deadline)
        elif i % 4 == 1:
            current_set, config_counts = _greedy_construct_priority(seed, deadline)
        elif i % 4 == 2:
            current_set, config_counts = _greedy_construct_least_conflict(seed, deadline)
        else:
            current_set, config_counts = _greedy_construct(seed, deadline)
        current_set, config_counts = _local_search_add_priority(current_set, config_counts, deadline)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
            best_counts = config_counts.copy()

    # Phase 2: Swaps and exchanges
    if time.time() < deadline - 30:
        swap_deadline = min(deadline - 25, time.time() + 25)
        best_result, best_counts = _local_search_swap(best_result, best_counts, swap_deadline)
    if time.time() < deadline - 20:
        exch_deadline = min(deadline - 15, time.time() + 15)
        best_result, best_counts = _local_search_exchange(best_result, best_counts, exch_deadline)

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_deadline = deadline - 10
        best_result, best_counts = _simulated_annealing(best_result, best_counts, sa_deadline)

    # Phase 4: Multi-swap restarts
    if time.time() < deadline - 10:
        restart_deadline = deadline - 5
        best_result, best_counts = _multi_swap_restart(best_result, best_counts, restart_deadline)

    # Phase 5: Final polish
    if time.time() < deadline:
        best_result, best_counts = _final_polish(best_result, best_counts, deadline - 1)

    return [list(ALL_TRIPLES[i]) for i in best_result]

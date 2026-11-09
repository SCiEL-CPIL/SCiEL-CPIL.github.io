"""Combined: bitmask K(2,2,2) constraints + greedy + add/swap/exchange local search + branch-and-bound."""
import time
import random
import itertools

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute K(2,2,2) configs as bitmasks of the 8 transversal triples.
CONFIG_MASKS = []          # int bitmask per config
TRIPLE_CFGS = [[] for _ in range(N_TRIPLES)]  # config ids per triple

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
                # index: for each set bit, add cid
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

def _local_search_add(current_set, config_counts, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        for t_idx in candidates:
            if time.time() > deadline:
                return current_set, config_counts
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _local_search_swap(current_set, config_counts, deadline, max_iters=20000):
    """1-out / 1-in swaps to change composition, then fill. Keep if size improves."""
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

        # try to add a better triple
        t_in = random.randrange(N_TRIPLES)
        if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            # fill
            current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            # revert
            if t_in in current_set:
                current_set.remove(t_in)
                _update_counts(t_in, -1, config_counts)
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
    return best_set, best_counts

def _local_search_exchange(current_set, config_counts, deadline, max_iters=10000):
    """1-out / 2-in to gain net +1 when stuck."""
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
            current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            # revert
            for t in added:
                current_set.discard(t)
                _update_counts(t, -1, config_counts)
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
    return best_set, best_counts

def _branch_and_bound(candidates, current_set, config_counts, deadline, time_budget):
    """B&B to maximize additions from candidates without completing any config."""
    start = time.time()
    best_extra = 0
    best_choice = []

    # sort candidates by how constrained they are (fewer configs => more flexible first)
    cand_list = list(candidates)
    cand_list.sort(key=lambda t: len(TRIPLE_CFGS[t]))

    # quick greedy lower bound
    greedy_set = set()
    greedy_counts = config_counts[:]
    for t in cand_list:
        ok = True
        for cid in TRIPLE_CFGS[t]:
            if greedy_counts[cid] == 7:
                ok = False
                break
        if ok:
            greedy_set.add(t)
            for cid in TRIPLE_CFGS[t]:
                greedy_counts[cid] += 1
    best_extra = len(greedy_set)
    best_choice = list(greedy_set)

    def rec(pos, chosen, counts):
        nonlocal best_extra, best_choice
        if time.time() - start > time_budget:
            return
        # upper bound: remaining candidates + chosen
        if len(chosen) + (len(cand_list) - pos) <= best_extra:
            return
        if pos >= len(cand_list):
            if len(chosen) > best_extra:
                best_extra = len(chosen)
                best_choice = list(chosen)
            return
        t = cand_list[pos]
        # branch 1: take t if valid
        ok = True
        for cid in TRIPLE_CFGS[t]:
            if counts[cid] == 7:
                ok = False
                break
        if ok:
            new_counts = counts[:]
            for cid in TRIPLE_CFGS[t]:
                new_counts[cid] += 1
            chosen.append(t)
            rec(pos + 1, chosen, new_counts)
            chosen.pop()
        # branch 2: skip t
        rec(pos + 1, chosen, counts)

    # only attempt B&B if candidate set is small enough
    if len(cand_list) <= 28:
        rec(0, [], config_counts[:])

    # apply best
    for t in best_choice:
        if t not in current_set:
            current_set.add(t)
            _update_counts(t, 1, config_counts)
    return current_set, config_counts

def _simulated_annealing(current_set, config_counts, deadline):
    best_set = current_set.copy()
    best_size = len(current_set)
    best_counts = config_counts.copy()

    current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set.copy()
        best_counts = config_counts.copy()

    temp = 5.0
    cooling = 0.995
    while time.time() < deadline and temp > 0.01:
        if not current_set:
            break
        t_out = random.choice(tuple(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)
        t_in = random.randrange(N_TRIPLES)
        if t_in not in current_set and _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                best_counts = config_counts.copy()
        else:
            if t_in in current_set:
                current_set.remove(t_in)
                _update_counts(t_in, -1, config_counts)
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
        temp *= cooling
    return best_set, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    best_result = set()
    best_size = 0
    best_counts = [0] * N_CONFIGS

    # Greedy restarts + local search
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        seed = i * 7919 + 104729
        current_set, config_counts = _greedy_construct(seed, deadline)
        current_set, config_counts = _local_search_add(current_set, config_counts, deadline)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
            best_counts = config_counts.copy()

    # Refine the best with swaps and exchanges
    if time.time() < deadline - 30:
        swap_deadline = min(deadline - 25, time.time() + 20)
        best_result, best_counts = _local_search_swap(best_result, best_counts, swap_deadline)
    if time.time() < deadline - 20:
        exch_deadline = min(deadline - 15, time.time() + 15)
        best_result, best_counts = _local_search_exchange(best_result, best_counts, exch_deadline)

    # Simulated annealing
    if time.time() < deadline - 10:
        sa_deadline = deadline - 5
        best_result, best_counts = _simulated_annealing(best_result, best_counts, sa_deadline)

    # Final fill
    if time.time() < deadline:
        best_result, best_counts = _local_search_add(best_result, best_counts, deadline)

    # Branch-and-bound on remaining candidates if time allows
    if time.time() < deadline - 3:
        candidates = [t for t in range(N_TRIPLES) if t not in best_result]
        bnb_deadline = deadline - 2
        best_result, best_counts = _branch_and_bound(
            candidates, best_result, best_counts, bnb_deadline, 10
        )

    return [list(ALL_TRIPLES[i]) for i in best_result]

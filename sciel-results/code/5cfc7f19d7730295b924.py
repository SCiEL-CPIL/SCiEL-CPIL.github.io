import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 58.0  # safety margin before 60s kill


def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """BFS on subset automaton; returns exact shortest reset word length, 0 if not synchronizing."""
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0


def _fast_threshold(next_a, next_b):
    """BFS on precomputed subset transitions; returns shortest reset word length, 0 if not syncing."""
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0


def _construct_cycle_compress(n, e, t, w):
    """Build automaton: a is full cycle, b maps [0,w-1]->identity, e->t, all others->0."""
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(w):
        b[i] = i
    b[e] = t
    return a, b


def _construct_pinwheel(n):
    """Construct a pinwheel automaton: a is a cycle, b shifts backward with a defect."""
    a = [(i + 1) % n for i in range(n)]
    b = [(i - 1) % n for i in range(n)]
    b[0] = 0  # defect: state 0 maps to itself instead of n-1
    return a, b


def _construct_defect_cycle(n, defect_state, defect_dest):
    """Construct a cycle with one defect in b."""
    a = [(i + 1) % n for i in range(n)]
    b = [(i - 1) % n for i in range(n)]
    b[defect_state] = defect_dest
    return a, b


def _hill_climb(a_init, b_init, rng, deadline, best_holder):
    """
    Hill climbing with restarts on the transition space.
    Focuses on b mutations (90%) and occasional a mutations (10%).
    Uses precomputed subset transitions for speed.
    """
    n = len(a_init)
    cur_a = a_init[:]
    cur_b = b_init[:]
    
    next_a = _build_subset_trans(cur_a)
    next_b = _build_subset_trans(cur_b)
    cur_threshold = _fast_threshold(next_a, next_b)
    
    if cur_threshold == 0:
        return None, None, 0
    
    best_a_local = cur_a[:]
    best_b_local = cur_b[:]
    best_threshold_local = cur_threshold
    
    if best_threshold_local > best_holder[0]:
        best_holder[0] = best_threshold_local
        best_holder[1] = best_a_local[:]
        best_holder[2] = best_b_local[:]
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with a new random b, keeping the best a
            cur_b = [rng.randint(0, n - 1) for _ in range(n)]
            # Ensure at least one fixed point or sink to help synchronization
            cur_b[rng.randint(0, n - 1)] = 0
            next_b = _build_subset_trans(cur_b)
            cur_threshold = _fast_threshold(next_a, next_b)
            if cur_threshold == 0:
                # Try a different a
                cur_a = [rng.randint(0, n - 1) for _ in range(n)]
                # Ensure a is a permutation (cycle-like)
                # Simple approach: random permutation
                cur_a = list(range(n))
                rng.shuffle(cur_a)
                next_a = _build_subset_trans(cur_a)
                cur_threshold = _fast_threshold(next_a, next_b)
                if cur_threshold == 0:
                    # Give up on this restart
                    break
            no_improve = 0
            continue
        
        # Mutate
        mutate_a = rng.random() < 0.1  # 10% chance to mutate a
        
        if mutate_a:
            old_a = cur_a[:]
            # Swap two elements to keep it a permutation
            s1, s2 = rng.sample(range(n), 2)
            cur_a[s1], cur_a[s2] = cur_a[s2], cur_a[s1]
            next_a = _build_subset_trans(cur_a)
            new_threshold = _fast_threshold(next_a, next_b)
        else:
            old_b = cur_b[:]
            state = rng.randint(0, n - 1)
            new_dest = rng.randint(0, n - 1)
            if new_dest == old_b[state]:
                continue
            cur_b[state] = new_dest
            next_b = _build_subset_trans(cur_b)
            new_threshold = _fast_threshold(next_a, next_b)
        
        if new_threshold > cur_threshold:
            cur_threshold = new_threshold
            no_improve = 0
            if new_threshold > best_threshold_local:
                best_threshold_local = new_threshold
                best_a_local = cur_a[:]
                best_b_local = cur_b[:]
                if best_threshold_local > best_holder[0]:
                    best_holder[0] = best_threshold_local
                    best_holder[1] = best_a_local[:]
                    best_holder[2] = best_b_local[:]
        else:
            # Revert
            if mutate_a:
                cur_a = old_a
                next_a = _build_subset_trans(cur_a)
            else:
                cur_b = old_b
                next_b = _build_subset_trans(cur_b)
            no_improve += 1
    
    return best_a_local, best_b_local, best_threshold_local


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(42)

    best_holder = [0, None, None]  # [score, a, b]

    # Phase 1: Analytic scan of the cycle+compress family
    n = N
    best_family = None
    best_family_score = 0
    for w in range(1, n // 2 + 1):
        for e in range(w, n):
            for t in range(w, n):
                if t == e:
                    continue
                # Analytic formula for threshold
                if e >= w and t >= w:
                    score = (n - 1) - w - ((e - t - 1) % n)
                    if score > best_family_score:
                        best_family_score = score
                        best_family = (e, t, w)
    
    # Verify the family best with real BFS and store
    if best_family is not None:
        e, t, w = best_family
        a, b = _construct_cycle_compress(n, e, t, w)
        real = compute_threshold(a, b)
        if real > best_holder[0]:
            best_holder[0] = real
            best_holder[1] = a[:]
            best_holder[2] = b[:]

    # Phase 2: Try pinwheel and defect cycle constructions
    pin_a, pin_b = _construct_pinwheel(n)
    pin_score = compute_threshold(pin_a, pin_b)
    if pin_score > best_holder[0]:
        best_holder[0] = pin_score
        best_holder[1] = pin_a[:]
        best_holder[2] = pin_b[:]

    # Try various defect positions
    for ds in range(n):
        for dd in range(n):
            if dd == ds:
                continue
            da, db = _construct_defect_cycle(n, ds, dd)
            ds_score = compute_threshold(da, db)
            if ds_score > best_holder[0]:
                best_holder[0] = ds_score
                best_holder[1] = da[:]
                best_holder[2] = db[:]

    # Phase 3: Hill climbing from the best found so far
    if time.time() < deadline - 5.0 and best_holder[1] is not None:
        _hill_climb(best_holder[1], best_holder[2], rng, deadline - 5.0, best_holder)

    # Phase 4: Multiple restarts with random initializations
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 3.0:
            break
        # Random initial a (permutation) and b
        ra = list(range(n))
        rng.shuffle(ra)
        rb = [rng.randint(0, n - 1) for _ in range(n)]
        rb[rng.randint(0, n - 1)] = 0  # Ensure some structure
        r_score = compute_threshold(ra, rb)
        if r_score == 0:
            continue
        if r_score > best_holder[0]:
            best_holder[0] = r_score
            best_holder[1] = ra[:]
            best_holder[2] = rb[:]
        if time.time() < deadline - 3.0:
            _hill_climb(ra, rb, rng, deadline - 3.0, best_holder)

    # Fallback: guaranteed synchronizing automaton
    if best_holder[1] is None or best_holder[0] == 0:
        a = [(i + 1) % n for i in range(n)]
        b = [0] * n
        return [a, b]

    return [best_holder[1], best_holder[2]]


if __name__ == "__main__":
    a, b = construct()
    print("Best score:", compute_threshold(a, b))
    print("A:", a)
    print("B:", b)

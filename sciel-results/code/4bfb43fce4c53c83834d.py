"""Parallel tempering + WalkSAT repair for 4-AP-free set maximizing 3-AP count."""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s


def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    # For each element, store the OTHER elements in each 4-AP it participates in
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # For each element, store the OTHER elements in each 3-AP it participates in
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

    # Precompute 3-AP weights
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]

    def is_safe_add(S, x):
        """Check if adding x to S (bitmask) creates any 4-AP involving x."""
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                return False
        return True

    def gain_3aps(S, x):
        """Number of new 3-APs formed by adding x to S."""
        gain = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        return gain

    def loss_3aps(S, x):
        """Number of 3-APs lost by removing x from S (x must be in S)."""
        loss = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                loss += 1
        return loss

    def count_3aps_bitmask(S):
        """Count 3-APs in bitmask S (0-indexed)."""
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def count_4aps_bitmask(S):
        """Count 4-APs fully contained in bitmask S."""
        cnt = 0
        for a, b, c, d in four_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs, prioritizing low 3-AP weight."""
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    elems = (a, b, c, d)
                    worst_idx = min(range(4), key=lambda i: (elem_3ap_weight[elems[i]], -elem_4ap_weight[elems[i]]))
                    worst = elems[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def walksat_repair(S, max_steps=200):
        """WalkSAT-style repair: randomly pick a violating 4-AP and remove one of its elements.
        Then try to re-add removed elements in a greedy manner."""
        removed = []
        for _ in range(max_steps):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            # Find a violating 4-AP
            found = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    # Remove a random element from this 4-AP
                    elems = [a, b, c, d]
                    victim = rng.choice(elems)
                    S &= ~(1 << victim)
                    removed.append(victim)
                    found = True
                    break
            if not found:
                break
        
        # Try to re-add removed elements greedily
        for x in removed:
            if is_safe_add(S, x):
                S |= (1 << x)
        
        return S

    def greedy_construct(order):
        """Greedy: add elements in given order, skip if creates 4-AP."""
        S = 0
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def local_search_2opt(S, max_iters=300):
        """2-opt local search: try all (remove x, add y) pairs to improve 3-AP count."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try swaps: remove x, add y
            best_swap = None
            best_gain = 0
            for x_remove in members:
                S_after_remove = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after_remove, x_add):
                        continue
                    gain = gain_3aps(S_after_remove, x_add)
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue

            if not improved:
                break

        return current_S, current_score

    # === PHASE 1: Multi-start greedy with structured orderings ===
    best_S = 0
    best_score = -1

    # Ordering 1: by 3-AP weight descending
    order1 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    S1 = greedy_construct(order1)
    S1 = repair_4aps(S1)
    score1 = count_3aps_bitmask(S1)
    if score1 > best_score:
        best_score = score1
        best_S = S1

    # Ordering 2: residue class mod 3
    order3 = []
    for r in range(3):
        for x in range(r, N, 3):
            order3.append(x)
    S3 = greedy_construct(order3)
    S3 = repair_4aps(S3)
    score3 = count_3aps_bitmask(S3)
    if score3 > best_score:
        best_score = score3
        best_S = S3

    # Ordering 3: random shuffle
    order5 = list(range(N))
    rng.shuffle(order5)
    S5 = greedy_construct(order5)
    S5 = repair_4aps(S5)
    score5 = count_3aps_bitmask(S5)
    if score5 > best_score:
        best_score = score5
        best_S = S5

    # === PHASE 2: Initial local search ===
    if time.time() - start_time < TIME_LIMIT - 20:
        S_ls, score_ls = local_search_2opt(best_S, max_iters=300)
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls

    # === PHASE 3: Parallel tempering with WalkSAT ===
    # Maintain multiple replicas at different temperatures
    num_replicas = 4
    # Temperatures: from low (greedy) to high (random)
    temperatures = [0.5, 1.0, 2.0, 4.0]
    
    # Initialize replicas
    replicas = []
    replica_scores = []
    for i in range(num_replicas):
        if i == 0:
            # Best solution so far
            S_rep = best_S
        else:
            # Diverse random starts
            order = list(range(N))
            rng.shuffle(order)
            S_rep = greedy_construct(order)
            S_rep = repair_4aps(S_rep)
        replicas.append(S_rep)
        replica_scores.append(count_3aps_bitmask(S_rep))
    
    # Track best across all replicas
    global_best_S = best_S
    global_best_score = best_score

    max_pt_iters = 2000
    swap_interval = 50  # swap replicas every 50 iterations

    for pt_iter in range(max_pt_iters):
        if time.time() - start_time > TIME_LIMIT - 5:
            break

        # Choose which replica to update (cycle through)
        rep_idx = pt_iter % num_replicas
        T = temperatures[rep_idx]
        
        S_cur = replicas[rep_idx]
        score_cur = replica_scores[rep_idx]

        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Single add
            non_members = [x for x in range(N) if not ((S_cur >> x) & 1)]
            if not non_members:
                continue
            x = rng.choice(non_members)
            if is_safe_add(S_cur, x):
                S_new = S_cur | (1 << x)
                score_new = score_cur + gain_3aps(S_cur, x)
            else:
                # Try with WalkSAT repair
                S_new = S_cur | (1 << x)
                S_new = walksat_repair(S_new, max_steps=50)
                score_new = count_3aps_bitmask(S_new)
        elif move_type < 0.5:
            # Single remove
            members = [x for x in range(N) if (S_cur >> x) & 1]
            if not members:
                continue
            x = rng.choice(members)
            S_new = S_cur & ~(1 << x)
            score_new = score_cur - loss_3aps(S_cur, x)
        elif move_type < 0.8:
            # Swap: remove one, add one
            members = [x for x in range(N) if (S_cur >> x) & 1]
            non_members = [x for x in range(N) if not ((S_cur >> x) & 1)]
            if not members or not non_members:
                continue
            x_remove = rng.choice(members)
            x_add = rng.choice(non_members)
            S_after_remove = S_cur & ~(1 << x_remove)
            if is_safe_add(S_after_remove, x_add):
                S_new = S_after_remove | (1 << x_add)
                loss = loss_3aps(S_cur, x_remove)
                gain = gain_3aps(S_after_remove, x_add)
                score_new = score_cur - loss + gain
            else:
                S_new = S_after_remove | (1 << x_add)
                S_new = walksat_repair(S_new, max_steps=50)
                score_new = count_3aps_bitmask(S_new)
        else:
            # Quantum tunneling: flip 2-3 random bits
            S_new = S_cur
            num_flips = rng.randint(2, 3)
            for _ in range(num_flips):
                x = rng.randint(0, N - 1)
                S_new ^= (1 << x)
            S_new = walksat_repair(S_new, max_steps=50)
            score_new = count_3aps_bitmask(S_new)

        # Accept/reject based on temperature
        if score_new >= score_cur:
            replicas[rep_idx] = S_new
            replica_scores[rep_idx] = score_new
            if score_new > global_best_score:
                global_best_score = score_new
                global_best_S = S_new
        else:
            # Metropolis acceptance
            delta = score_cur - score_new  # positive means worse
            accept_prob = math.exp(-delta / max(T, 0.1))
            if rng.random() < accept_prob:
                replicas[rep_idx] = S_new
                replica_scores[rep_idx] = score_new

        # Periodic swap between adjacent replicas (parallel tempering)
        if (pt_iter + 1) % swap_interval == 0:
            for i in range(num_replicas - 1):
                # Swap with some probability based on temperature difference
                if rng.random() < 0.5:
                    replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                    replica_scores[i], replica_scores[i+1] = replica_scores[i+1], replica_scores[i]

        # Periodic local search on best replica
        if (pt_iter + 1) % 100 == 0:
            best_rep_idx = max(range(num_replicas), key=lambda i: replica_scores[i])
            if replica_scores[best_rep_idx] >= global_best_score:
                S_ls, score_ls = local_search_2opt(replicas[best_rep_idx], max_iters=50)
                if score_ls > replica_scores[best_rep_idx]:
                    replicas[best_rep_idx] = S_ls
                    replica_scores[best_rep_idx] = score_ls
                    if score_ls > global_best_score:
                        global_best_score = score_ls
                        global_best_S = S_ls

    # Use the best found
    if global_best_score > best_score:
        best_score = global_best_score
        best_S = global_best_S

    # === PHASE 4: Final local search polishing ===
    if time.time() - start_time < TIME_LIMIT - 3:
        S_final, score_final = local_search_2opt(best_S, max_iters=200)
        if score_final > best_score:
            best_score = score_final
            best_S = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(best_S) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (best_S >> x) & 1]

    if not result:
        # Fallback: simple greedy
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result

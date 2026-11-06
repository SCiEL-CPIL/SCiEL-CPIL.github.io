"""MaxSAT-style WalkSAT for 4-AP-free set maximizing 3-AP count."""
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

    # For each element, which 4-APs does it participate in? Store other elements
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # For each element, which 3-APs does it participate in? Store other elements
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

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

    def greedy_construct():
        """Greedy: add elements in order of decreasing 3-AP weight, skip if creates 4-AP."""
        S = 0
        # Weight: number of 3-APs each element participates in
        weights = [len(elem_to_3aps[x]) for x in range(N)]
        order = sorted(range(N), key=lambda x: (-weights[x], x))
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs."""
        S = S
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    # Remove the element with lowest 3-AP weight
                    weights = [len(elem_to_3aps[x]) for x in (a, b, c, d)]
                    worst_idx = min(range(4), key=lambda i: weights[i])
                    worst = (a, b, c, d)[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def local_search(S, max_iters=200):
        """Hill-climbing: try adds, removes, swaps to improve 3-AP count while maintaining no 4-APs."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False

            # Try adding an element
            candidates = [x for x in range(N) if not ((current_S >> x) & 1)]
            rng.shuffle(candidates)
            for x in candidates:
                if not is_safe_add(current_S, x):
                    continue
                new_S = current_S | (1 << x)
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S = new_S
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try removing an element
            sel_list = [x for x in range(N) if (current_S >> x) & 1]
            rng.shuffle(sel_list)
            for x in sel_list:
                new_S = current_S & ~(1 << x)
                new_score = current_score - loss_3aps(current_S, x)
                if new_score > current_score:
                    current_S = new_S
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one, add another
            sel_list = [x for x in range(N) if (current_S >> x) & 1]
            rng.shuffle(sel_list)
            for x_remove in sel_list:
                if time.time() - start_time > TIME_LIMIT - 3:
                    return current_S, current_score
                candidates = [x for x in range(N) if not ((current_S >> x) & 1)]
                rng.shuffle(candidates)
                for x_add in candidates:
                    new_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                    # Check 4-AP violations for x_add in new_S without x_remove
                    if not is_safe_add(current_S & ~(1 << x_remove), x_add):
                        continue
                    new_score = current_score - loss_3aps(current_S, x_remove) + gain_3aps(current_S & ~(1 << x_remove), x_add)
                    if new_score > current_score:
                        current_S = new_S
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return current_S, current_score

    def walksat(S_init, time_limit):
        """WalkSAT-style MaxSAT solver.
        Hard clauses: for each 4-AP (a,b,c,d), clause (¬a ∨ ¬b ∨ ¬c ∨ ¬d) must be satisfied.
        Soft clauses: for each 3-AP (a,b,c), clause (a ∧ b ∧ c) contributes to score.
        We maximize soft score subject to all hard clauses being satisfied.
        """
        current_S = S_init
        current_3ap = count_3aps_bitmask(current_S)
        current_4ap = count_4aps_bitmask(current_S)
        # Ensure feasibility at start
        if current_4ap > 0:
            current_S = repair_4aps(current_S)
            current_3ap = count_3aps_bitmask(current_S)
            current_4ap = count_4aps_bitmask(current_S)

        best_S = current_S
        best_score = current_3ap

        T = 5.0
        alpha = 0.995

        while time.time() < time_limit:
            # Pick a variable to flip
            # Bias toward flipping variables involved in unsatisfied hard clauses or high-value soft clauses
            # With probability p, pick from 4-AP-conflict elements; else random
            if current_4ap > 0:
                # Find elements in violated 4-APs
                conflict_elems = set()
                for a, b, c, d in four_aps:
                    if (current_S >> a) & 1 and (current_S >> b) & 1 and (current_S >> c) & 1 and (current_S >> d) & 1:
                        conflict_elems.update([a, b, c, d])
                if conflict_elems:
                    x = rng.choice(list(conflict_elems))
                else:
                    x = rng.randint(0, N - 1)
            else:
                # All hard clauses satisfied. Pick element to flip.
                # With some probability, pick an element that would gain 3-APs if added
                # or that has high loss if removed
                if rng.random() < 0.5:
                    # Try to add a random non-member
                    non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
                    if non_members:
                        x = rng.choice(non_members)
                    else:
                        x = rng.randint(0, N - 1)
                else:
                    # Try to remove a random member
                    members = [x for x in range(N) if (current_S >> x) & 1]
                    if members:
                        x = rng.choice(members)
                    else:
                        x = rng.randint(0, N - 1)

            # Flip x
            if (current_S >> x) & 1:
                # Remove x
                new_S = current_S & ~(1 << x)
                delta_3ap = -loss_3aps(current_S, x)
            else:
                # Add x
                new_S = current_S | (1 << x)
                delta_3ap = gain_3aps(current_S, x)

            new_3ap = current_3ap + delta_3ap
            new_4ap = count_4aps_bitmask(new_S)

            # Score: 3-AP count minus large penalty for 4-AP violations
            penalty = 1000
            old_score = current_3ap - penalty * current_4ap
            new_score = new_3ap - penalty * new_4ap

            delta = new_score - old_score

            # Accept if improving, or with probability exp(delta/T)
            if delta >= 0 or (T > 0.01 and rng.random() < math.exp(delta / T)):
                current_S = new_S
                current_3ap = new_3ap
                current_4ap = new_4ap

                # Track best feasible solution
                if current_4ap == 0 and current_3ap > best_score:
                    best_score = current_3ap
                    best_S = current_S
            else:
                # Revert (do nothing)
                pass

            T *= alpha
            if T < 0.1:
                T = 0.1

        return best_S, best_score

    # Phase 1: Greedy construction
    S_greedy = greedy_construct()
    S_greedy = repair_4aps(S_greedy)
    score_greedy = count_3aps_bitmask(S_greedy)

    # Phase 2: Local search on greedy solution
    S_ls, score_ls = local_search(S_greedy, max_iters=300)
    if score_ls > score_greedy:
        S_best = S_ls
        best_score = score_ls
    else:
        S_best = S_greedy
        best_score = score_greedy

    # Phase 3: WalkSAT with multiple restarts
    deadline = start_time + TIME_LIMIT - 5
    num_restarts = 20

    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break

        # Generate diverse initial solutions
        if i == 0:
            S_init = S_best
        else:
            # Randomized greedy with different orderings
            S_init = 0
            order = list(range(N))
            rng.shuffle(order)
            for x in order:
                if is_safe_add(S_init, x):
                    S_init |= (1 << x)
            S_init = repair_4aps(S_init)

        # Allocate time for this restart
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        restart_time = start_time + min(time_left - 1.0, 3.0)

        S_walk, score_walk = walksat(S_init, restart_time)

        if score_walk > best_score:
            best_score = score_walk
            S_best = S_walk

    # Final local search on best solution
    if time.time() < deadline - 1:
        S_final, score_final = local_search(S_best, max_iters=100)
        if score_final > best_score:
            best_score = score_final
            S_best = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(S_best) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (S_best >> x) & 1]

    if not result:
        # Fallback
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result

"""Two Sidon sets with disjoint difference sets – Bitset + multi-strategy restarts + local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before 90s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    s2 = elem + elem
    if s2 <= 512 and ((sum_bits >> s2) & 1):
        return False
    for a in elems:
        s = elem + a
        if s <= 512 and ((sum_bits >> s) & 1):
            return False
        d = abs(elem - a)
        if d > 0:
            if ((diff_bits >> d) & 1) or ((other_diff_bits >> d) & 1):
                return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side. state = [elems0, elems1, sum_bits0, sum_bits1, diff_bits0, diff_bits1]"""
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]

    elems.append(elem)
    for a in elems[:-1]:
        s = elem + a
        if s <= 512:
            sum_bits |= (1 << s)
    s2 = elem + elem
    if s2 <= 512:
        sum_bits |= (1 << s2)
    for a in elems[:-1]:
        d = abs(elem - a)
        if d > 0:
            diff_bits |= (1 << d)

    state[side] = elems
    state[2 + side] = sum_bits
    state[4 + side] = diff_bits


def _remove_elem(state, side, elem):
    """Remove elem from side and rebuild bits."""
    elems = state[side]
    idx = elems.index(elem)
    elems.pop(idx)

    new_sum_bits = 0
    new_diff_bits = 0
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            s = elems[i] + elems[j]
            if s <= 512:
                new_sum_bits |= (1 << s)
            if i != j:
                d = abs(elems[i] - elems[j])
                if d > 0:
                    new_diff_bits |= (1 << d)

    state[side] = elems
    state[2 + side] = new_sum_bits
    state[4 + side] = new_diff_bits


def _greedy_build(deadline, order=None):
    """Greedy construction over a given order."""
    if order is None:
        order = list(range(1, N + 1))
    state = [[], [], 0, 0, 0, 0]
    for x in order:
        if time.time() > deadline:
            break
        for side in (0, 1):
            if len(state[side]) >= MAX_EACH:
                continue
            elems = state[side]
            sum_bits = state[2 + side]
            diff_bits = state[4 + side]
            other_diff_bits = state[4 + (1 - side)]
            if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
                _add_elem(state, side, x)
    return state


def _greedy_extend(state, side, deadline):
    """Extend one side greedily with remaining candidates."""
    if len(state[side]) >= MAX_EACH:
        return
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]
    other_diff_bits = state[4 + (1 - side)]
    in_set = set(elems)
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x in in_set:
            continue
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            _add_elem(state, side, x)
            in_set.add(x)


def _local_search(state, deadline):
    """1-for-1 swap local search to escape local minima."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for side in (0, 1):
            if time.time() > deadline:
                break
            elems = state[side]
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                if time.time() > deadline:
                    break
                elem_to_remove = elems[i]
                _remove_elem(state, side, elem_to_remove)
                elems_after = state[side]
                sum_bits = state[2 + side]
                diff_bits = state[4 + side]
                other_diff_bits = state[4 + (1 - side)]

                best_new_elem = None
                for x in range(1, N + 1):
                    if x in elems_after:
                        continue
                    if _can_add(x, elems_after, sum_bits, diff_bits, other_diff_bits):
                        best_new_elem = x
                        break

                if best_new_elem is not None and best_new_elem != elem_to_remove:
                    _add_elem(state, side, best_new_elem)
                    improved = True
                else:
                    _add_elem(state, side, elem_to_remove)
    return state


def _try_add_random(state, side, rng, deadline):
    """Try to add a random element to side; return True if it grew."""
    if len(state[side]) >= MAX_EACH:
        return False
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]
    other_diff_bits = state[4 + (1 - side)]
    for _ in range(15):
        if time.time() > deadline:
            return False
        x = rng.randint(1, N)
        if x in elems:
            continue
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            _add_elem(state, side, x)
            return True
    return False


def _multi_start(deadline, rng):
    """Run multiple greedy strategies with different orderings, then local search each."""
    best_state = None
    best_score = -1

    num_starts = 40
    for start in range(num_starts):
        if time.time() > deadline:
            break

        state = [[], [], 0, 0, 0, 0]
        strat = start % 4

        if strat == 0:
            # deterministic ascending
            order = list(range(1, N + 1))
            for x in order:
                if time.time() > deadline:
                    break
                for side in (0, 1):
                    if len(state[side]) >= MAX_EACH:
                        continue
                    elems = state[side]
                    sum_bits = state[2 + side]
                    diff_bits = state[4 + side]
                    other_diff_bits = state[4 + (1 - side)]
                    if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
                        _add_elem(state, side, x)
        elif strat == 1:
            # seed A, extend A, extend B
            seed_a = rng.randint(1, N)
            _add_elem(state, 0, seed_a)
            _greedy_extend(state, 0, deadline)
            _greedy_extend(state, 1, deadline)
        elif strat == 2:
            # seed B, extend B, extend A
            seed_b = rng.randint(1, N)
            _add_elem(state, 1, seed_b)
            _greedy_extend(state, 1, deadline)
            _greedy_extend(state, 0, deadline)
        else:
            # random small valid sets, then extend both
            size_a = rng.randint(1, 4)
            size_b = rng.randint(1, 4)
            A_init = sorted(rng.sample(range(1, N + 1), size_a))
            B_init = sorted(rng.sample(range(1, N + 1), size_b))
            ok = True
            for x in A_init:
                if not _can_add(x, state[0], state[2], state[4], state[5]):
                    ok = False
                    break
                _add_elem(state, 0, x)
            if ok:
                for x in B_init:
                    if not _can_add(x, state[1], state[3], state[5], state[4]):
                        ok = False
                        break
                    _add_elem(state, 1, x)
            if not ok:
                continue
            _greedy_extend(state, 0, deadline)
            _greedy_extend(state, 1, deadline)

        if not state[0]:
            _add_elem(state, 0, 1)
        if not state[1]:
            _add_elem(state, 1, 2)

        # local search
        state = _local_search(state, deadline)

        # a few random add attempts to grow further
        for _ in range(6):
            _try_add_random(state, 0, rng, deadline)
            _try_add_random(state, 1, rng, deadline)

        score = len(state[0]) + len(state[1])
        if score > best_score:
            best_score = score
            best_state = [state[0][:], state[1][:], state[2], state[3], state[4], state[5]]

    return best_state, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random()

    # Quick greedy build for a valid baseline
    state = _greedy_build(deadline - 10.0)
    best_state = [state[0][:], state[1][:], state[2], state[3], state[4], state[5]]
    best_score = len(state[0]) + len(state[1])

    # Multi-strategy search with remaining time
    if time.time() < deadline - 8.0:
        search_deadline = deadline - 8.0
        state2, score2 = _multi_start(search_deadline, rng)
        if state2 is not None and score2 > best_score:
            best_score = score2
            best_state = state2

    # Final polish: try to grow the best solution with random adds + local search
    if time.time() < deadline - 2.0:
        state3 = [best_state[0][:], best_state[1][:], best_state[2], best_state[3], best_state[4], best_state[5]]
        for _ in range(12):
            _try_add_random(state3, 0, rng, deadline - 2.0)
            _try_add_random(state3, 1, rng, deadline - 2.0)
        state3 = _local_search(state3, deadline - 2.0)
        score3 = len(state3[0]) + len(state3[1])
        if score3 > best_score:
            best_score = score3
            best_state = state3

    return [best_state[0], best_state[1]]
# EVOLVE-BLOCK-END

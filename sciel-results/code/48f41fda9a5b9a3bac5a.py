"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    # Check elem+elem sum
    s2 = elem + elem
    if (sum_bits >> s2) & 1:
        return False
    # Check elem+a sums and |elem-a| diffs
    for a in elems:
        s = elem + a
        if (sum_bits >> s) & 1:
            return False
        d = abs(elem - a)
        if (diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
            return False
    return True


def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits from scratch for a set."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits


def _add_elem(state, side, elem):
    """Add elem to side; update bitsets. Returns None if invalid."""
    elems = state['sets'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    other_diff = state['diff_bits'][1 - side]

    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff):
        return None

    # Update sum bits
    s2 = elem + elem
    sum_bits |= (1 << s2)
    for a in elems:
        sum_bits |= (1 << (elem + a))

    # Update diff bits
    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)

    state['sets'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    return state


def _remove_elem(state, side, idx):
    """Remove element at index idx from side; rebuild bitsets."""
    elems = state['sets'][side]
    elem = elems.pop(idx)
    new_sum, new_diff = _rebuild_bits(elems)
    state['sum_bits'][side] = new_sum
    state['diff_bits'][side] = new_diff
    return elem


def _number_theoretic_seed():
    """Try to construct a good initial seed using number-theoretic methods.
    
    Use the fact that if p is prime, {x^2 mod p : x in some subset} has good
    Sidon-like properties. We'll use a simple approach: try to find large Sidon
    sets in [1, 256] using greedy with a number-theoretic bias.
    """
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    
    # Strategy: use quadratic residues mod a prime to bias the order
    # For p=257 (Fermat prime), QRs are x^2 mod 257 for x=1..256
    p = 257
    qr_set = set()
    for x in range(1, p):
        qr_set.add((x * x) % p)
    # Map to [1, 256]: QRs mod 257 are in {0, 1, ..., 256}, remove 0, keep 1-256
    qr_list = sorted([x for x in qr_set if 1 <= x <= 256])
    
    # Build A from QRs, B from non-QRs (complement), with balancing
    non_qr_list = [x for x in range(1, 257) if x not in qr_set]
    
    # Interleave: try adding QRs to A and non-QRs to B alternately
    qr_idx = 0
    non_qr_idx = 0
    for elem in range(1, 257):
        # Try to add to whichever set is smaller
        if len(state['sets'][0]) <= len(state['sets'][1]):
            sides = [0, 1]
        else:
            sides = [1, 0]
        for side in sides:
            if len(state['sets'][side]) < MAX_EACH:
                if _add_elem(state, side, elem) is not None:
                    break
    
    return state


def _greedy_build(order, max_time):
    """Greedy build using given element order, balancing set sizes."""
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    for elem in order:
        if time.time() > max_time:
            break
        # Try adding to the smaller set first to balance
        if len(state['sets'][0]) <= len(state['sets'][1]):
            sides = [0, 1]
        else:
            sides = [1, 0]
        for side in sides:
            if len(state['sets'][side]) < MAX_EACH:
                r = _add_elem(state, side, elem)
                if r is not None:
                    break
    return state


def _get_unused(state):
    """Get set of unused elements."""
    return set(range(1, N + 1)) - (set(state['sets'][0]) | set(state['sets'][1]))


def _local_search(state, max_time):
    """Local search: try additions, 1-for-1 swaps, and 1-for-2 swaps."""
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for side in (0, 1):
            if time.time() > max_time:
                return state
            # Phase 1: Try adding a new element
            if len(state['sets'][side]) < MAX_EACH:
                unused = _get_unused(state)
                for candidate in sorted(unused):
                    if time.time() > max_time:
                        return state
                    r = _add_elem(state, side, candidate)
                    if r is not None:
                        improved = True
                        break

            # Phase 2: Try 1-for-1 swap (replace one element with another)
            if len(state['sets'][side]) >= 1:
                elems = state['sets'][side]
                for i in range(len(elems)):
                    if time.time() > max_time:
                        return state
                    removed = _remove_elem(state, side, i)
                    unused = _get_unused(state)
                    swapped = False
                    for candidate in sorted(unused):
                        if candidate == removed:
                            continue
                        if time.time() > max_time:
                            break
                        r = _add_elem(state, side, candidate)
                        if r is not None:
                            swapped = True
                            improved = True
                            break
                    if not swapped:
                        # Restore
                        _add_elem(state, side, removed)

            # Phase 3: Try 1-for-2 swap (remove one, add two)
            if len(state['sets'][side]) >= 1 and len(state['sets'][side]) < MAX_EACH:
                elems = state['sets'][side]
                for i in range(len(elems)):
                    if time.time() > max_time:
                        return state
                    removed = _remove_elem(state, side, i)
                    added = 0
                    unused = _get_unused(state)
                    for candidate in sorted(unused):
                        if candidate == removed:
                            continue
                        if time.time() > max_time:
                            break
                        r = _add_elem(state, side, candidate)
                        if r is not None:
                            added += 1
                            if added >= 2:
                                improved = True
                                break
                    if added < 2:
                        # Restore removed element
                        _add_elem(state, side, removed)
    return state


def _polish(state, max_time):
    """Final polish: exhaustively try all additions and swaps until no improvement."""
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for side in (0, 1):
            if time.time() > max_time:
                return state
            # Try all possible additions
            if len(state['sets'][side]) < MAX_EACH:
                unused = _get_unused(state)
                for candidate in sorted(unused):
                    if time.time() > max_time:
                        return state
                    r = _add_elem(state, side, candidate)
                    if r is not None:
                        improved = True
                        break
            # Try all 1-for-1 swaps
            if len(state['sets'][side]) >= 1:
                elems = state['sets'][side]
                for i in range(len(elems)):
                    if time.time() > max_time:
                        return state
                    removed = _remove_elem(state, side, i)
                    unused = _get_unused(state)
                    swapped = False
                    for candidate in sorted(unused):
                        if candidate == removed:
                            continue
                        if time.time() > max_time:
                            break
                        r = _add_elem(state, side, candidate)
                        if r is not None:
                            swapped = True
                            improved = True
                            break
                    if not swapped:
                        _add_elem(state, side, removed)
    return state


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_state = None
    best_size = 0

    # First try number-theoretic seed
    nt_state = _number_theoretic_seed()
    nt_size = len(nt_state['sets'][0]) + len(nt_state['sets'][1])
    if nt_size > best_size:
        best_size = nt_size
        best_state = {
            'sets': [list(nt_state['sets'][0]), list(nt_state['sets'][1])],
            'sum_bits': [nt_state['sum_bits'][0], nt_state['sum_bits'][1]],
            'diff_bits': [nt_state['diff_bits'][0], nt_state['diff_bits'][1]]
        }

    # Multi-start with different random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 8.0:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)

        state = _greedy_build(order, deadline - 5.0)
        state = _local_search(state, deadline - 2.0)

        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_size = size
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
                'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
                'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
            }

        if best_size >= 2 * MAX_EACH:
            break

    # Fallback: simple greedy if nothing found
    if best_state is None:
        order = list(range(1, N + 1))
        state = _greedy_build(order, deadline - 1.0)
        best_state = {
            'sets': [list(state['sets'][0]), list(state['sets'][1])],
            'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
            'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
        }

    # Final polish on the best state found
    if time.time() < deadline - 0.5:
        polish_state = {
            'sets': [list(best_state['sets'][0]), list(best_state['sets'][1])],
            'sum_bits': [best_state['sum_bits'][0], best_state['sum_bits'][1]],
            'diff_bits': [best_state['diff_bits'][0], best_state['diff_bits'][1]]
        }
        polish_state = _polish(polish_state, deadline - 0.2)
        polish_size = len(polish_state['sets'][0]) + len(polish_state['sets'][1])
        if polish_size > best_size:
            best_state = polish_state
            best_size = polish_size

    return [best_state['sets'][0], best_state['sets'][1]]
# EVOLVE-BLOCK-END

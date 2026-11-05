import random
import time
from itertools import combinations

def _build_conflicts():
    """
    Precompute conflict bitsets for all 192 numbers.
    For each number a (1..192):
      conflict[a] is a Python int bitset where bit i is set if number (i+1)
      participates in a forbidden configuration with a.
    A number b is in conflict[a] if:
      (a) there exist distinct c,d in 1..192 with c,d != a,b and a | (b+c+d)
      (b) there exist distinct c,d in 1..192 with c,d != a,b and b | (a+c+d)
    During SA, when the state is a bitset S, adding x is valid iff
      (conflict[x] & S) == 0
    Removing x is always valid.
    """
    N = 192
    nums = list(range(1, N + 1))
    num_set = set(nums)
    conflicts = [0] * (N + 1)  # 1-indexed

    # For each a, find all b that conflict with a
    for a in nums:
        # conflict[a] starts as 0
        # Condition (a): a divides (b+c+d) for distinct c,d != a,b
        # Condition (b): b divides (a+c+d) for distinct c,d != a,b
        # We'll iterate over all triples (c,d,e) distinct, and for each:
        #   if a | (c+d+e): then a conflicts with c, d, e (each as the "tested" b)
        #   Actually: a conflicts with b if a|(b+c+d) for some c,d distinct from a,b
        #   So for a fixed a, for each pair (c,d) distinct from a:
        #     if a | (c+d+b) for some b distinct from a,c,d: mark b in conflict[a]
        #     if b | (a+c+d) for some b: mark b in conflict[a]  -- this is condition (b)
        #
        # More efficient: iterate over all unordered pairs (c,d) with c<d, c,d != a
        #   sum_cd = c+d
        #   Condition (a): find b != a,c,d such that (b + sum_cd) % a == 0
        #     b ≡ -sum_cd (mod a), b in 1..192, b != a, b != c, b != d
        #   Condition (b): find b != a,c,d such that b | (a + sum_cd)
        #     b is a divisor of (a+sum_cd), b in 1..192, b != a, b != c, b != d
        #
        # This is O(N^2 * N) total which is ~7M, acceptable.

        for c in range(1, N + 1):
            if c == a:
                continue
            for d in range(c + 1, N + 1):
                if d == a:
                    continue
                s_cd = c + d

                # Condition (a): a | (b + s_cd) for some b distinct from a,c,d
                # b ≡ -s_cd (mod a)
                rem = (-s_cd) % a
                # b values: rem, rem+a, rem+2a, ... up to 192
                # Note: if rem == 0, b must be a multiple of a: a, 2a, ...
                if rem == 0:
                    candidates_a = list(range(a, N + 1, a))
                else:
                    candidates_a = list(range(rem, N + 1, a))
                for b in candidates_a:
                    if b == a or b == c or b == d:
                        continue
                    conflicts[a] |= (1 << b)

                # Condition (b): b | (a + s_cd) for some b distinct from a,c,d
                total = a + s_cd
                # Find all divisors of total in 1..192
                # b must divide total
                for b in range(1, N + 1):
                    if b == a or b == c or b == d:
                        continue
                    if total % b == 0:
                        conflicts[a] |= (1 << b)

    return conflicts

def _is_valid_bitset(state, conflicts):
    """
    Verify that a state (bitset) is valid: no element in state conflicts with
    any other element in state.
    For each element x in state, check (conflicts[x] & state) == 0.
    But note: conflict[x] includes y if x and y are in a forbidden config.
    So we need: for all x in state, (conflicts[x] & (state & ~(1<<x))) == 0.
    Actually conflicts[x] should not include x itself, so just (conflicts[x] & state) == 0.
    """
    # Iterate over set bits in state
    s = state
    while s:
        # Get lowest set bit
        lsb = s & (-s)
        x = lsb.bit_length()  # 1-indexed: bit_length of (1<<x) is x+1, so x = bit_length - 1
        # Wait: if lsb = 1<<x, then lsb.bit_length() = x+1, so x = lsb.bit_length() - 1
        x = lsb.bit_length() - 1
        if x < 1 or x > 192:
            s ^= lsb
            continue
        if conflicts[x] & state:
            return False
        s ^= lsb
    return True

def _count_bits(state):
    return bin(state).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin for 90s limit
    rng = random.Random(42)

    # Precompute conflicts
    conflicts = _build_conflicts()

    # Helper: check if adding x to state is valid
    def can_add(state, x):
        return (conflicts[x] & state) == 0

    # Helper: get list of elements in state
    def state_to_list(state):
        result = []
        s = state
        while s:
            lsb = s & (-s)
            result.append(lsb.bit_length() - 1)
            s ^= lsb
        return result

    # Helper: list to state
    def list_to_state(lst):
        s = 0
        for x in lst:
            s |= (1 << x)
        return s

    # Greedy construction to get initial state
    def greedy():
        state = 0
        # Try multiple orderings
        best_state = 0
        best_size = 0

        # Descending
        for x in range(192, 0, -1):
            if can_add(state, x):
                state |= (1 << x)
        sz = _count_bits(state)
        if sz > best_size:
            best_size = sz
            best_state = state

        # Ascending
        state = 0
        for x in range(1, 193):
            if can_add(state, x):
                state |= (1 << x)
        sz = _count_bits(state)
        if sz > best_size:
            best_size = sz
            best_state = state

        # Randomized greedy
        for trial in range(50):
            state = 0
            nums = list(range(1, 193))
            rng.shuffle(nums)
            for x in nums:
                if can_add(state, x):
                    state |= (1 << x)
            sz = _count_bits(state)
            if sz > best_size:
                best_size = sz
                best_state = state

        return best_state, best_size

    # Simulated Annealing
    def simulated_annealing(init_state, init_size, max_iters=20000):
        state = init_state
        cur_size = init_size
        best_state = init_state
        best_size = init_size

        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))

        T = T_start
        stall = 0

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Choose a random move: add, remove, or swap
            move_type = rng.random()
            improved = False

            if move_type < 0.3:
                # Add a random element not in state
                not_in_state = [x for x in range(1, 193) if not (state & (1 << x))]
                if not not_in_state:
                    continue
                x = rng.choice(not_in_state)
                new_state = state | (1 << x)
                new_size = cur_size + 1
                # Check validity: x must not conflict with any in state
                if can_add(state, x):
                    state = new_state
                    cur_size = new_size
                    improved = True
                else:
                    # Invalid, reject
                    pass

            elif move_type < 0.6:
                # Remove a random element from state
                in_state = [x for x in range(1, 193) if (state & (1 << x))]
                if not in_state:
                    continue
                x = rng.choice(in_state)
                new_state = state & ~(1 << x)
                new_size = cur_size - 1
                # Always valid to remove
                # Accept if better or with probability exp(-delta/T)
                delta = new_size - cur_size  # -1
                if delta >= 0 or (T > 0 and rng.random() < (2.718281828 ** (delta / T))):
                    state = new_state
                    cur_size = new_size
                    improved = True

            else:
                # Swap: remove one, add another
                in_state = [x for x in range(1, 193) if (state & (1 << x))]
                not_in_state = [x for x in range(1, 193) if not (state & (1 << x))]
                if not in_state or not not_in_state:
                    continue
                x = rng.choice(in_state)
                y = rng.choice(not_in_state)
                new_state = (state & ~(1 << x)) | (1 << y)
                new_size = cur_size  # same size
                # Check: y must not conflict with (state without x)
                state_without_x = state & ~(1 << x)
                if can_add(state_without_x, y):
                    state = new_state
                    cur_size = new_size
                    improved = True

            if improved:
                if cur_size > best_size:
                    best_size = cur_size
                    best_state = state
                stall = 0
            else:
                stall += 1
                if stall > 5000:
                    break

            T *= cooling

        return best_state, best_size

    # Local search: greedy add, then swaps
    def local_search(state, size):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding
            for x in range(192, 0, -1):
                if not (state & (1 << x)) and can_add(state, x):
                    state |= (1 << x)
                    size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try swaps
            in_state = sorted([x for x in range(1, 193) if (state & (1 << x))])
            not_in_state = sorted([x for x in range(1, 193) if not (state & (1 << x))], reverse=True)
            for x in in_state:
                for y in not_in_state[:20]:
                    state_without_x = state & ~(1 << x)
                    if can_add(state_without_x, y):
                        new_state = state_without_x | (1 << y)
                        # Check full validity
                        if _is_valid_bitset(new_state, conflicts):
                            state = new_state
                            improved = True
                            break
                if improved:
                    break
        return state, size

    # Run initial greedy
    best_state, best_size = greedy()

    # Local search on best
    best_state, best_size = local_search(best_state, best_size)

    # Run SA with multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            init_state = best_state
            init_size = best_size
        else:
            # Perturb: start from best, remove some, add random
            init_state = best_state
            init_size = best_size
            # Remove 5-15 random elements
            in_state = [x for x in range(1, 193) if (init_state & (1 << x))]
            n_remove = rng.randint(5, min(15, len(in_state)))
            to_remove = rng.sample(in_state, n_remove)
            for x in to_remove:
                init_state &= ~(1 << x)
                init_size -= 1
            # Add some random
            not_in = [x for x in range(1, 193) if not (init_state & (1 << x))]
            n_add = rng.randint(0, min(10, len(not_in)))
            to_add = rng.sample(not_in, n_add)
            for x in to_add:
                if can_add(init_state, x):
                    init_state |= (1 << x)
                    init_size += 1

        # SA
        sa_state, sa_size = simulated_annealing(init_state, init_size, max_iters=5000)

        # Local search on SA result
        sa_state, sa_size = local_search(sa_state, sa_size)

        if sa_size > best_size:
            best_size = sa_size
            best_state = sa_state

        if best_size >= 48:
            break

    # Final local search
    best_state, best_size = local_search(best_state, best_size)

    # Convert to list
    result = state_to_list(best_state)

    # Verify validity as a safety check
    if not _is_valid_bitset(best_state, conflicts):
        # Fallback: rebuild greedily
        state = 0
        for x in range(192, 0, -1):
            if can_add(state, x):
                state |= (1 << x)
        result = state_to_list(state)

    return result

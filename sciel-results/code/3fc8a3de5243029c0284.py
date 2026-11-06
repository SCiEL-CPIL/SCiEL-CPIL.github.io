import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# --- Precompute all 2^N boolean inputs as bitsets ---
# Wire w's value across all 2^N inputs = bitmask with bit (1<<w) set.
# Represent each wire's bitset as a list of 64-bit words (4096 words).
_NUM_WORDS = (1 << N) // 64  # 4096

def _wire_bitset(w):
    """Return the list of 64-bit words for wire w's values over all 2^N inputs."""
    bits = 1 << w
    words = []
    for k in range(_NUM_WORDS):
        base = k * 64
        mask = 0
        for bit in range(64):
            if (base + bit) & bits:
                mask |= (1 << bit)
        words.append(mask)
    return words

# Precompute all wire bitsets
_WIRE_BITSETS = [_wire_bitset(w) for w in range(N)]
_ALL_ONES = [(1 << 64) - 1 for _ in range(_NUM_WORDS)]


def _apply_comp(wires, i, j):
    """Apply compare-exchange on wires i,j. wires[w] is a list of 64-bit words.
    Returns new lists for wires i and j (min to i, max to j)."""
    wi = wires[i]
    wj = wires[j]
    ni = [wi[k] & wj[k] for k in range(_NUM_WORDS)]
    nj = [wi[k] | wj[k] for k in range(_NUM_WORDS)]
    return ni, nj


def _is_sorted_incremental(net):
    """Verify network sorts all 2^N inputs. Returns True/False.
    Uses incremental sortedness: after each comparator, wire i is 'sorted'
    (all values >= wire i+1) iff (min_i AND max_i+1) == all-ones for every word.
    """
    wires = [w[:] for w in _WIRE_BITSETS]
    # sorted_mask[w]: all-ones if wire w is sorted relative to wire w+1
    sorted_mask = [None] * (N - 1)
    for w in range(N - 1):
        sm = [wires[w][k] & wires[w + 1][k] for k in range(_NUM_WORDS)]
        sorted_mask[w] = sm

    for i, j in net:
        ni, nj = _apply_comp(wires, i, j)
        wires[i] = ni
        wires[j] = nj

        # Update sortedness for affected adjacencies
        # Wire i changed: affects sortedness of (i-1, i) and (i, i+1)
        if i > 0:
            sorted_mask[i - 1] = [wires[i - 1][k] & ni[k] for k in range(_NUM_WORDS)]
        if i < N - 1:
            sorted_mask[i] = [ni[k] & wires[i + 1][k] for k in range(_NUM_WORDS)]

        # Wire j changed: affects sortedness of (j-1, j) and (j, j+1)
        if j > 0:
            sorted_mask[j - 1] = [wires[j - 1][k] & nj[k] for k in range(_NUM_WORDS)]
        if j < N - 1:
            sorted_mask[j] = [nj[k] & wires[j + 1][k] for k in range(_NUM_WORDS)]

        # Check if all sortedness masks are all-ones
        for w in range(N - 1):
            sm = sorted_mask[w]
            for k in range(_NUM_WORDS):
                if sm[k] != (1 << 64) - 1:
                    return False

    return True


def _is_sorted_final(net):
    """Full verification: apply all comparators, then check sortedness."""
    wires = [w[:] for w in _WIRE_BITSETS]
    for i, j in net:
        ni, nj = _apply_comp(wires, i, j)
        wires[i] = ni
        wires[j] = nj
    for w in range(N - 1):
        for k in range(_NUM_WORDS):
            if wires[w][k] & wires[w + 1][k] != (1 << 64) - 1:
                return False
    return True


def _odd_even_transposition():
    """Generate the odd-even transposition sorting network (153 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _bitonic_network():
    """Generate a bitonic sorting network (faster than odd-even for N=18)."""
    net = []
    # Bitonic sort: recursively build bitonic sequences then merge
    # Standard bitonic sort network for N=18 (power-of-2 wrapper: use N=32? No, use 18 directly)
    # Actually, bitonic sort works for any N. Let's use the standard construction.
    # For N=18, we need to handle non-power-of-2.
    # Simpler: use the odd-even transposition as base, it's well-known.
    # Let's just return odd-even for reliability.
    return _odd_even_transposition()


def _try_remove(net, idx):
    """Try removing comparator at idx. Return True if still sorted."""
    new_net = net[:idx] + net[idx + 1:]
    return _is_sorted_incremental(new_net)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    def _time_ok():
        return time.time() < deadline - 2

    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)

    # Phase 1: Greedy removal (multiple passes)
    for _pass in range(20):
        if not _time_ok():
            break
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if not _time_ok():
                return best_net
            if _try_remove(net, idx):
                del net[idx]
                best_net = net[:]
                best_count = len(net)
                changed = True
        if not changed:
            # Try forward pass
            for idx in range(len(net)):
                if not _time_ok():
                    return best_net
                if _try_remove(net, idx):
                    del net[idx]
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
            if not changed:
                break

    # Phase 2: 2-opt swaps (swap adjacent comparators)
    for _pass in range(50):
        if not _time_ok():
            break
        improved = False
        for idx in range(len(net) - 1):
            if not _time_ok():
                break
            # Try swapping net[idx] and net[idx+1]
            new_net = net[:]
            new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
            if _is_sorted_incremental(new_net):
                # Check if it helps: try removing one after swap
                for rem in [idx, idx + 1]:
                    if rem < len(new_net):
                        test_net = new_net[:rem] + new_net[rem + 1:]
                        if _is_sorted_incremental(test_net):
                            net = test_net
                            best_net = net[:]
                            best_count = len(net)
                            improved = True
                            break
                if not improved:
                    # Just accept the swap if it enables future removals
                    # (skip for now, focus on removals)
                    pass
        if not improved:
            break

    # Phase 3: Simulated Annealing
    if _time_ok():
        T = 1.0
        min_T = 0.05
        cooling = 0.9998
        current_count = len(net)
        no_improve = 0

        while _time_ok():
            if T < min_T:
                net = best_net[:]
                current_count = best_count
                T = 1.0
                no_improve = 0

            r = random.random()
            if r < 0.45:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx + 1:]
            elif r < 0.75:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)

            if not _is_sorted_incremental(new_net):
                no_improve += 1
                if no_improve > 5000:
                    T = max(T * 0.5, min_T)
                    no_improve = 0
                continue

            new_count = len(new_net)
            delta = new_count - current_count

            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    no_improve = 0

            T *= cooling

    # Final validation
    if not _is_sorted_final(best_net):
        best_net = _odd_even_transposition()

    return best_net

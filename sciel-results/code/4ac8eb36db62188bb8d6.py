import time
import numpy as np

N = 18
M = 1 << N            # 262144 Boolean inputs
DEADLINE = 90.0
SAFE = 10.0           # seconds of safety margin before the hard kill


def _bitonic_comparators(n):
    """Recursive bitonic comparator list for n (power of two) wires."""
    if n == 1:
        return []
    h = n // 2
    A = _bitonic_comparators(h)
    B = _bitonic_comparators(h)
    comps = list(A)
    comps += [(a, a + h) for a in range(h)]
    comps += [(a + h, a) for a in range(h)]
    comps += [(a, a + h) for a in range(h)]
    comps += [(a + h, a) for a in range(h)]
    comps += [ (a, b) for a, b in A ]
    comps += [ (a + h, b + h) for a, b in B ]
    return comps


def _make_bitonic_network():
    """Bitonic network on 16 wires + merge wires 16,17."""
    comps = _bitonic_comparators(16)
    # merge the two extra wires into the sorted 16-wire network
    # insert wire 16 and 17 by comparing against the whole 16-block
    comps += [(16, 17)]
    # now bring 16,17 into the 16-wire sorted structure via a small merge
    # compare 16 against 0..15 and 17 against 0..15 (upper triangle merge)
    for i in range(16):
        comps += [(i, 16)]
        comps += [(i, 17)]
    # re-sort the 16-wire block after the merges
    comps += _bitonic_comparators(16)
    # final compare the two top wires
    comps += [(16, 17)]
    return comps


def _verify_bitmatrix(net):
    """Return True if the network sorts all Boolean inputs.

    State matrix S: shape (N, M), S[w, b] = bit w of input b after the
    network. Correct iff for every k=0..N-1, the set of inputs where wires
    0..k are all 1 equals the set where the top k+1 inputs are 1.
    """
    S = np.zeros((N, M), dtype=np.int8)
    for w in range(N):
        S[w, :] = (np.arange(M, dtype=np.int32) >> w) & 1

    for (i, j) in net:
        row_i = S[i]
        row_j = S[j]
        mn = np.minimum(row_i, row_j)
        mx = np.maximum(row_i, row_j)
        S[i] = mn
        S[j] = mx

    # check prefix invariant: for k in 0..N-1
    # all(k wires 0..k are 1) == all(k+1 highest inputs are 1)
    cum = np.ones(M, dtype=np.int8)
    for k in range(N):
        cum = cum & S[k]
        # target: inputs >= (2**(N-k-1)) have top k+1 bits all 1
        target = (np.arange(M, dtype=np.int32) >> (N - k - 1)) & 1
        if not np.array_equal(cum, target):
            return False
    return True


def _verify_net(net):
    return _verify_bitmatrix(net)


def _random_permutation(n):
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = (i * 2654435761) % (i + 1)  # deterministic-ish shuffle
        p[i], p[j] = p[j], p[i]
    return p


def construct():
    t0 = time.time()

    # Start from a correct network (bitonic-based).
    best = _make_bitonic_network()
    best_ok = _verify_net(best)
    if not best_ok:
        # Fallback to guaranteed-correct odd-even transposition.
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_ok = True

    best_len = len(best)
    cur = list(best)
    cur_len = best_len

    rng_state = 12345
    def rnd():
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        return rng_state

    def rnd_below(b):
        return rnd() % b

    def try_move(net, move):
        # move: ('remove', idx), ('swap', a, b), ('insert', (i,j), pos)
        kind = move[0]
        if kind == 'remove':
            idx = move[1]
            if idx < 0 or idx >= len(net):
                return None
            cand = net[:idx] + net[idx + 1:]
            return cand
        if kind == 'swap':
            a, b = move[1], move[2]
            if a < 0 or b < 0 or a >= len(net) or b >= len(net) or a == b:
                return None
            cand = list(net)
            cand[a], cand[b] = cand[b], cand[a]
            return cand
        if kind == 'insert':
            pair, pos = move[1], move[2]
            if not (0 <= pair[0] < pair[1] < N):
                return None
            pos = max(0, min(pos, len(net)))
            cand = list(net)
            cand.insert(pos, pair)
            return cand
        return None

    # Simulated-annealing local search.
    best_len_now = best_len
    temp = 0.05
    cooldown = 0.9999
    min_temp = 1e-4
    no_improve = 0
    max_no_improve = 600

    while True:
        if time.time() - t0 > DEADLINE - SAFE:
            break
        L = len(cur)
        r = rnd_below(100)
        if r < 55:
            # removal
            idx = rnd_below(L)
            cand = try_move(cur, ('remove', idx))
        elif r < 85:
            # swap
            a = rnd_below(L)
            b = rnd_below(L)
            cand = try_move(cur, ('swap', a, b))
        else:
            # insert a random comparator
            i = rnd_below(N - 1)
            j = i + 1 + rnd_below(N - 1 - i)
            pos = rnd_below(L + 1)
            cand = try_move(cur, ('insert', (i, j), pos))
        if cand is None:
            continue
        ok = _verify_net(cand)
        if not ok:
            continue
        if len(cand) < best_len_now:
            best_len_now = len(cand)
            best = list(cand)
            best_len = best_len_now
            no_improve = 0
            cur = list(cand)
            temp = min(0.05, temp * 1.05)  # reheat a bit on improvement
        else:
            no_improve += 1
            # accept with SA probability
            if rnd() < 100000 * temp:
                cur = cand
        temp = max(min_temp, temp * cooldown)

        if no_improve > max_no_improve and len(cur) == best_len_now:
            # restart current from best to escape local optima
            cur = list(best)
            no_improve = 0
            temp = 0.03

    if time.time() - t0 > DEADLINE - 3.0:
        # not enough time to verify the final candidate; return best (known good)
        return best

    # Final safety: ensure returned network is valid.
    if _verify_net(best):
        return best
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

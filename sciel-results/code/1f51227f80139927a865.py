"""HP lattice protein folding on the cubic lattice via tail-shift simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_AXIS_SET = set(_AXES)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _is_unit(v):
    return v in _AXIS_SET


def _init_chain(sequence, rng=None):
    n = len(sequence)
    chain = [(i, 0, 0) for i in range(n)]
    if rng is not None and n >= 3:
        for _ in range(min(4, n - 1)):
            idx = rng.randint(1, n - 2)
            d = _AXES[rng.randint(0, 5)]
            cand = _add(chain[idx], d)
            if cand in chain:
                continue
            if _is_unit(_sub(cand, chain[idx - 1])) and _is_unit(_sub(chain[idx + 1], cand)):
                chain[idx] = cand
    return chain


def _energy_full(chain, sequence):
    occ = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for d in _AXES:
            j = occ.get(_add(p, d))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _anneal_proper(sequence, chain, time_limit, rng, t0):
    n = len(sequence)
    if n < 2:
        return chain
    occ = {p: i for i, p in enumerate(chain)}
    energy = _energy_full(chain, sequence)
    best_chain = list(chain)
    best_energy = energy

    T0 = 1.5
    Tend = 0.01
    horizon = 120000.0
    step = 0
    check = 0

    while True:
        p = rng.randint(1, n - 2)
        bond = _sub(chain[p], chain[p - 1])
        rev = (-bond[0], -bond[1], -bond[2])
        cands = [d for d in _AXES if d != rev]
        d_new = cands[rng.randint(0, len(cands) - 1)]
        if d_new != bond:
            delta = (d_new[0] - bond[0], d_new[1] - bond[1], d_new[2] - bond[2])
            moved = list(range(p + 1, n))
            fixed_set = set(chain[i] for i in range(0, p + 1))
            new_pos = {}
            ok = True
            for k in moved:
                np_ = _add(chain[k], delta)
                if np_ in fixed_set:
                    ok = False
                    break
                new_pos[k] = np_
            if ok:
                old_energy = energy
                for k in moved:
                    del occ[chain[k]]
                    chain[k] = new_pos[k]
                    occ[new_pos[k]] = k
                energy = _energy_full(chain, sequence)
                dE = energy - old_energy
                if dE <= 0:
                    accept = True
                else:
                    T = T0 * (1.0 - step / horizon)
                    if T < Tend:
                        T = Tend
                    accept = (rng.random() < math.exp(-dE / T))
                if not accept:
                    for k in moved:
                        del occ[chain[k]]
                        chain[k] = _sub(chain[k], delta)
                        occ[chain[k]] = k
                    energy = old_energy
                else:
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)

        step += 1
        check += 1
        if check >= 1000:
            check = 0
            if time.time() - t0 > time_limit:
                break

    for i in range(n):
        chain[i] = best_chain[i]
    return chain


def fold(sequence, dimension):
    t_start = time.time()
    hard_deadline = t_start + 285.0
    n = len(sequence)
    rng = random.Random(987654321 + n * 7919)

    # Budget: 26 sequences must fit in ~240s total -> keep each call small.
    per_call = min(6.0 + 0.03 * n, 9.0)
    per_call = min(per_call, max(0.0, hard_deadline - time.time() - 5.0))

    chain = _init_chain(sequence, rng)
    _anneal_proper(sequence, chain, per_call, rng, time.time())
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END

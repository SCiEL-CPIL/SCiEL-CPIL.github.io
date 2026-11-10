"""Parallel tempering Monte Carlo on a fixed SAW scaffold for HP protein folding."""
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_saw(n, dimension, rng):
    """Build a self-avoiding walk of length n by random walk with backtracking."""
    if n <= 1:
        return [(0,) * dimension]
    chain = [(0,) * dimension]
    occupied = {chain[0]}
    max_depth = n - 1
    while len(chain) < n:
        last = chain[-1]
        cands = [s for s in _neighbours(last) if s not in occupied]
        if len(chain) >= 2:
            prev = chain[-2]
            cands = [s for s in cands if s != prev]
        if not cands:
            # backtrack
            chain.pop()
            if len(chain) > 1:
                occupied.discard(chain[-1])
            if not chain:
                return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            continue
        nxt = cands[rng.randrange(len(cands))]
        chain.append(nxt)
        occupied.add(nxt)
    return chain


def _compute_contacts(sequence, chain):
    """Return set of (i, j) H-H contacts, i<j, |i-j|>1, lattice-neighbouring."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts


def _energy(contacts):
    return -len(contacts)


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [list((0,) * dimension)]

    rng = random.Random(12345 + len(sequence))
    scaffold = _build_saw(n, dimension, rng)

    # Parallel tempering setup
    n_replicas = 5
    T_low, T_high = 0.3, 6.0
    temps = [T_low + (T_high - T_low) * (i / (n_replicas - 1)) for i in range(n_replicas)]

    replicas = []
    best_overall = list(scaffold)
    best_overall_e = _energy(_compute_contacts(sequence, scaffold))
    for r in range(n_replicas):
        chain = list(scaffold)
        contacts = _compute_contacts(sequence, chain)
        e = _energy(contacts)
        replicas.append({
            'chain': chain,
            'contacts': contacts,
            'energy': e,
            'best': list(chain),
            'best_e': e,
        })
        if e < best_overall_e:
            best_overall_e = e
            best_overall = list(chain)

    # Number of sweeps: tune for time. Each move: O(n) for contact recompute.
    # Let's do 40 sweeps, 3n moves per sweep per replica.
    n_sweeps = 40
    for sweep in range(n_sweeps):
        for r in range(n_replicas):
            rep = replicas[r]
            chain = rep['chain']
            T = temps[r]
            n_moves = 3 * n
            for _ in range(n_moves):
                if rng.random() < 0.5:
                    # Pivot move
                    k = rng.randrange(1, n - 1)
                    pivot = chain[k - 1]
                    axis = rng.randrange(dimension)
                    new_tail = []
                    prefix_set = set(chain[:k])
                    seen = set()
                    ok = True
                    for p in chain[k:]:
                        q = list(p)
                        q[axis] = 2 * pivot[axis] - p[axis]
                        qt = tuple(q)
                        if qt in prefix_set or qt in seen:
                            ok = False
                            break
                        seen.add(qt)
                        new_tail.append(qt)
                    if not ok:
                        continue
                    new_chain = chain[:k] + new_tail
                    new_contacts = _compute_contacts(sequence, new_chain)
                    new_e = _energy(new_contacts)
                    delta = new_e - rep['energy']
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        rep['chain'] = new_chain
                        rep['contacts'] = new_contacts
                        rep['energy'] = new_e
                        if new_e < rep['best_e']:
                            rep['best_e'] = new_e
                            rep['best'] = list(new_chain)
                        if new_e < best_overall_e:
                            best_overall_e = new_e
                            best_overall = list(new_chain)
                else:
                    # Internal move
                    if n < 5:
                        continue
                    a = rng.randrange(1, n - 2)
                    b = rng.randrange(a + 1, n - 1)
                    pivot = chain[a - 1]
                    axis = rng.randrange(dimension)
                    new_seg = []
                    prefix_set = set(chain[:a])
                    suffix_set = set(chain[b:])
                    seen = set()
                    ok = True
                    for p in chain[a:b]:
                        q = list(p)
                        q[axis] = 2 * pivot[axis] - p[axis]
                        qt = tuple(q)
                        if qt in prefix_set or qt in suffix_set or qt in seen:
                            ok = False
                            break
                        seen.add(qt)
                        new_seg.append(qt)
                    if not ok:
                        continue
                    new_chain = chain[:a] + new_seg + chain[b:]
                    new_contacts = _compute_contacts(sequence, new_chain)
                    new_e = _energy(new_contacts)
                    delta = new_e - rep['energy']
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        rep['chain'] = new_chain
                        rep['contacts'] = new_contacts
                        rep['energy'] = new_e
                        if new_e < rep['best_e']:
                            rep['best_e'] = new_e
                            rep['best'] = list(new_chain)
                        if new_e < best_overall_e:
                            best_overall_e = new_e
                            best_overall = list(new_chain)

        # Swaps between adjacent replicas
        for r in range(n_replicas - 1):
            if rng.random() < 0.4:
                r1, r2 = r, r + 1
                e1, e2 = replicas[r1]['energy'], replicas[r2]['energy']
                T1, T2 = temps[r1], temps[r2]
                delta = (e2 / T2) - (e1 / T1)
                if delta <= 0 or rng.random() < math.exp(-delta):
                    replicas[r1]['chain'], replicas[r2]['chain'] = replicas[r2]['chain'], replicas[r1]['chain']
                    replicas[r1]['contacts'], replicas[r2]['contacts'] = replicas[r2]['contacts'], replicas[r1]['contacts']
                    replicas[r1]['energy'], replicas[r2]['energy'] = replicas[r2]['energy'], replicas[r1]['energy']

    return [list(p) for p in best_overall]

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


class _FoldingEngine:
    def __init__(self, dim=3):
        self.dim = dim
        self._deadline = None

    def _time_left(self, total=330.0, margin=0.88):
        if self._deadline is None:
            self._deadline = time.time() + total * margin
        return self._deadline - time.time()

    def _adjacent(self, pt):
        for ax in range(self.dim):
            for d in (-1, 1):
                yield tuple(pt[ax] + d if i == ax else pt[i] for i in range(self.dim))

    def _count_contacts(self, seq, chain):
        idx = {p: i for i, p in enumerate(chain)}
        c = 0
        for i, p in enumerate(chain):
            if seq[i] == 'H':
                for nb in self._adjacent(p):
                    j = idx.get(nb)
                    if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                        c += 1
        return c

    def _build_random(self, seq, dim, rng):
        n = len(seq)
        if n <= 2:
            return [(0,) * dim for _ in range(n)]
        pos = [(0,) * dim]
        occ = {pos[0]: 0}
        for i in range(1, n):
            cands = []
            last = pos[-1]
            for nb in self._adjacent(last):
                if nb in occ:
                    continue
                h_gain = 0
                if seq[i] == 'H':
                    for nb2 in self._adjacent(nb):
                        j = occ.get(nb2)
                        if j is not None and j <= i - 2 and seq[j] == 'H':
                            h_gain += 1
                open_sp = sum(1 for x in self._adjacent(nb) if x not in occ)
                cands.append((h_gain * 3.0 + open_sp * 0.5 + rng.random() * 2.0, nb))
            if not cands:
                return None
            cands.sort(reverse=True)
            k = min(len(cands) - 1, int(rng.expovariate(1.0 / 2.0)))
            chosen = cands[k][1]
            pos.append(chosen)
            occ[chosen] = i
        return pos

    def _anneal(self, seq, chain, dim, rng, t_limit=2.0):
        n = len(chain)
        if n < 6:
            return chain, self._count_contacts(seq, chain)
        chain = [tuple(p) for p in chain]
        occ = {p: i for i, p in enumerate(chain)}
        cur_energy = self._count_contacts(seq, chain)
        best_energy = cur_energy
        best = list(chain)
        t0 = time.time()
        t_end = t0 + t_limit
        temp = 0.8
        cool = 0.995
        steps = 0
        while time.time() < t_end and self._time_left() > 5.0:
            steps += 1
            if steps % 500 == 0:
                temp *= cool
            if temp < 0.01:
                temp = 0.01
            i = rng.randint(1, n - 2)
            for axis in range(dim):
                for sign in (-1, 1):
                    new_pt = list(chain[i])
                    new_pt[axis] += sign
                    new_pt = tuple(new_pt)
                    if new_pt in occ:
                        continue
                    p_prev = chain[i - 1]
                    p_next = chain[i + 1]
                    if sum(abs(a - b) for a, b in zip(new_pt, p_prev)) != 1:
                        continue
                    if sum(abs(a - b) for a, b in zip(new_pt, p_next)) != 1:
                        continue
                    old_c = 0
                    if seq[i] == 'H':
                        for nb in self._adjacent(chain[i]):
                            j = occ.get(nb)
                            if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                                old_c += 1
                    new_c = 0
                    if seq[i] == 'H':
                        for nb in self._adjacent(new_pt):
                            j = occ.get(nb)
                            if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                                new_c += 1
                    delta = new_c - old_c
                    accept = delta < 0 or rng.random() < min(1.0, (delta / temp) if delta > 0 else 1.0)
                    if accept:
                        occ.pop(chain[i])
                        occ[new_pt] = i
                        chain[i] = new_pt
                        cur_energy += delta
                        if cur_energy > best_energy:
                            best_energy = cur_energy
                            best = list(chain)
                        break
        return best, best_energy

    def _pivot_move(self, seq, chain, dim, rng):
        n = len(chain)
        if n < 8:
            return None
        occ = {p: i for i, p in enumerate(chain)}
        i = rng.randint(2, n - 3)
        for axis in range(dim):
            for sign in (-1, 1):
                pivot = list(chain[i])
                pivot[axis] += sign
                pivot = tuple(pivot)
                if pivot in occ:
                    continue
                if sum(abs(a - b) for a, b in zip(pivot, chain[i - 1])) != 1:
                    continue
                if sum(abs(a - b) for a, b in zip(pivot, chain[i + 1])) != 1:
                    continue
                seg = list(chain[i + 1:])
                new_occ = {p: k for k, p in enumerate(chain[:i + 1])}
                for k, pt in enumerate(seg):
                    new_occ[pt] = k + i + 1
                trial = list(chain[:i + 1]) + seg
                trial[i] = pivot
                new_occ.pop(pivot, None)
                valid = True
                for k in range(i + 1, n):
                    pt = trial[k]
                    if k + i in new_occ or pt in new_occ:
                        if pt in new_occ and new_occ[pt] != k:
                            valid = False
                            break
                    for nb in self._adjacent(pt):
                        if nb in new_occ:
                            j = new_occ[nb]
                            if abs(j - k) > 1:
                                pass
                if not valid:
                    continue
                new_chain = list(chain[:i + 1]) + seg
                new_chain[i] = pivot
                new_occ_check = {p: idx for idx, p in enumerate(new_chain)}
                conflict = False
                for k in range(i + 1, n):
                    for nb in self._adjacent(new_chain[k]):
                        j = new_occ_check.get(nb)
                        if j is not None and abs(j - k) > 1:
                            pass
                if self._is_valid_conformation(new_chain):
                    return new_chain
        return None

    def _is_valid_conformation(self, chain):
        n = len(chain)
        occ = {}
        for i, p in enumerate(chain):
            if p in occ:
                return False
            occ[p] = i
            if i > 0 and sum(abs(a - b) for a, b in zip(p, chain[i - 1])) != 1:
                return False
        return True


_ENGINE = _FoldingEngine(dim=3)


def fold(sequence, dimension):
    eng = _FoldingEngine(dim=dimension)
    rng = random.Random(42 + len(sequence))
    n = len(sequence)
    best_chain = None
    best_energy = 0

    t_start = time.time()
    total_budget = 330.0 * 0.88 - (time.time() - t_start)
    if total_budget <= 0:
        total_budget = 10.0

    if n <= 100:
        n_builds = 40
    else:
        n_builds = 20

    per_build = total_budget / max(1, n_builds)

    for b in range(n_builds):
        remaining = total_budget - (time.time() - t_start)
        if remaining < 1.0:
            break
        chain = eng._build_random(sequence, dimension, rng)
        if chain is None:
            continue
        e = eng._count_contacts(sequence, chain)
        if e > best_energy:
            best_chain = list(chain)
            best_energy = e
        alloc = min(remaining * 0.6, 1.5)
        if alloc > 0.3:
            refined, e2 = eng._anneal(sequence, chain, dimension, rng, t_limit=alloc)
            if e2 > best_energy:
                best_chain = list(refined)
                best_energy = e2

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

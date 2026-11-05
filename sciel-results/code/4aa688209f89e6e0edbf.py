# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


_DIRS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]
_OPP = [1, 0, 3, 2, 5, 4]


class _State:
    __slots__ = ("seq", "steps", "pos", "occ", "contacts")

    def __init__(self, seq, steps):
        self.seq = seq
        self.steps = list(steps)
        self.pos = [(0, 0, 0)]
        self.occ = {(0, 0, 0): 0}
        x = y = z = 0
        for i, s in enumerate(self.steps):
            d = _DIRS[s]
            x += d[0]; y += d[1]; z += d[2]
            p = (x, y, z)
            self.pos.append(p)
            self.occ[p] = i
        self.contacts = self._count_contacts()

    def _count_contacts(self):
        contacts = 0
        n = len(self.pos)
        for i in range(n):
            if self.seq[i] != "H":
                continue
            p = self.pos[i]
            for d in _DIRS:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                j = self.occ.get(q)
                if j is not None and j > i + 1 and self.seq[j] == "H":
                    contacts += 1
        return contacts

    def try_move(self, i, new_step):
        if i < 0 or i >= len(self.steps):
            return False
        old_step = self.steps[i]
        if new_step == old_step or new_step == _OPP[old_step]:
            return False
        start = self.pos[i]
        d = _DIRS[new_step]
        new_end = (start[0] + d[0], start[1] + d[1], start[2] + d[2])
        if new_end in self.occ:
            return False
        old_end = self.pos[i + 1]
        seq = self.seq
        delta = 0
        if seq[i + 1] == "H":
            for dd in _DIRS:
                q = (old_end[0] + dd[0], old_end[1] + dd[1], old_end[2] + dd[2])
                j = self.occ.get(q)
                if j is not None and j != i + 1 and j != i and seq[j] == "H":
                    delta -= 1
            for dd in _DIRS:
                q = (new_end[0] + dd[0], new_end[1] + dd[1], new_end[2] + dd[2])
                j = self.occ.get(q)
                if j is not None and j != i + 1 and j != i and seq[j] == "H":
                    delta += 1
        self.steps[i] = new_step
        self.pos[i + 1] = new_end
        del self.occ[old_end]
        self.occ[new_end] = i + 1
        self.contacts += delta
        return True

    def energy(self):
        return -self.contacts

    def snapshot_steps(self):
        return list(self.steps)


def _greedy_grow(seq, rng, temperature):
    n = len(seq)
    if n <= 1:
        return []
    steps = []
    pos = [(0, 0, 0)]
    occ = {(0, 0, 0): 0}
    for i in range(1, n):
        last = pos[-1]
        cands = []
        for s in range(6):
            d = _DIRS[s]
            q = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            if q in occ:
                continue
            gain = 0
            if seq[i] == "H":
                for dd in _DIRS:
                    r = (q[0] + dd[0], q[1] + dd[1], q[2] + dd[2])
                    j = occ.get(r)
                    if j is not None and j < i - 1 and seq[j] == "H":
                        gain += 1
            free = 0
            for dd in _DIRS:
                r = (q[0] + dd[0], q[1] + dd[1], q[2] + dd[2])
                if r not in occ:
                    free += 1
            score = gain + 0.05 * free + temperature * rng.random()
            cands.append((score, s))
        if not cands:
            return None
        cands.sort(reverse=True)
        top = cands[:3]
        s = top[0][1]
        steps.append(s)
        d = _DIRS[s]
        q = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
        pos.append(q)
        occ[q] = i
    return steps


def _hill_climb(state, max_iters, rng, deadline_check):
    iters = 0
    improved = True
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if iters % 25 == 0 and deadline_check is not None and deadline_check() < 0:
            break
        n = len(state.steps)
        if n == 0:
            break
        i = rng.randrange(n)
        old_step = state.steps[i]
        old_contacts = state.contacts
        for _ in range(8):
            new_step = rng.randrange(6)
            if new_step == old_step or new_step == _OPP[old_step]:
                continue
            if state.try_move(i, new_step):
                if state.contacts > old_contacts:
                    old_contacts = state.contacts
                    improved = True
                    break
                else:
                    if not state.try_move(i, old_step):
                        state.contacts = state._count_contacts()
    return state


def _anneal(state, rng, t0, t1, iters, deadline_check):
    current = state
    best_contacts = current.contacts
    for it in range(iters):
        if it % 200 == 0 and deadline_check is not None and deadline_check() < 0:
            break
        n = len(current.steps)
        if n == 0:
            break
        i = rng.randrange(n)
        old_step = current.steps[i]
        old_contacts = current.contacts
        new_step = rng.randrange(6)
        if new_step == old_step or new_step == _OPP[old_step]:
            continue
        if not current.try_move(i, new_step):
            continue
        delta = current.contacts - old_contacts
        if delta >= 0:
            pass
        else:
            T = t0 + (t1 - t0) * (it / max(1, iters))
            if T <= 0:
                T = 1e-6
            if rng.random() < math.exp(delta / T):
                pass
            else:
                if not current.try_move(i, old_step):
                    current.contacts = current._count_contacts()
                continue
        if current.contacts > best_contacts:
            best_contacts = current.contacts
    return current


def fold(sequence, dimension):
    seq = sequence
    n = len(seq)
    if n <= 1:
        return [[0, 0, 0]] * n
    rng = random.Random(12345 + n)
    deadline_check = _time_left

    best_steps = None
    best_energy = 1

    def consider(state):
        nonlocal best_steps, best_energy
        e = state.energy()
        if e < best_energy:
            best_energy = e
            best_steps = state.snapshot_steps()

    # Phase 1: greedy growths + hill climb
    for g in range(500):
        if deadline_check() < 5.0:
            break
        steps = _greedy_grow(seq, rng, temperature=1.0)
        if steps is None:
            continue
        state = _State(seq, steps)
        _hill_climb(state, max_iters=1500, rng=rng, deadline_check=deadline_check)
        consider(state)

    if best_steps is None:
        return [[i, 0, 0] for i in range(n)]

    # Phase 2: anneal from best
    if deadline_check() > 10.0:
        state = _State(seq, best_steps)
        state = _anneal(state, rng, t0=1.0, t1=0.05, iters=80000,
                        deadline_check=deadline_check)
        consider(state)

    # Phase 3: a couple more seeds + anneal if time remains
    if deadline_check() > 15.0:
        for g in range(4):
            if deadline_check() < 10.0:
                break
            steps = _greedy_grow(seq, rng, temperature=2.0)
            if steps is None:
                continue
            state = _State(seq, steps)
            _hill_climb(state, max_iters=1200, rng=rng, deadline_check=deadline_check)
            state = _anneal(state, rng, t0=1.0, t1=0.05, iters=50000,
                            deadline_check=deadline_check)
            consider(state)

    # Final hill climb on best to squeeze out more
    if deadline_check() > 3.0:
        state = _State(seq, best_steps)
        _hill_climb(state, max_iters=5000, rng=rng, deadline_check=deadline_check)
        consider(state)

    final = _State(seq, best_steps)
    return [list(p) for p in final.pos]
# EVOLVE-BLOCK-END

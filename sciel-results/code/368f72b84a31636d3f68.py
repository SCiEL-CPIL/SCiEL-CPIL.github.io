"""Combined program: greedy growth + optimized incremental local search + proper SA with snapshot/restore + basin-escape perturbations."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _lattice_adjacent(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_contacts(sequence, chain):
    """Return set of (i,j) H-H contact pairs (i<j, not chain-adjacent)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts

def _rotation_matrices():
    """Return list of 24 rotation matrices for the cubic group."""
    mats = []
    e = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for x in e:
        for y in e:
            cx = x[1]*y[2] - x[2]*y[1]
            cy = x[2]*y[0] - x[0]*y[2]
            cz = x[0]*y[1] - x[1]*y[0]
            z = (cx, cy, cz)
            if sum(a*b for a,b in zip(x,y)) != 0:
                continue
            if sum(a*b for a,b in zip(x,z)) != 0:
                continue
            if sum(a*b for a,b in zip(y,z)) != 0:
                continue
            m = (x, y, z)
            if m not in mats:
                mats.append(m)
    return mats

_ROT_MATS = _rotation_matrices()

def _apply_rotation(mat, vec):
    r0, r1, r2 = mat
    return (
        r0[0]*vec[0] + r0[1]*vec[1] + r0[2]*vec[2],
        r1[0]*vec[0] + r1[1]*vec[1] + r1[2]*vec[2],
        r2[0]*vec[0] + r2[1]*vec[1] + r2[2]*vec[2],
    )

class _FoldingState:
    """State with optimized contact tracking and proper snapshot/restore for SA."""
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = _compute_contacts(sequence, chain)
        self.contact_map = {i: set() for i in range(self.n)}
        for i, j in self.contacts:
            self.contact_map[i].add(j)
            self.contact_map[j].add(i)
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)

    def _remove_contacts_involving(self, indices):
        """Remove all contacts involving any index in the set."""
        indices_set = set(indices)
        removed_count = 0
        for i in indices_set:
            partners = list(self.contact_map[i])
            for j in partners:
                if j in indices_set:
                    # Both in moved set - only remove once (when i < j)
                    if i >= j:
                        continue
                pair = (i, j) if i < j else (j, i)
                if pair in self.contacts:
                    self.contacts.discard(pair)
                    self.energy += 1
                    self.contact_map[i].discard(j)
                    self.contact_map[j].discard(i)
                    removed_count += 1
        return removed_count

    def _add_new_contacts(self, indices, new_positions):
        """Add new contacts for moved indices at their new positions."""
        indices_set = set(indices)
        added_count = 0
        for i in indices_set:
            if not self.is_h[i]:
                continue
            pos = new_positions[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is None or j == i:
                    continue
                if abs(j - i) == 1:
                    continue
                if self.is_h[j]:
                    pair = (i, j) if i < j else (j, i)
                    if pair not in self.contacts:
                        self.contacts.add(pair)
                        self.energy -= 1
                        self.contact_map[i].add(j)
                        self.contact_map[j].add(i)
                        added_count += 1
        return added_count

    def _update_energy_for_move(self, moved_indices, new_positions):
        """Incrementally update energy after moving residues."""
        self._remove_contacts_involving(moved_indices)
        self._add_new_contacts(moved_indices, new_positions)

    def _snapshot(self):
        """Save full state for undo."""
        return (
            list(self.chain),
            dict(self.occupied),
            set(self.contacts),
            {i: set(s) for i, s in self.contact_map.items()},
            self.energy,
        )

    def _restore(self, snap):
        """Restore state from snapshot."""
        self.chain = snap[0]
        self.occupied = snap[1]
        self.contacts = snap[2]
        self.contact_map = snap[3]
        self.energy = snap[4]

    def _pivot_move(self, rng):
        """True pivot: rotate subchain around a bond using a rotation matrix."""
        n = self.n
        if n < 4:
            return False
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        valid_mats = []
        for mat in _ROT_MATS:
            new_dir = _apply_rotation(mat, orig_dir)
            if new_dir != orig_dir and new_dir != tuple(-x for x in orig_dir):
                valid_mats.append((mat, new_dir))

        if not valid_mats:
            return False

        mat, new_dir = valid_mats[rng.randint(0, len(valid_mats) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return False

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_step = _apply_rotation(mat, old_step)
            new_pos = tuple(new_subchain[k-1][m] + new_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos

        if not valid:
            return False

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return False
            seen.add(pos)

        if pivot_idx + subchain_len < n:
            if not _lattice_adjacent(new_subchain[-1], self.chain[pivot_idx + subchain_len]):
                return False

        moved_indices = list(range(pivot_idx + 1, n))
        old_positions = {i: self.chain[i] for i in moved_indices}

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        return True

    def _pivot_move_translate(self, rng):
        """Translation-based pivot: shift subchain by changing first step direction."""
        n = self.n
        if n < 4:
            return False
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        possible_dirs = []
        for ax in range(3):
            for s in (-1, 1):
                d = [0, 0, 0]
                d[ax] = s
                d = tuple(d)
                if d != orig_dir and d != tuple(-x for x in orig_dir):
                    possible_dirs.append(d)

        if not possible_dirs:
            return False

        new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return False

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_pos = tuple(new_subchain[k-1][m] + old_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos

        if not valid:
            return False

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return False
            seen.add(pos)

        if pivot_idx + subchain_len < n:
            if not _lattice_adjacent(new_subchain[-1], self.chain[pivot_idx + subchain_len]):
                return False

        moved_indices = list(range(pivot_idx + 1, n))

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        return True

    def _crank_move(self, rng):
        """Crank: translate a subchain by one unit in a random direction."""
        n = self.n
        if n < 5:
            return False
        start_idx = rng.randint(1, n - 3)
        end_idx = rng.randint(start_idx + 1, n - 1)
        subchain = list(self.chain[start_idx:end_idx + 1])

        ax = rng.randint(0, 2)
        s = rng.choice((-1, 1))
        new_subchain = []
        for p in subchain:
            np_ = list(p)
            np_[ax] += s
            new_subchain.append(tuple(np_))

        seen = set()
        for np_ in new_subchain:
            if np_ in seen:
                return False
            seen.add(np_)

        for p in subchain:
            del self.occupied[p]

        for np_ in new_subchain:
            if np_ in self.occupied:
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        if start_idx > 0:
            if not _lattice_adjacent(self.chain[start_idx - 1], new_subchain[0]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        if end_idx < n - 1:
            if not _lattice_adjacent(new_subchain[-1], self.chain[end_idx + 1]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        moved_indices = list(range(start_idx, end_idx + 1))

        for i, np_ in enumerate(new_subchain):
            self.chain[start_idx + i] = np_
            self.occupied[np_] = start_idx + i

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        return True

    def _basin_escape(self, rng):
        """Larger perturbation: re-grow a middle section to jump between basins."""
        n = self.n
        if n < 10:
            return False
        # Pick a middle segment to re-grow
        seg_start = rng.randint(2, n - 5)
        seg_end = rng.randint(seg_start + 3, n - 2)
        seg_len = seg_end - seg_start + 1

        # Save the segment
        old_seg = list(self.chain[seg_start:seg_end + 1])

        # Try to regrow the segment with randomization
        # Start from chain[seg_start-1], grow seg_len residues
        anchor = self.chain[seg_start - 1]
        # Remove old segment from occupied
        for p in old_seg:
            del self.occupied[p]

        # Regrow
        new_seg = [anchor]
        for k in range(seg_len):
            options = []
            for site in _neighbours(new_seg[-1]):
                if site in self.occupied:
                    continue
                # Check it's not the next anchor
                if k == seg_len - 1 and site == self.chain[seg_end + 1]:
                    pass  # will check later
                gain = 0
                if self.seq[seg_start + k] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in self.occupied and self.is_h[self.occupied[q]])
                free = sum(1 for q in _neighbours(site) if q not in self.occupied)
                options.append((gain + 0.1 * free + rng.random(), site))
            if not options:
                # Restore
                for i, p in enumerate(old_seg):
                    self.chain[seg_start + i] = p
                    self.occupied[p] = seg_start + i
                return False
            site = max(options)[1]
            new_seg.append(site)

        # Check connection to next anchor
        if seg_end + 1 < n:
            if not _lattice_adjacent(new_seg[-1], self.chain[seg_end + 1]):
                for i, p in enumerate(old_seg):
                    self.chain[seg_start + i] = p
                    self.occupied[p] = seg_start + i
                return False

        # Check self-avoidance
        seen = set(new_seg[1:])
        if len(seen) != len(new_seg) - 1:
            for i, p in enumerate(old_seg):
                self.chain[seg_start + i] = p
                self.occupied[p] = seg_start + i
            return False

        # Apply
        moved_indices = list(range(seg_start, seg_end + 1))
        for i, p in enumerate(new_seg[1:]):
            self.chain[seg_start + i] = p
            self.occupied[p] = seg_start + i

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        return True

    def _random_move(self, rng):
        r = rng.random()
        if r < 0.30:
            return self._pivot_move(rng)
        elif r < 0.60:
            return self._pivot_move_translate(rng)
        elif r < 0.85:
            return self._crank_move(rng)
        else:
            return self._basin_escape(rng)

def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Local search with incremental energy updates."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)
    fail_count = 0

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        ok = state._random_move(rng)
        if ok:
            state._accept_if_better()
            fail_count = 0
        else:
            fail_count += 1
            # If many failures, try a basin escape
            if fail_count > 20:
                state._basin_escape(rng)
                state._accept_if_better()
                fail_count = 0

    return state.best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing with proper accept/reject using snapshot/restore."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)
    T_start = 4.0
    T_end = 0.1
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling_rate

        snap = state._snapshot()
        ok = state._random_move(rng)
        if not ok:
            continue

        delta = state.energy - snap[4]
        if delta < 0 or (T > 0.05 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._restore(snap)

    return state.best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth + incremental local search + simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Number of restarts based on sequence length and time budget
    time_per_restart = 0.12 if n <= 50 else (0.25 if n <= 80 else 0.5)
    max_restarts = int(_time_left() / time_per_restart)
    max_restarts = max(20, min(max_restarts, 2000))

    for r in range(max_restarts):
        if _time_left() < 3.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        if _time_left() > 2.0:
            # Alternate between local search and simulated annealing
            if r % 3 == 0:
                max_ls = 350 if n <= 50 else (200 if n <= 80 else 100)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
            elif r % 3 == 1:
                max_sa = 400 if n <= 50 else (250 if n <= 80 else 120)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
            else:
                # Combined: SA then LS
                max_sa = 200 if n <= 50 else (150 if n <= 80 else 80)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
                if chain is not None and _time_left() > 2.0:
                    max_ls = 150 if n <= 50 else (100 if n <= 80 else 60)
                    chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)

        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

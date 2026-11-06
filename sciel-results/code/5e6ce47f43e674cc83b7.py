"""HP lattice protein folding: beam search + simulated annealing with segment moves."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_T0 = time.time()
_DEADLINE = 195.0


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    h_positions = [i for i, c in enumerate(seq) if c == 'H']
    if len(h_positions) < 2:
        return 0
    # Build position lookup
    pos_map = {}
    for i, p in enumerate(chain):
        pos_map[p] = i
    contacts = 0
    for i in h_positions:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_map.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(seq, beam_width=80):
    """Beam search to find good initial conformation."""
    n = len(seq)
    if n <= 2:
        return [[0, 0], [1, 0]] if n == 2 else [[0, 0]]

    # Beams: list of (energy, chain_list, occupied_dict)
    beams = [(-1, [[0, 0]], {(0, 0): 0})]

    for i in range(1, n):
        new_beams = []
        for energy, chain, occupied in beams:
            x, y = chain[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                if new_pos in occupied:
                    continue
                new_chain = chain + [[x + dx, y + dy]]
                new_occupied = dict(occupied)
                new_occupied[new_pos] = i

                new_energy = energy
                if seq[i] == 'H':
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = new_occupied.get((x + dx + dx2, y + dy + dy2))
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            new_energy -= 1

                new_beams.append((new_energy, new_chain, new_occupied))

        if not new_beams:
            return beams[0][1]

        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]

    return beams[0][1]


def _simulated_annealing(seq, chain, max_time=2.0):
    """Simulated annealing with segment reversal moves."""
    n = len(seq)
    if n <= 3:
        return chain

    best_chain = [list(p) for p in chain]
    best_energy = _compute_energy(seq, best_chain)
    current_chain = list(best_chain)
    current_energy = best_energy

    start_temp = 3.0
    end_temp = 0.01
    start_time = time.time()
    time_budget = max_time

    # Precompute H indices
    h_set = set(i for i, c in enumerate(seq) if c == 'H')

    step = 0
    while time.time() - start_time < time_budget and _remaining() > 5.0:
        # Annealing schedule
        progress = min(1.0, (time.time() - start_time) / time_budget)
        temp = start_temp * (end_temp / start_temp) ** progress

        # Choose a move: reverse a random segment
        if n <= 4:
            i = 1
            j = n - 1
        else:
            i = random.randint(1, n - 3)
            j = random.randint(i + 2, n - 1)

        # Reverse segment [i, j]
        new_chain = current_chain[:i] + current_chain[i:j+1][::-1] + current_chain[j+1:]

        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))

        if not valid:
            continue

        # Check unit steps
        for k in range(n - 1):
            if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                valid = False
                break

        if not valid:
            continue

        new_energy = _compute_energy(seq, new_chain)
        delta_e = new_energy - current_energy

        if delta_e < 0 or random.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in current_chain]

        step += 1

    return best_chain


def _try_segment_moves(seq, chain, max_moves=200):
    """Try specific segment reversals for refinement."""
    n = len(seq)
    if n <= 3:
        return chain

    best_chain = [list(p) for p in chain]
    best_energy = _compute_energy(seq, best_chain)
    h_set = set(i for i, c in enumerate(seq) if c == 'H')

    for _ in range(max_moves):
        if _remaining() < 3.0:
            break

        improved = False
        # Try reversing segments of various lengths
        for seg_len in [2, 3, 4, 5]:
            for i in range(1, n - seg_len - 1):
                j = i + seg_len
                new_chain = best_chain[:i] + best_chain[i:j+1][::-1] + best_chain[j+1:]

                # Quick self-avoidance check
                seen = set()
                valid = True
                for p in new_chain:
                    tp = tuple(p)
                    if tp in seen:
                        valid = False
                        break
                    seen.add(tp)
                if not valid:
                    continue

                # Unit step check
                for k in range(n - 1):
                    if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                        valid = False
                        break
                if not valid:
                    continue

                new_energy = _compute_energy(seq, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True

        if not improved:
            break

    return best_chain


def fold(sequence, dimension):
    """Fold a protein sequence into a low-energy conformation."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]

    best_chain = None
    best_energy = 0

    # Phase 1: Beam search with different widths
    for bw in [30, 60, 100]:
        if _remaining() < 15.0:
            break
        chain = _beam_search(sequence, bw)
        if chain is None or len(chain) != n:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)

    # Phase 2: Simulated annealing refinement
    if _remaining() > 10.0:
        refined = _simulated_annealing(sequence, best_chain, max_time=min(3.0, _remaining() * 0.4))
        energy = _compute_energy(sequence, refined)
        if energy < best_energy:
            best_chain = refined
            best_energy = energy

    # Phase 3: Segment move refinement
    if _remaining() > 5.0:
        refined = _try_segment_moves(sequence, best_chain, max_moves=100)
        energy = _compute_energy(sequence, refined)
        if energy < best_energy:
            best_chain = refined
            best_energy = energy

    # Phase 4: Another SA round if time permits
    if _remaining() > 8.0:
        refined = _simulated_annealing(sequence, best_chain, max_time=min(2.0, _remaining() * 0.5))
        energy = _compute_energy(sequence, refined)
        if energy < best_energy:
            best_chain = refined
            best_energy = energy

    return best_chain
# EVOLVE-BLOCK-END

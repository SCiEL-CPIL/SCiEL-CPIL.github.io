"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 10.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate the energy of a chain (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _build_h_positions(sequence):
    """Precompute set of H residue indices for fast lookup."""
    return set(i for i, c in enumerate(sequence) if c == "H")

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    Returns a valid self-avoiding chain or None."""
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

def _beam_search_fold(sequence, dimension, beam_width, rng, time_limit):
    """Beam search for HP lattice folding. Returns the best chain found or None."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]

    start_point = (0,) * dimension
    start_chain = [start_point]
    start_occupied = {start_point: 0}
    start_energy = 0
    start_score = 0

    beam = [(start_score, start_energy, start_chain, start_occupied, start_point)]
    h_pos = _build_h_positions(sequence)

    for i in range(1, n):
        res = sequence[i]
        new_beam_candidates = []

        if time.time() > time_limit:
            return None

        for score, energy_so_far, chain, occupied_dict, last_point in beam:
            for nb in _neighbours(last_point):
                if nb in occupied_dict:
                    continue

                delta_e = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in occupied_dict:
                            j = occupied_dict[q]
                            if j != i - 1:
                                if sequence[j] == "H":
                                    delta_e += 1
                new_energy = energy_so_far - delta_e

                free_count = 0
                for q in _neighbours(nb):
                    if q not in occupied_dict:
                        free_count += 1

                noise = rng.random() * 0.01
                heuristic_weight = 0.1
                new_score = new_energy + heuristic_weight * free_count + noise

                new_chain = chain + [nb]
                new_occupied = occupied_dict.copy()
                new_occupied[nb] = i

                new_beam_candidates.append((new_score, new_energy, new_chain, new_occupied, nb))

        if not new_beam_candidates:
            return None

        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[:beam_width]

        if not beam:
            return None

    best_state = min(beam, key=lambda x: x[1])
    return best_state[2]

def _local_search(sequence, chain, max_iters, rng, time_limit, use_annealing=True, polish=False, pivot_prob=0.4):
    """
    Hybrid local search: single-residue moves + pivot (sub-chain rotation) moves,
    with optional simulated annealing. Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain

    h_pos = _build_h_positions(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}

    if use_annealing:
        if polish:
            T_start = 0.4
            T_end = 0.005
        else:
            T_start = 1.6
            T_end = 0.05
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))

    iters = 0
    no_improve_count = 0
    patience = 250 if polish else 700

    def _try_single_move():
        """Attempt one single-residue move. Returns (accepted, new_energy, delta) or None if no valid move."""
        nonlocal current_energy, best_energy, best_chain, no_improve_count
        i = rng.randrange(1, n - 1)
        old_site = chain[i]
        for new_site in _neighbours(old_site):
            if new_site in occupied:
                continue
            if i > 0:
                prev = chain[i - 1]
                if abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1]) + abs(new_site[2] - prev[2]) != 1:
                    continue
            if i < n - 1:
                nxt = chain[i + 1]
                if abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1]) + abs(new_site[2] - nxt[2]) != 1:
                    continue

            delta = 0
            if i in h_pos:
                for q in _neighbours(old_site):
                    j = occupied.get(q)
                    if j is not None and j != i and j in h_pos and abs(i - j) > 1:
                        delta += 1
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j != i and j in h_pos and abs(i - j) > 1:
                        delta -= 1

            new_energy = current_energy + delta
            if _accept(delta, new_energy):
                occupied.pop(old_site)
                occupied[new_site] = i
                chain[i] = new_site
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                return True
        return False

    def _try_pivot_move():
        """Attempt one pivot (sub-chain rotation) move. Returns True if a move was applied."""
        nonlocal current_energy, best_energy, best_chain, no_improve_count
        # Pick a pivot (fixed) residue.
        pivot = rng.randrange(1, n - 1)
        # Pick a segment length to rotate.
        max_len = min(2 + rng.randrange(0, 12), n - pivot - 1, pivot - 1)
        if max_len < 1:
            return False
        L = rng.randrange(1, max_len + 1)
        # Decide which side to rotate.
        if rng.random() < 0.5:
            # Rotate the segment [pivot-L, pivot-1] (the part before the pivot).
            start = pivot - L
            end = pivot - 1
            fixed_side = "before"
        else:
            # Rotate the segment [pivot+1, pivot+L] (the part after the pivot).
            start = pivot + 1
            end = pivot + L
            fixed_side = "after"
        if start < 1 or end > n - 2:
            return False

        # Collect the moving residues (contiguous, excluding the pivot).
        moving = list(range(start, end + 1))
        # The two residues adjacent to the moving block that are FIXED:
        # For "before": fixed neighbors are chain[start-1] and chain[pivot].
        # For "after":  fixed neighbors are chain[pivot] and chain[end+1].
        if fixed_side == "before":
            anchor_a = chain[start - 1]  # residue just before the block
            anchor_b = chain[pivot]      # the pivot
        else:
            anchor_a = chain[pivot]      # the pivot
            anchor_b = chain[end + 1]    # residue just after the block

        old_sites = {idx: chain[idx] for idx in moving}

        # Compute delta: contacts lost/gained for each moving residue relative to
        # residues that are NOT moving (i.e., all occupied minus the moving set).
        delta = 0
        moving_set = set(moving)
        for idx in moving:
            if idx not in h_pos:
                continue
            old = old_sites[idx]
            for q in _neighbours(old):
                j = occupied.get(q)
                if j is None or j in moving_set or abs(idx - j) <= 1:
                    continue
                # contact lost
                delta += 1

        # Find a valid rotation: try all 6 neighbor directions for the first moving
        # residue (the one adjacent to a fixed anchor), and propagate the rest.
        # The first moving residue must stay adjacent to its fixed anchor.
        # Determine the anchor that the first moving residue is attached to.
        if fixed_side == "before":
            first_moving = moving[0]   # start
            first_anchor = anchor_a    # chain[start-1]
        else:
            first_moving = moving[0]   # pivot+1
            first_anchor = anchor_a    # chain[pivot]

        # Also, the last moving residue is attached to the other anchor; we must ensure
        # the rotated chain's last residue stays adjacent to the other anchor.
        last_moving = moving[-1]
        if fixed_side == "before":
            last_anchor = anchor_b  # chain[pivot]
        else:
            last_anchor = anchor_b  # chain[end+1]

        best_rotation = None
        best_rotation_delta = None
        for new_first in _neighbours(old_sites[first_moving]):
            if new_first in occupied and occupied[new_first] not in moving_set:
                continue
            if abs(new_first[0] - first_anchor[0]) + abs(new_first[1] - first_anchor[1]) + abs(new_first[2] - first_anchor[2]) != 1:
                continue
            # Propagate the rest of the segment as a rigid translation of the original
            # relative offsets, anchored at new_first.
            base_old_first = old_sites[first_moving]
            dx = new_first[0] - base_old_first[0]
            dy = new_first[1] - base_old_first[1]
            dz = new_first[2] - base_old_first[2]
            # Build candidate positions for all moving residues.
            cand = {}
            valid = True
            for idx in moving:
                o = old_sites[idx]
                cand[idx] = (o[0] + dx, o[1] + dy, o[2] + dz)
            # Check self-avoidance: no candidate collides with a fixed residue.
            for idx in moving:
                c = cand[idx]
                if c in occupied and occupied[c] not in moving_set:
                    valid = False
                    break
            if not valid:
                continue
            # Check the last moving residue is adjacent to the last anchor.
            lc = cand[last_moving]
            if abs(lc[0] - last_anchor[0]) + abs(lc[1] - last_anchor[1]) + abs(lc[2] - last_anchor[2]) != 1:
                continue
            # Compute the gained contacts.
            gained = 0
            for idx in moving:
                if idx not in h_pos:
                    continue
                c = cand[idx]
                for q in _neighbours(c):
                    j = occupied.get(q)
                    if j is None or j in moving_set or abs(idx - j) <= 1:
                        continue
                    gained += 1
            rot_delta = delta - gained  # delta is losses (positive), gained subtracted
            if best_rotation is None or rot_delta < best_rotation_delta:
                best_rotation = cand
                best_rotation_delta = rot_delta

        if best_rotation is None:
            return False

        # Acceptance for the pivot move.
        if _accept(best_rotation_delta, current_energy + best_rotation_delta):
            # Apply the move.
            for idx in moving:
                occupied.pop(old_sites[idx])
            for idx in moving:
                occupied[best_rotation[idx]] = idx
                chain[idx] = best_rotation[idx]
            current_energy += best_rotation_delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
            return True
        return False

    def _accept(delta, new_energy):
        if new_energy < current_energy:
            return True
        if use_annealing:
            T = T_start * (cooling_rate ** iters)
            if T > 0:
                if delta < 0:
                    return True
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    return True
        return False

    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1

        improved = False
        if rng.random() < pivot_prob and n >= 5:
            improved = _try_pivot_move()
        else:
            improved = _try_single_move()

        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search for construction and hybrid
    (single-residue + pivot) simulated annealing for refinement.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = float('inf')

    n = len(sequence)

    # Determine beam width based on chain length.
    if n <= 48:
        beam_width = 200
        num_bs_runs = 5
    elif n <= 80:
        beam_width = 150
        num_bs_runs = 3
    else:
        beam_width = 100
        num_bs_runs = 2

    total_time_available = _remaining_time()
    bs_time_budget = total_time_available * 0.45
    ls_time_budget = total_time_available * 0.55

    bs_deadline = time.time() + bs_time_budget

    # Run Beam Search.
    for run in range(num_bs_runs):
        if time.time() > bs_deadline:
            break
        chain = _beam_search_fold(sequence, dimension, beam_width, rng, bs_deadline)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain

    # Fallback: fast randomized growth (guarantees a valid chain quickly).
    if best_chain is None:
        for _ in range(30):
            if time.time() > bs_deadline + 1.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is not None and len(chain) == n:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
        if best_chain is None:
            # Absolute fallback: straight line (always self-avoiding).
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = _energy(sequence, best_chain)

    # Local Search Refinement.
    if _remaining_time() > 5.0:
        ls_deadline = time.time() + ls_time_budget
        max_ls_iters = max(2000, min(60000, 400 * n))
        improved_chain = _local_search(
            sequence, best_chain, max_ls_iters, rng, ls_deadline,
            use_annealing=True, polish=False, pivot_prob=0.4
        )
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

        # Polish pass with lower temperature and more pivot moves.
        if _remaining_time() > 3.0:
            polish_deadline = time.time() + _remaining_time() * 0.8
            max_polish_iters = max(2000, min(60000, 400 * n))
            polished_chain = _local_search(
                sequence, best_chain, max_polish_iters, rng, polish_deadline,
                use_annealing=True, polish=True, pivot_prob=0.5
            )
            polished_energy = _energy(sequence, polished_chain)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished_chain

    # Final safety: ensure a valid self-avoiding chain is returned.
    if best_chain is None or len(best_chain) != n:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

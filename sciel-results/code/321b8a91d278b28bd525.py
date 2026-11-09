"""HP lattice protein folding: beam search + incremental energy SA + local polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 118.0  # Safety margin for 120s budget


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0

    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1

    return -contacts // 2


def _build_contact_info(sequence, chain):
    """Build pos_to_idx and set of H positions for fast lookups."""
    n = len(chain)
    pos_to_idx = {}
    h_positions = set()
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        if sequence[i] == "H":
            h_positions.add(p)
    return pos_to_idx, h_positions


def _count_contacts(sequence, chain, pos_to_idx, h_positions):
    """Count H-H contacts using precomputed structures."""
    n = len(chain)
    if n < 2:
        return 0
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return contacts // 2


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _local_polish(sequence, chain, pos_to_idx, h_positions, time_limit, rng):
    """Systematic local search: try all short reversals and pivots, keep best."""
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain, pos_to_idx, h_positions)

    current_chain = chain[:]
    current_pos_to_idx = dict(pos_to_idx)
    current_h_pos = set(h_positions)
    current_contacts = _count_contacts(sequence, current_chain, current_pos_to_idx, current_h_pos)
    current_energy = -current_contacts

    best_chain = current_chain[:]
    best_energy = current_energy

    deadline = time.time() + time_limit

    while time.time() < deadline:
        if _remaining_time() < 0.5:
            break

        improved = False

        # Try short reversals (length 2 to 6)
        max_len = min(6, n - 1)
        for length in range(2, max_len + 1):
            for i in range(0, n - length):
                j = i + length
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is None:
                    continue
                # Quick check: just compute energy
                new_e = _energy(sequence, new_chain)
                if new_e < current_energy:
                    current_chain = new_chain
                    current_energy = new_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try random pivots
            for _ in range(5):
                i = rng.randint(1, n - 2)
                angle = rng.choice([1, 2, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                if new_chain is None:
                    continue
                new_e = _energy(sequence, new_chain)
                if new_e < current_energy:
                    current_chain = new_chain
                    current_energy = new_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    improved = True
                    break

        if not improved:
            break

    return best_chain, best_energy


# _grow_csp removed - replaced by improved _grow_beam


def _grow_beam(sequence, rng, beam_width=5, time_limit=3.0):
    """
    Beam Search growth with fast state management.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    # Beam: list of (chain, occupied_set, h_positions_set, heuristic_score)
    initial_chain = [(0, 0)]
    initial_occ = {(0, 0)}
    initial_h = {(0, 0)} if sequence[0] == "H" else set()

    beam = [(initial_chain, initial_occ, initial_h, 0.0)]

    for i in range(1, n):
        new_beam = []

        for chain, occ, h_pos, score in beam:
            current_pos = chain[-1]
            options = []

            for site in _neighbours(current_pos):
                if site in occ:
                    continue

                move_score = 0.0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in h_pos:
                            move_score += 5.0

                free_count = 0
                for q in _neighbours(site):
                    if q not in occ:
                        free_count += 1
                move_score -= free_count * 0.8
                move_score += rng.random() * 0.3

                options.append((move_score, site))

            if not options:
                continue

            options.sort(key=lambda x: x[0], reverse=True)

            # Expand top 2 options from each beam member
            for score_opt, site in options[:2]:
                new_chain = chain + [site]
                new_occ = set(occ)
                new_occ.add(site)
                new_h = set(h_pos)
                if sequence[i] == "H":
                    new_h.add(site)

                total_score = score + score_opt
                new_beam.append((new_chain, new_occ, new_h, total_score))

        if not new_beam:
            return None

        new_beam.sort(key=lambda x: x[3], reverse=True)
        beam = new_beam[:beam_width]

        if _remaining_time() < 5.0:
            # Return best complete chain if we're close to deadline
            # But we're mid-growth, so return None to let fold() handle fallback
            pass

    if beam:
        return beam[0][0]
    return None


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    0 <= i < j < len(chain).
    """
    n = len(chain)
    if i >= j:
        return None

    # Check boundary conditions
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None

    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None

    # Build new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]

    # Check self-avoidance: only need to check new segment against prefix and suffix
    # The reversed segment is self-avoiding (reversal of a SAW)
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])

    for p in new_chain[i:j+1]:
        if p in prefix_set or p in suffix_set:
            return None

    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i].
    angle is 1, 2, or 3 (multiples of 90 degrees).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None

    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]

    # Precompute rotation
    if angle == 1:
        rot = lambda x, y: (-y, x)
    elif angle == 2:
        rot = lambda x, y: (-x, -y)
    else:  # angle == 3
        rot = lambda x, y: (y, -x)

    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        nx, ny = rot(x, y)
        rotated_tail.append((pivot[0] + nx, pivot[1] + ny))

    # Check collision with prefix
    prefix_set = set(prefix)
    for p in rotated_tail:
        if p in prefix_set:
            return None

    return prefix + rotated_tail


# _apply_slide removed - not used effectively


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing with incremental energy tracking.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)

    current_chain = initial_chain[:]
    pos_to_idx, h_positions = _build_contact_info(sequence, current_chain)
    current_contacts = _count_contacts(sequence, current_chain, pos_to_idx, h_positions)
    current_energy = -current_contacts

    best_chain = current_chain[:]
    best_energy = current_energy

    T = 1.5
    cooling_rate = 0.997

    deadline = time.time() + time_limit

    stall_counter = 0
    max_stall = 200

    while time.time() < deadline:
        if _remaining_time() < 0.5:
            break

        # Choose move type
        r = rng.random()
        if r < 0.6:
            # Reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            # Pivot
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
            if new_chain is None:
                continue

        # Compute new energy (full recomputation for simplicity, but this is the bottleneck)
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy

        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            stall_counter = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stall_counter += 1

        T *= cooling_rate
        if T < 0.03:
            if rng.random() < 0.3:
                T = 1.2
            else:
                break

        if stall_counter > max_stall:
            # Perturb with a larger move
            for _ in range(3):
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_energy = _energy(sequence, current_chain)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    break
            stall_counter = 0

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using Beam Search growth,
    local polish, and SA.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0  # Energy is negative

    # Phase 1: Generate diverse initial chains using Beam Search
    initial_chains = []

    configs = [
        (8, 42),
        (12, 123),
        (5, 777),
        (10, 999),
        (15, 555)
    ]

    for width, seed in configs:
        if _remaining_time() < 15.0:
            break
        local_rng = random.Random(seed)
        chain = _grow_beam(sequence, local_rng, beam_width=width, time_limit=2.0)
        if chain is not None and len(chain) == n and _is_valid_chain(chain):
            initial_chains.append(chain)

    # If no valid chains, fallback
    if not initial_chains:
        chain = _grow_beam(sequence, rng, beam_width=1, time_limit=1.0)
        if chain and len(chain) == n:
            initial_chains.append(chain)
        else:
            # Spiral fallback
            chain = []
            x, y = 0, 0
            dx, dy = 1, 0
            for step in range(n):
                chain.append((x, y))
                if step < n - 1:
                    x += dx
                    y += dy
                    if step % 2 == 1:
                        dx, dy = -dy, dx
            initial_chains.append(chain)

    # Evaluate and sort initial chains by energy
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))

    scored_chains.sort(key=lambda x: x[0])

    # Keep top K candidates
    top_k = min(4, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]

    # Phase 2: Local polish + Anneal each candidate
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / len(candidates)

    for idx, chain in enumerate(candidates):
        if _remaining_time() < 3.0:
            break

        alloc_time = min(time_per_candidate, _remaining_time() - 1.5)
        if alloc_time < 0.5:
            continue

        # Split time: 30% for local polish, 70% for SA
        polish_time = alloc_time * 0.3
        sa_time = alloc_time * 0.7

        # Local polish
        pos_to_idx, h_positions = _build_contact_info(sequence, chain)
        polished_chain, polished_e = _local_polish(
            sequence, chain, pos_to_idx, h_positions, polish_time, rng
        )

        # SA from polished state
        best_c, best_e = _anneal(sequence, polished_chain, sa_time, rng)

        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c

    # Fallback
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

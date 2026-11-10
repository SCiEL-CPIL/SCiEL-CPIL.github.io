"""HP lattice protein folding on cubic lattice - MCTS growth + spatial-hash local search."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # hard safety margin


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Energy change when moving residue i from old_pos to new_pos.
    Returns delta_e such that new_energy = old_energy + delta_e."""
    if sequence[i] != 'H':
        return 0
    delta_e = 0
    x, y, z = old_pos
    for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta_e += 1
    x, y, z = new_pos
    for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta_e -= 1
    return delta_e


def _mcts_grow(sequence, dimension, rng, deadline):
    """Generate initial chain via Monte Carlo Tree Search over growth decisions."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # MCTS parameters
    C = 1.414  # exploration constant
    max_sim_per_node = 3
    total_budget = deadline - time.time()
    if total_budget < 0.5:
        total_budget = 0.5

    # Pre-compute H positions for quick reference
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')

    class Node:
        __slots__ = ('state', 'children', 'visits', 'value', 'parent', 'action', 'done')
        def __init__(self, state, parent=None, action=None):
            self.state = state  # (chain_tuple, occupied_dict)
            self.children = {}
            self.visits = 0
            self.value = 0.0
            self.parent = parent
            self.action = action
            self.done = len(state[0]) == n

    root_state = ((0, 0, 0), {(0, 0, 0): 0})
    root = Node(root_state)

    def _simulate(chain, occupied):
        """Fast greedy rollout to completion."""
        chain = list(chain)
        occupied = dict(occupied)
        cur_len = len(chain)
        for i in range(cur_len, n):
            last = chain[-1]
            prev = chain[-2] if len(chain) >= 2 else None
            best_site = None
            best_score = -1e9
            x, y, z = last
            for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                if site in occupied:
                    continue
                if prev is not None and site == prev:
                    continue
                score = 0.0
                # Compactness
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                cz = sum(p[2] for p in chain) / len(chain)
                score -= (abs(site[0]-cx) + abs(site[1]-cy) + abs(site[2]-cz)) * 0.2
                # H-H contact gain
                if i in h_positions:
                    sx, sy, sz = site
                    for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            score += 4.0
                score += rng.random() * 0.3
                if score > best_score:
                    best_score = score
                    best_site = site
            if best_site is None:
                # Fallback
                for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                    if site not in occupied:
                        best_site = site
                        break
                if best_site is None:
                    best_site = (x+1, y, z)
            chain.append(best_site)
            occupied[best_site] = i
        return chain, occupied

    def _select(node):
        """Select leaf node using UCB1."""
        while not node.done:
            if not node.children:
                return node
            # Expand if needed
            if node.visits < max_sim_per_node:
                # Create a child
                chain, occupied = node.state
                i = len(chain)
                last = chain[-1]
                prev = chain[-2] if len(chain) >= 2 else None
                x, y, z = last
                candidates = []
                for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                    if site in occupied:
                        continue
                    if prev is not None and site == prev:
                        continue
                    candidates.append(site)
                if not candidates:
                    node.done = True
                    return node
                site = rng.choice(candidates)
                new_chain = chain + (site,)
                new_occ = dict(occupied)
                new_occ[site] = i
                child = Node((new_chain, new_occ), parent=node, action=site)
                node.children[site] = child
                return child
            # Select best child by UCB1
            best_child = None
            best_ucb = -1e9
            for child in node.children.values():
                if child.visits == 0:
                    ucb = 1e9
                else:
                    ucb = child.value / child.visits + C * math.sqrt(math.log(node.visits + 1) / child.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            node = best_child
        return node

    def _evaluate(node):
        """Evaluate a node via simulation."""
        if node.done:
            chain, _ = node.state
            return _compute_energy(sequence, chain)
        chain, occupied = node.state
        # Run a few simulations
        best_e = 0
        for _ in range(max_sim_per_node):
            sim_chain, _ = _simulate(chain, occupied)
            e = _compute_energy(sequence, sim_chain)
            if e < best_e:
                best_e = e
        return best_e

    # MCTS iterations
    max_iters = int(total_budget * 8000)  # ~8000 iters/sec
    max_iters = max(max_iters, 500)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        if (it + 1) % 200 == 0 and time.time() > deadline:
            break

        node = _select(root)
        reward = _evaluate(node)
        # Backpropagate
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    # Extract best path from root
    chain, occupied = root.state
    cur = root
    while not cur.done:
        if not cur.children:
            break
        best_child = None
        best_val = -1e9
        for child in cur.children.values():
            if child.visits > 0:
                avg = child.value / child.visits
                if avg > best_val:
                    best_val = avg
                    best_child = child
            else:
                if best_val == -1e9:
                    best_child = child
        if best_child is None:
            break
        cur = best_child
        chain, occupied = cur.state

    return list(chain)


def _greedy_grow(sequence, dimension, rng):
    """Fast greedy growth as fallback."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')

    for i in range(1, n):
        last = chain[-1]
        prev = chain[-2] if len(chain) >= 2 else None
        x, y, z = last
        best_site = None
        best_score = -1e9
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            if prev is not None and site == prev:
                continue
            score = 0.0
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            score -= (abs(site[0]-cx) + abs(site[1]-cy) + abs(site[2]-cz)) * 0.3
            if i in h_positions:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 5.0
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                if site not in occupied:
                    best_site = site
                    break
            if best_site is None:
                best_site = (x+1, y, z)
        chain.append(best_site)
        occupied[best_site] = i

    return chain


def _initial_chain(sequence, dimension, rng, deadline):
    """Generate initial chain using MCTS with greedy fallback."""
    n = len(sequence)
    if n == 0:
        return []

    # Allocate time: 60% for MCTS, 40% for greedy fallback if needed
    mcts_time = (deadline - time.time()) * 0.55
    mcts_deadline = time.time() + max(mcts_time, 0.3)

    chain = _mcts_grow(sequence, dimension, rng, mcts_deadline)
    if chain is None or len(chain) != n:
        chain = _greedy_grow(sequence, dimension, rng)

    # Verify self-avoidance
    if len(set(chain)) != n:
        chain = _greedy_grow(sequence, dimension, rng)

    return chain


def _try_single_move(replica, sequence, occupied, rng, T):
    """Attempt a single-residue move. Returns (success, delta_e, new_occupied)."""
    n = len(replica)
    if n < 3:
        return False, 0, occupied

    i = rng.randint(1, n - 2)
    old = replica[i]
    prev = replica[i-1]
    next_pos = replica[i+1]

    px, py, pz = prev
    candidates = []
    for site in ((px+1,py,pz),(px-1,py,pz),(px,py+1,pz),(px,py-1,pz),(px,py,pz+1),(px,py,pz-1)):
        if site == old:
            continue
        if site in occupied and occupied[site] != i:
            continue
        dist = abs(site[0]-next_pos[0]) + abs(site[1]-next_pos[1]) + abs(site[2]-next_pos[2])
        if dist == 1:
            candidates.append(site)

    if not candidates:
        return False, 0, occupied

    site = rng.choice(candidates)
    delta_e = _delta_energy_move(sequence, replica, i, old, site, occupied)

    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
        new_occupied = dict(occupied)
        new_occupied[old] = -1
        new_occupied[site] = i
        replica[i] = site
        return True, delta_e, new_occupied

    return False, 0, occupied


def _try_2opt_move(replica, sequence, occupied, rng, T):
    """Attempt a 2-opt move (reverse segment). Returns (success, delta_e, new_occupied)."""
    n = len(replica)
    if n < 4:
        return False, 0, occupied

    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)

    segment = replica[i:j+1]
    new_segment = list(reversed(segment))

    valid = True
    if i > 0:
        prev_pos = replica[i-1]
        first_new = new_segment[0]
        if abs(prev_pos[0]-first_new[0]) + abs(prev_pos[1]-first_new[1]) + abs(prev_pos[2]-first_new[2]) != 1:
            valid = False
    if valid and j < n - 1:
        next_pos = replica[j+1]
        last_new = new_segment[-1]
        if abs(next_pos[0]-last_new[0]) + abs(next_pos[1]-last_new[1]) + abs(next_pos[2]-last_new[2]) != 1:
            valid = False

    if not valid:
        return False, 0, occupied

    # Check self-avoidance
    occ_set = set()
    for idx, pos in enumerate(replica):
        if idx < i or idx > j:
            occ_set.add(pos)
    for pos in new_segment:
        if pos in occ_set:
            return False, 0, occupied

    # Compute energy delta
    occ_map = {}
    for idx, pos in enumerate(replica):
        if idx < i or idx > j:
            occ_map[pos] = idx

    old_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            pos = replica[idx]
            x, y, z = pos
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    old_cc += 1

    new_cc = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            new_pos = replica[i+j-idx]
            x, y, z = new_pos
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                oj = occ_map.get(q)
                if oj is not None and sequence[oj] == 'H':
                    new_cc += 1

    delta_e = -(new_cc - old_cc)

    if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
        for k in range(j - i + 1):
            replica[i + k] = new_segment[k]

        new_occupied = {}
        for idx, pos in enumerate(replica):
            new_occupied[pos] = idx
        return True, delta_e, new_occupied

    return False, 0, occupied


def _simulated_annealing(sequence, chain, rng, deadline, num_reps=4):
    """Multi-replica simulated annealing with spatial-hash occupancy."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    T_start = 8.0
    T_end = 0.05

    replicas = [list(chain) for _ in range(num_reps)]
    replica_energies = [_compute_energy(sequence, c) for c in replicas]
    replica_occupied = [{p: i for i, p in enumerate(c)} for c in replicas]

    best_energy = min(replica_energies)
    best_chain = list(replicas[replica_energies.index(best_energy)])

    total_time = deadline - time.time()
    if total_time <= 0:
        return best_chain, best_energy

    target_iters = int(total_time * 8000)
    target_iters = max(target_iters, 1000)

    for it in range(target_iters):
        if time.time() > deadline:
            break

        frac = it / max(target_iters - 1, 1)
        T = T_start * (T_end / T_start) ** frac

        r = rng.randint(0, num_reps - 1)

        if rng.random() < 0.65:
            success, delta_e, new_occ = _try_single_move(
                replicas[r], sequence, replica_occupied[r], rng, T)
            if success:
                replica_occupied[r] = new_occ
                replica_energies[r] += delta_e
                if replica_energies[r] < best_energy:
                    best_energy = replica_energies[r]
                    best_chain = list(replicas[r])
        else:
            success, delta_e, new_occ = _try_2opt_move(
                replicas[r], sequence, replica_occupied[r], rng, T)
            if success:
                replica_occupied[r] = new_occ
                replica_energies[r] += delta_e
                if replica_energies[r] < best_energy:
                    best_energy = replica_energies[r]
                    best_chain = list(replicas[r])

        # Replica exchange
        if rng.random() < 0.15 and num_reps > 1:
            r1 = rng.randint(0, num_reps - 2)
            r2 = r1 + 1
            E1, E2 = replica_energies[r1], replica_energies[r2]
            if E1 < E2 and r1 > r2:
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]
            elif E2 < E1 and r2 > r1:
                replicas[r1], replicas[r2] = replicas[r2], replicas[r1]
                replica_energies[r1], replica_energies[r2] = replica_energies[r2], replica_energies[r1]
                replica_occupied[r1], replica_occupied[r2] = replica_occupied[r2], replica_occupied[r1]

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via MCTS growth + multi-replica SA."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))

    # Per-sequence time budget
    per_seq_time = 11.0
    deadline = time.time() + per_seq_time

    # Hard global deadline guard
    if time.time() > _HARD_DEADLINE - 5.0:
        # Not enough time, return a quick greedy fold
        chain = _greedy_grow(sequence, dimension, rng)
        if len(set(chain)) != n:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]

    # Generate initial chain via MCTS
    chain = _initial_chain(sequence, dimension, rng, deadline)
    if chain is None or len(chain) != n:
        chain = [(i, 0, 0) for i in range(n)]

    # Verify self-avoidance
    if len(set(chain)) != n:
        chain = _greedy_grow(sequence, dimension, rng)
        if len(set(chain)) != n:
            chain = [(i, 0, 0) for i in range(n)]

    # Run simulated annealing with remaining time
    sa_deadline = min(deadline, _HARD_DEADLINE - 5.0)
    if time.time() < sa_deadline - 0.5:
        chain, energy = _simulated_annealing(sequence, chain, rng, sa_deadline, num_reps=4)
    else:
        energy = _compute_energy(sequence, chain)

    # Final verification
    if len(set(chain)) != n:
        chain = [(i, 0, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)

    return [list(p) for p in chain]

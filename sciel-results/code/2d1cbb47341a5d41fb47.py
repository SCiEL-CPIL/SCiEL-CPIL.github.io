"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _neighbours_tuple(point):
    """Return tuple of 4 neighbours of a 2D lattice point."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _build_contact_map(sequence, chain):
    """Build a set of all H-H contacts (as frozensets of index pairs) and total count."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours_tuple(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add(frozenset((i, j)))
    return contacts, len(contacts)


def _energy_delta(sequence, chain, old_pos, new_pos, idx):
    """Compute energy delta when moving residue idx from old_pos to new_pos."""
    where = {p: i for i, p in enumerate(chain)}
    delta = 0
    # Remove contacts involving idx at old_pos
    if sequence[idx] == "H":
        for q in _neighbours_tuple(old_pos):
            j = where.get(q)
            if j is not None and j != idx and sequence[j] == "H":
                if j > idx + 1 or j < idx - 1:
                    delta += 1
    # Add contacts involving idx at new_pos
    if sequence[idx] == "H":
        for q in _neighbours_tuple(new_pos):
            j = where.get(q)
            if j is not None and j != idx and sequence[j] == "H":
                if j > idx + 1 or j < idx - 1:
                    delta -= 1
    return delta


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0, 0)
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
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    if len(chain) < 4:
        return None
    
    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]
    
    # Precompute fixed part positions for fast collision check
    fixed_set = set(chain[:pivot_idx])
    
    # Try different rotations
    for angle in [90, -90, 180, 270]:
        # Rotate segment around pivot
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:  # 270
                ndx, ndy = dy, -dx
            new_pos = (pivot[0] + ndx, pivot[1] + ndy)
            # Check collision with fixed part (exclude pivot itself)
            if new_pos in fixed_set and new_pos != pivot:
                valid = False
                break
            rotated.append(new_pos)
        
        if not valid:
            continue
        
        # Check for self-collisions within the rotated segment
        rotated_set = set(rotated)
        if len(rotated_set) != len(rotated):
            continue
        
        # Check unit steps between rotated segment and fixed part, and within rotated
        # Check connection at pivot
        if pivot_idx > 0:
            prev_pos = chain[pivot_idx - 1]
            dx = abs(rotated[0][0] - prev_pos[0])
            dy = abs(rotated[0][1] - prev_pos[1])
            if dx + dy != 1:
                continue
        
        # Check unit steps within rotated segment
        for i in range(1, len(rotated)):
            dx = abs(rotated[i][0] - rotated[i-1][0])
            dy = abs(rotated[i][1] - rotated[i-1][1])
            if dx + dy != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_chain = chain[:pivot_idx] + rotated
        return new_chain
    
    return None


def _slither_move(chain, rng):
    """Slither move: move the last residue to a random empty neighbor of its current position."""
    if len(chain) < 3:
        return None
    
    last_idx = len(chain) - 1
    last_pos = chain[last_idx]
    
    # Get empty neighbors of the last position
    empty_neighbors = [q for q in _neighbours_tuple(last_pos) if q not in set(chain)]
    
    if not empty_neighbors:
        return None
    
    new_pos = rng.choice(empty_neighbors)
    new_chain = chain[:-1] + [new_pos]
    return new_chain


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a short segment (length 2-3) around two pivot points."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length
    seg_len = rng.choice([2, 3])
    
    # Choose start index such that segment fits
    max_start = n - seg_len - 2
    if max_start < 1:
        return None
    start = rng.randint(1, max_start)
    
    # The segment is chain[start:start+seg_len]
    # Pivot points are chain[start-1] and chain[start+seg_len]
    pivot1 = chain[start - 1]
    pivot2 = chain[start + seg_len]
    
    segment = chain[start:start + seg_len]
    
    # For 2D, we can try rotating the segment by 90, -90, 180 degrees
    # around the midpoint or try all valid positions
    # Simpler approach: try all 4 rotations of each point relative to pivot1
    
    # Actually, for crankshaft in 2D, we need to find new positions for the segment
    # such that they connect pivot1 to pivot2 and are self-avoiding
    
    # Try rotating the entire segment around pivot1
    for angle in [90, -90, 180]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot1[0], p[1] - pivot1[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            rotated.append((pivot1[0] + ndx, pivot1[1] + ndy))
        
        # Check if the last point connects to pivot2
        if rotated[-1] != pivot2:
            continue
        
        # Check self-avoidance with the rest of the chain
        new_chain = chain[:start] + rotated + chain[start + seg_len:]
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            # Check unit steps
            valid = True
            for i in range(1, len(new_chain)):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _mcts_fold(sequence, dimension, rng, time_budget=10.0, max_depth=None):
    """
    Monte Carlo Tree Search for protein folding.
    
    Each node represents a partial conformation (prefix of the chain).
    We use UCB to balance exploration and exploitation.
    """
    seq_len = len(sequence)
    if max_depth is None:
        max_depth = seq_len
    
    class Node:
        __slots__ = ['chain', 'occupied', 'energy', 'visits', 'value', 'children', 'parent', 'residue_idx']
        
        def __init__(self, chain, occupied, energy, residue_idx, parent=None):
            self.chain = chain
            self.occupied = occupied
            self.energy = energy
            self.residue_idx = residue_idx
            self.parent = parent
            self.visits = 0
            self.value = 0.0
            self.children = []
    
    # Root node: start at (0,0)
    root = Node([(0, 0)], {(0, 0): 0}, 0, 0)
    
    start_time = time.time()
    exploration_const = 1.4142  # sqrt(2)
    
    best_chain = None
    best_energy = 0
    
    def select(node):
        """Select a node using UCB."""
        while node.residue_idx < seq_len - 1 and node.children:
            # Compute UCB for each child
            best_child = None
            best_ucb = -1e9
            for child in node.children:
                if child.visits == 0:
                    ucb = 1e9
                else:
                    ucb = child.value / child.visits + exploration_const * (
                        (2 * (node.visits ** 0.5) / child.visits) ** 0.5
                    )
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            node = best_child
        return node
    
    def expand(node):
        """Expand a node by adding one more residue."""
        if node.residue_idx >= seq_len - 1:
            return None
        
        last_pos = node.chain[-1]
        residue_idx = node.residue_idx + 1
        
        options = []
        for site in _neighbours(last_pos):
            if site in node.occupied:
                continue
            # Compute immediate contact gain
            gain = 0
            if sequence[residue_idx] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in node.occupied and node.occupied[q] < residue_idx - 1 
                           and sequence[node.occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in node.occupied)
            score = gain * 2.0 + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort by score and pick the best few
        options.sort(key=lambda x: -x[0])
        # Add up to 4 children (top scoring options)
        added = 0
        for score, site in options[:4]:
            new_chain = node.chain + [site]
            new_occupied = dict(node.occupied)
            new_occupied[site] = residue_idx
            new_energy = node.energy - gain  # gain is positive, so energy decreases
            child = Node(new_chain, new_occupied, new_energy, residue_idx, node)
            node.children.append(child)
            added += 1
            if added >= 4:
                break
        
        if node.children:
            # Return the best child
            return max(node.children, key=lambda c: c.energy)
        return None
    
    def simulate(node):
        """Simulate from a node to completion using greedy growth."""
        if node.residue_idx >= seq_len - 1:
            return node.energy
        
        # Greedy completion
        chain = list(node.chain)
        occupied = dict(node.occupied)
        energy = node.energy
        
        for i in range(node.residue_idx + 1, seq_len):
            last_pos = chain[-1]
            best_site = None
            best_score = -1e9
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 
                               and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 3.0 + 0.1 * free + rng.random() * 0.3
                if score > best_score:
                    best_score = score
                    best_site = site
            if best_site is None:
                return energy  # Stuck, return current energy
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(best_site)
                           if q in occupied and occupied[q] < i - 1 
                           and sequence[occupied[q]] == "H")
                energy -= gain
            occupied[best_site] = i
            chain.append(best_site)
        
        return energy
    
    def backpropagate(node, reward):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    # MCTS main loop
    iterations = 0
    while _time_left() > 25:
        if time.time() - start_time > time_budget:
            break
        iterations += 1
        
        # Selection
        node = select(root)
        
        # Expansion
        if node.residue_idx < seq_len - 1:
            expanded = expand(node)
            if expanded is not None:
                node = expanded
        elif not node.children:
            # Leaf node, use simulation
            pass
        
        # Simulation
        reward = simulate(node)
        
        # Backpropagation
        backpropagate(node, reward)
        
        # Track best complete chain
        if node.residue_idx == seq_len - 1:
            if node.energy < best_energy:
                best_energy = node.energy
                best_chain = list(node.chain)
    
    return best_chain, best_energy


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0, use_crankshaft=True):
    """Simulated annealing local search to improve a conformation with incremental energy."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_contacts, n_contacts = _build_contact_map(sequence, current_chain)
    current_energy = -n_contacts
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 2.0
    cooling = 0.995
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Choose move type
        if use_crankshaft and rng.random() < 0.3:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _pivot_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        # Recompute energy for the new chain
        new_contacts, new_n_contacts = _build_contact_map(sequence, new_chain)
        new_energy = -new_n_contacts
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_contacts = new_contacts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    # Deterministic polish pass: accept only strictly improving moves
    polish_chain = list(best_chain)
    polish_contacts, polish_n = _build_contact_map(sequence, polish_chain)
    polish_energy = -polish_n
    polish_time_limit = min(2.0, max(0.5, time_limit * 0.2))
    polish_start = time.time()
    
    for _ in range(200):
        if time.time() - polish_start > polish_time_limit:
            break
        if _time_left() < 8:
            break
        
        # Try all single-residue moves
        improved = False
        for idx in range(1, len(polish_chain) - 1):
            old_pos = polish_chain[idx]
            for new_pos in _neighbours_tuple(old_pos):
                if new_pos in set(polish_chain):
                    continue
                # Check if this move is valid (unit steps maintained)
                if idx > 0:
                    prev = polish_chain[idx - 1]
                    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                        continue
                if idx < len(polish_chain) - 1:
                    nxt = polish_chain[idx + 1]
                    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                        continue
                
                # Compute energy delta
                delta = _energy_delta(sequence, polish_chain, old_pos, new_pos, idx)
                if delta < 0:
                    polish_chain[idx] = new_pos
                    polish_energy += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if polish_energy < best_energy:
        best_chain = polish_chain
        best_energy = polish_energy
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using MCTS + local search hybrid."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        mcts_time = 15.0
        num_grow = 200
        local_search_iters = 3000
        local_search_time = 10.0
        num_ls_chains = 30
    elif seq_len <= 50:
        mcts_time = 12.0
        num_grow = 150
        local_search_iters = 2000
        local_search_time = 8.0
        num_ls_chains = 25
    elif seq_len <= 70:
        mcts_time = 10.0
        num_grow = 100
        local_search_iters = 1500
        local_search_time = 6.0
        num_ls_chains = 20
    else:
        mcts_time = 8.0
        num_grow = 80
        local_search_iters = 1000
        local_search_time = 4.0
        num_ls_chains = 15
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First: Try MCTS
    if _time_left() > 40:
        mcts_chain, mcts_energy = _mcts_fold(sequence, dimension, rng, time_budget=mcts_time)
        if mcts_chain is not None:
            best_chain = mcts_chain
            best_energy = mcts_energy
    
    # Second: Greedy growth with varying bias
    for i in range(num_grow):
        if _time_left() < 20:
            break
        
        # Vary the bias to explore different strategies
        if i < num_grow // 4:
            bias = 1.0
        elif i < num_grow // 2:
            bias = 1.5
        elif i < 3 * num_grow // 4:
            bias = 0.5
        else:
            bias = 2.0
        
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Third: Local search on best chains
    if best_chain is not None and _time_left() > 30:
        # Try local search on the best chain
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time, use_crankshaft=True)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Try local search on a few other good chains
        for i in range(num_ls_chains):
            if _time_left() < 25:
                break
            
            chain = _grow(sequence, dimension, rng, bias=1.0 + 0.5 * rng.random())
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 4:  # Only improve chains close to best
                improved = _local_search(sequence, chain, max_iterations=local_search_iters // 2,
                                          time_limit=local_search_time / 2, use_crankshaft=True)
                if improved is not None:
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

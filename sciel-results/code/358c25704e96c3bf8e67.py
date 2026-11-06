# EVOLVE-BLOCK-START
import time
import math
import random
import heapq
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _dfs_fold(sequence, dimension, time_limit):
    """
    Depth-first search with pruning to find a high-energy fold.
    Uses a simple heuristic to guide the search.
    """
    n = len(sequence)
    if n == 0: return []
    if n == 1: return [[0,0,0]]
    
    best_chain = None
    best_energy = -float('inf')
    
    # Precompute H positions
    h_indices = [i for i, res in enumerate(sequence) if res == 'H']
    
    # State: (path, occupied_set, current_energy, last_pos)
    # We use a stack for DFS to manage memory, but we need to be careful with depth.
    # For n=48, DFS might explode. We use a limited DFS with backtracking.
    
    # To make it feasible, we use a "guided" DFS where we prioritize moves that
    # create contacts or move H residues closer to other H residues.
    
    # Heuristic: For the next residue, prefer moves that are adjacent to existing H's
    # if the next residue is H, or moves that keep the chain compact.
    
    def dfs(path, occupied, energy, i, time_left):
        nonlocal best_chain, best_energy
        
        if time_left <= 0:
            return
            
        if i == n:
            if energy < best_energy:
                best_energy = energy
                best_chain = list(path)
            return
            
        current_pos = path[-1]
        res = sequence[i]
        
        # Generate neighbors
        neighbors = list(_neighbours(current_pos))
        
        # Score neighbors
        scored_neighbors = []
        for nb in neighbors:
            if nb in occupied:
                continue
            
            score = 0
            if res == 'H':
                for q in _neighbours(nb):
                    if q in occupied:
                        j = list(occupied.keys()).index(q) # This is slow. Use a dict.
                        # Optimization: occupied is a dict {pos: index}
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10 # Strong preference for immediate contact
            else:
                # For P, prefer moves that don't block potential H-H contacts?
                # Or just random order.
                score = random.random()
                
            scored_neighbors.append((score, nb))
            
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        for score, nb in scored_neighbors:
            new_occupied = occupied.copy()
            new_occupied[nb] = i
            
            new_energy = energy
            if res == 'H':
                # Recalculate contact gain for speed? 
                # We already estimated it in score. Let's just add the actual gain.
                for q in _neighbours(nb):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            new_energy -= 1 # Energy decreases (more negative) for more contacts? 
                            # Wait, _energy returns -contacts. So energy decreases.
                            # In DFS, we track 'energy' as the value from _energy.
                            # So new_energy = energy - 1 if contact formed.
            
            new_path = path + [nb]
            
            # Pruning: If the maximum possible future energy is worse than best, skip?
            # Max future contacts: (n - i - 1) * 4 ? Too loose.
            # Let's just rely on time limit and backtracking.
            
            dfs(new_path, new_occupied, new_energy, i + 1, time.time() - start_time)
            
            if time.time() - start_time > time_limit:
                return

    start_time = time.time()
    init_pos = (0, 0, 0)
    init_occupied = {init_pos: 0}
    dfs([init_pos], init_occupied, 0, 1, time_limit)
    
    if best_chain:
        return [list(p) for p in best_chain]
    return None

def _sa_fold(sequence, chain, dimension, time_limit):
    """
    Simulated Annealing with Pivot Moves.
    """
    if not chain:
        return chain, 0
        
    current_chain = [tuple(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 5.0
    alpha = 0.995
    
    n = len(current_chain)
    
    while time.time() < start_time + time_limit:
        if n < 4:
            break
            
        # Random pivot
        pivot_idx = random.randint(1, n - 2)
        
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        bond_vector = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if bond_vector[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        r_axis = random.choice(rot_axes)
        angle = random.choice([1, -1])
        
        head_set = set(current_chain[:pivot_idx + 1])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1: nx, ny, nz = dx, -dz, dy
                else: nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1: nx, ny, nz = dz, dy, -dx
                else: nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1: nx, ny, nz = -dy, dx, dz
                else: nx, ny, nz = dy, -dx, dz
                
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(new_chain)
        else:
            prob = math.exp(-delta_e / T)
            if random.random() < prob:
                current_chain = new_chain
                current_energy = new_energy
                
        T *= alpha
        
    return best_chain, best_energy

def _constructive_fold(sequence, dimension, time_limit):
    """
    A smarter constructive method:
    1. Place H residues on a compact grid first (greedy).
    2. Fill P residues to connect them.
    This is hard to implement robustly. 
    Instead, we use a multi-start beam search but with a better heuristic:
    "Contact Potential" = number of unplaced H residues that are within distance 2 of the current tip.
    """
    n = len(sequence)
    if n == 0: return []
    if n == 1: return [[0,0,0]]
    
    best_chain = None
    best_energy = 0
    
    # Multi-start with different random seeds
    for seed in range(5):
        rng = random.Random(seed)
        
        # Beam Search
        width = 50
        beam = [(0, [(0,0,0)], {(0,0,0): 0})] # (energy, path, occupied)
        
        for i in range(1, n):
            next_beam = []
            seen = {}
            res = sequence[i]
            
            for energy, path, occ in beam:
                last_pos = path[-1]
                
                # Generate candidates
                candidates = []
                for nb in _neighbours(last_pos):
                    if nb in occ:
                        continue
                    
                    new_occ = occ.copy()
                    new_occ[nb] = i
                    
                    new_energy = energy
                    if res == 'H':
                        for q in _neighbours(nb):
                            if q in new_occ:
                                j = new_occ[q]
                                if j < i - 1 and sequence[j] == 'H':
                                    new_energy -= 1
                                    
                    # Heuristic: prefer moves that keep the chain compact
                    # Calculate distance from origin
                    dist = sum(abs(x) for x in nb)
                    
                    # Prefer lower distance (compactness) and higher energy
                    # Score = new_energy * 10 - dist
                    score = new_energy * 10 - dist
                    
                    candidates.append((score, nb, new_energy, new_occ))
                    
                # Sort by score
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Keep top few
                for score, nb, new_energy, new_occ in candidates[:10]:
                    new_path = path + [nb]
                    key = (nb, frozenset(new_occ.keys()))
                    
                    if key in seen:
                        if new_energy < seen[key][0]:
                            seen[key] = (new_energy, new_path, new_occ)
                    else:
                        seen[key] = (new_energy, new_path, new_occ)
                        
            # Convert to list and keep top W
            next_beam = list(seen.values())
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:width]
            
            if not beam:
                break
                
        if beam:
            # Take the best from beam
            best_e, best_p, _ = beam[0]
            if best_chain is None or best_e < best_energy:
                best_energy = best_e
                best_chain = [list(p) for p in best_p]
                
    return best_chain

def fold(sequence, dimension):
    """
    Hybrid strategy:
    1. Constructive Beam Search (Multi-start)
    2. Local Optimization (SA)
    """
    start_time = time.time()
    global_deadline = start_time + 280.0 # Safety margin
    
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # 1. Constructive Beam Search
    # Allocate 20% of time to beam search
    beam_time = (global_deadline - start_time) * 0.2
    chain = _constructive_fold(sequence, dimension, beam_time)
    
    if chain is not None:
        # 2. Local Optimization
        opt_time = (global_deadline - time.time()) * 0.8
        opt_chain, opt_energy = _sa_fold(sequence, chain, dimension, opt_time)
        if opt_chain is not None:
            best_chain = opt_chain
            best_energy = opt_energy
    else:
        # Fallback: Random + SA
        rng = random.Random(12345 + len(sequence))
        for attempt in range(5):
            chain = [(0, 0, 0)]
            occupied = {chain[0]: 0}
            for i in range(1, n):
                neighbors = list(_neighbours(chain[-1]))
                rng.shuffle(neighbors)
                placed = False
                for nb in neighbors:
                    if nb not in occupied:
                        chain.append(nb)
                        occupied[nb] = i
                        placed = True
                        break
                if not placed:
                    break
            if len(chain) == n:
                opt_time = (global_deadline - time.time()) * 0.8
                if opt_time > 5:
                    opt_chain, opt_energy = _sa_fold(sequence, chain, dimension, opt_time)
                    if opt_chain is not None:
                        if best_chain is None or opt_energy < best_energy:
                            best_chain = opt_chain
                            best_energy = opt_energy
                if time.time() > global_deadline - 10:
                    break

    if best_chain is None:
        # Absolute fallback
        best_chain = [[i, 0, 0] for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time


def _energy(sequence, chain):
    """Compute the HP energy: -number of non-adjacent H-H contacts."""
    if chain is None:
        return 0
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == "H":
            px, py = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (px + dx, py + dy)
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    if i < j:
                        contacts += 1
    return -contacts


def _grow_sa(sequence, dimension, rng, max_iter=5000, cooling=0.999):
    """
    Simulated annealing on a grid representation.
    We represent the walk as a sequence of grid points.
    Moves: 
      1. Rotate a segment (pivot move) - pick a pivot, rotate the tail.
      2. Local move: try to move a single interior point to a nearby empty spot 
         (but this breaks unit steps easily, so we stick to pivot moves and 
         endpoint moves).
    
    Actually, for self-avoiding walks on a lattice, standard moves are:
    - Pivot move: pick a pivot point, rotate the part of the chain after the pivot
      by 90, 180, or 270 degrees around the pivot.
    - We need to check self-avoidance after the move.
    
    Let's implement a robust pivot-based SA.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n

    # Initialize with a random self-avoiding walk using a simple random walk 
    # that backtracks if stuck, or just start with a straight line and let SA fix it.
    # Better: generate a random valid walk using a biased random walk.
    # For simplicity and speed, let's start with a straight line and use pivot moves.
    # But a straight line has 0 energy. SA needs to explore.
    
    # Let's try to generate a decent initial walk using a simple random walk 
    # with backtracking (depth-first search with random ordering).
    def _random_walk(n, rng):
        path = [(0, 0)]
        visited = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs(depth):
            if depth == n:
                return True
            rng.shuffle(directions)
            for dx, dy in directions:
                new_pos = (path[-1][0] + dx, path[-1][1] + dy)
                if new_pos not in visited:
                    visited.add(new_pos)
                    path.append(new_pos)
                    if dfs(depth + 1):
                        return True
                    path.pop()
                    visited.remove(new_pos)
            return False
        
        if not dfs(1):
            # Fallback: straight line
            return [(i, 0) for i in range(n)]
        return path

    current_chain = _random_walk(n, rng)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 1.0
    for it in range(max_iter):
        # Pivot move
        if n <= 2:
            break
        pivot_idx = rng.randint(0, n - 2)  # Pivot between pivot_idx and pivot_idx+1
        # Actually, pivot is a point. Let's pick a point index k from 1 to n-2.
        # Rotate the segment from k to n-1 around point k.
        k = rng.randint(1, n - 2)
        pivot = current_chain[k]
        # Get the tail
        tail = current_chain[k:]
        
        # Choose rotation: 90, 180, 270 degrees
        rot = rng.choice([1, 2, 3])
        new_tail = []
        for i, (x, y) in enumerate(tail):
            # Relative to pivot
            rx, ry = x - pivot[0], y - pivot[1]
            if rot == 1:  # 90 deg clockwise
                nx, ny = ry, -rx
            elif rot == 2:  # 180 deg
                nx, ny = -rx, -ry
            else:  # 270 deg clockwise (90 ccw)
                nx, ny = -ry, rx
            new_tail.append((pivot[0] + nx, pivot[1] + ny))
        
        # Check if new_tail is valid (self-avoiding and no collision with head)
        # Head is current_chain[:k]
        head_set = set(tuple(p) for p in current_chain[:k])
        new_tail_set = set(tuple(p) for p in new_tail)
        
        # Check for collisions
        valid = True
        if new_tail_set & head_set:
            valid = False
        if len(new_tail_set) != len(new_tail):
            valid = False
        
        # Also check that the connection between head and tail is valid
        # The first point of new_tail is the pivot itself, so it's fine.
        # But we need to ensure the step from head[-1] to new_tail[0] is unit?
        # No, pivot is at index k. head is [:k], so head[-1] is k-1.
        # new_tail[0] is k.
        # The step from k-1 to k is already in the original chain and was valid.
        # Wait, if we rotate the tail, the point at index k stays the same.
        # So the connection k-1 -> k is unchanged.
        
        if valid:
            new_chain = current_chain[:k] + new_tail
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        T *= cooling
    
    return best_chain


def _min_conflicts_fold(sequence, rng, max_iters=10000, max_restarts=3):
    """
    Min-conflicts style local search on grid representation.
    We maintain a self-avoiding walk and use pivot moves + endpoint moves.
    Objective: minimize energy (maximize H-H contacts).
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n

    def _random_walk(n, rng):
        path = [(0, 0)]
        visited = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs(depth):
            if depth == n:
                return True
            rng.shuffle(directions)
            for dx, dy in directions:
                new_pos = (path[-1][0] + dx, path[-1][1] + dy)
                if new_pos not in visited:
                    visited.add(new_pos)
                    path.append(new_pos)
                    if dfs(depth + 1):
                        return True
                    path.pop()
                    visited.remove(new_pos)
            return False
        
        if not dfs(1):
            return [(i, 0) for i in range(n)]
        return path

    best_chain = None
    best_energy = float('inf')
    
    for restart in range(max_restarts):
        current_chain = _random_walk(n, rng)
        current_energy = _energy(sequence, current_chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)
        
        no_improve = 0
        for it in range(max_iters):
            # Choose move type: pivot (80%) or endpoint (20%)
            move_type = rng.random()
            
            if move_type < 0.8 and n > 2:
                # Pivot move
                k = rng.randint(1, n - 2)
                pivot = current_chain[k]
                tail = current_chain[k:]
                rot = rng.choice([1, 2, 3])
                new_tail = []
                for (x, y) in tail:
                    rx, ry = x - pivot[0], y - pivot[1]
                    if rot == 1:
                        nx, ny = ry, -rx
                    elif rot == 2:
                        nx, ny = -rx, -ry
                    else:
                        nx, ny = -ry, rx
                    new_tail.append((pivot[0] + nx, pivot[1] + ny))
                
                head_set = set(tuple(p) for p in current_chain[:k])
                new_tail_set = set(tuple(p) for p in new_tail)
                
                if new_tail_set & head_set or len(new_tail_set) != len(new_tail):
                    continue
                
                new_chain = current_chain[:k] + new_tail
            else:
                # Endpoint move: try to move the last point to a different neighbor
                if n < 2:
                    continue
                last = current_chain[-1]
                prev = current_chain[-2]
                candidates = []
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    np_ = (prev[0] + dx, prev[1] + dy)
                    if np_ != last and tuple(np_) not in set(tuple(p) for p in current_chain[:-1]):
                        candidates.append(np_)
                if not candidates:
                    continue
                new_last = rng.choice(candidates)
                new_chain = current_chain[:-1] + [new_last]
            
            new_energy = _energy(sequence, new_chain)
            
            # Min-conflicts: accept if better, or with small probability if equal
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            elif new_energy == current_energy and rng.random() < 0.3:
                current_chain = new_chain
                no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > 2000:
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using simulated annealing with pivot moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Use multiple restarts with different seeds to escape local minima
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worse than any valid fold
    
    # Number of restarts depends on time budget. Let's do a few.
    # Total time budget is 120s for all sequences.
    # We have 10 sequences. So ~12s per sequence.
    # Let's allocate time per sequence dynamically or just use a fixed number of iterations.
    
    # Let's use a time-based approach to be safe.
    start_time = time.time()
    time_limit = 10.0  # seconds per sequence, leaving buffer
    
    restarts = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Increase max_iter for later restarts if time permits? 
        # Or just use a fixed reasonable number.
        # Let's use 2000 iterations per restart.
        max_iter = 2000
        seed = 12345 + len(sequence) + restarts * 1000
        rng = random.Random(seed)
        
        chain = _grow_sa(sequence, dimension, rng, max_iter=max_iter)
        
        if chain is not None:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        
        restarts += 1
        # Stop if we've done enough restarts
        if restarts >= 5:
            break
    
    # Also try min-conflicts approach
    if time.time() - start_time < time_limit * 0.5:
        rng2 = random.Random(99999 + len(sequence))
        mc_chain = _min_conflicts_fold(sequence, rng2, max_iters=5000, max_restarts=2)
        if mc_chain is not None:
            mc_energy = _energy(sequence, mc_chain)
            if best_chain is None or mc_energy < best_energy:
                best_chain = mc_chain
                best_energy = mc_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

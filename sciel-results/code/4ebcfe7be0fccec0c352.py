"""HP lattice protein folding: constraint-satisfaction backtracking with lookahead + local refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp
from heapq import heappush, heappop

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _backtrack_fold(sequence, dimension, max_time=5.0):
    """
    Constraint-satisfaction backtracking with lookahead.
    Places residues one by one, using a priority queue to explore
    promising positions first. Uses time budget to limit search.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    start_time = time.time()
    
    # Use iterative deepening with a simple backtracking approach
    # For each residue, we try placing it at neighbor positions of the previous residue
    
    def backtrack(chain, occupied, idx):
        nonlocal best_chain, best_energy
        nonlocal start_time
        
        if time.time() - start_time > max_time:
            return
        
        if idx == n:
            # Complete chain found
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            return
        
        # Get candidate positions (neighbors of last placed residue)
        last_pos = chain[-1]
        candidates = []
        for pos in _neighbours(last_pos):
            if pos not in occupied:
                # Calculate immediate gain (contacts formed)
                gain = 0
                if sequence[idx] == "H":
                    for q in _neighbours(pos):
                        if q in occupied:
                            j = list(occupied.keys())[list(occupied.values()).index(occupied[q])]
                            # Actually, let's store index in occupied
                            pass
                candidates.append(pos)
        
        if not candidates:
            return
        
        # Sort candidates by heuristic: prefer positions that form H-H contacts
        def score_candidate(pos):
            if sequence[idx] != "H":
                return 0
            s = 0
            for q in _neighbours(pos):
                if q in occupied:
                    j = occupied[q]
                    if j < idx - 1 and sequence[j] == "H":
                        s += 1
            return s
        
        candidates.sort(key=score_candidate, reverse=True)
        
        for pos in candidates:
            chain.append(pos)
            occupied[pos] = idx
            backtrack(chain, occupied, idx + 1)
            chain.pop()
            del occupied[pos]
            
            if time.time() - start_time > max_time:
                return
    
    # Start backtracking from origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Try multiple random orderings to escape local minima
    rng = random.Random(42)
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # Add some randomness to the ordering
        def backtrack_random(chain, occupied, idx):
            nonlocal best_chain, best_energy
            nonlocal start_time
            
            if time.time() - start_time > max_time:
                return
            
            if idx == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
                return
            
            last_pos = chain[-1]
            candidates = []
            for pos in _neighbours(last_pos):
                if pos not in occupied:
                    candidates.append(pos)
            
            if not candidates:
                return
            
            # Randomize the order slightly
            rng.shuffle(candidates)
            
            for pos in candidates:
                chain.append(pos)
                occupied[pos] = idx
                backtrack_random(chain, occupied, idx + 1)
                chain.pop()
                del occupied[pos]
                
                if time.time() - start_time > max_time:
                    return
        
        backtrack_random(chain, occupied, 1)
    
    if best_chain is not None:
        return best_chain
    else:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]


def _local_move(sequence, chain, rng):
    """
    Perform a single local move (pivot or slide) on the chain.
    Returns the new chain if the move is valid (self-avoiding), else None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Try a slide move: pick a random position and try to move it
    # by one step in a random direction, if it doesn't collide.
    i = rng.randint(1, n - 2)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rng.shuffle(directions)
    
    for dx, dy in directions:
        new_pos = (chain[i][0] + dx, chain[i][1] + dy)
        
        # Check unit step from previous
        if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
            continue
        
        # Check unit step to new position from next
        if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
            continue
        
        # Check no collision with other positions
        collides = False
        for j, pos in enumerate(chain):
            if j == i:
                continue
            if pos == new_pos:
                collides = True
                break
        
        if not collides:
            new_chain = chain.copy()
            new_chain[i] = new_pos
            return new_chain
    
    # Try a pivot move: rotate the tail
    if n >= 4:
        pivot = rng.randint(1, n - 3)
        tail = chain[pivot + 1:]
        
        # Get the direction from pivot to tail[0]
        dx = tail[0][0] - chain[pivot][0]
        dy = tail[0][1] - chain[pivot][1]
        
        # Try rotating by 90, 180, 270 degrees
        rotations = [
            (-dy, dx),    # 90 deg
            (-dx, -dy),   # 180 deg
            (dy, -dx),    # 270 deg
        ]
        
        for rd, rd2 in rotations:
            # Check if the new direction is valid (unit step)
            if abs(rd) + abs(rd2) != 1:
                continue
            
            new_tail = [None] * len(tail)
            new_tail[0] = (chain[pivot][0] + rd, chain[pivot][1] + rd2)
            
            # Check if new_tail[0] collides with head
            head_set = set(chain[:pivot+1])
            if new_tail[0] in head_set:
                continue
            
            # Check if the rest of the tail can be placed
            valid = True
            for k in range(1, len(tail)):
                # The direction from new_tail[k-1] to new_tail[k] should be the same
                # as the original direction from tail[k-1] to tail[k]
                odx = tail[k][0] - tail[k-1][0]
                ody = tail[k][1] - tail[k-1][1]
                new_tail[k] = (new_tail[k-1][0] + odx, new_tail[k-1][1] + ody)
                
                if new_tail[k] in head_set or new_tail[k] in new_tail[:k]:
                    valid = False
                    break
            
            if valid:
                new_chain = chain[:pivot+1] + new_tail
                return new_chain
    
    return None


def _anneal(sequence, chain, max_steps, rng):
    """
    Perform simulated annealing on the chain.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break
        
        T = T_start * (cooling ** step)
        
        new_chain = _local_move(sequence, chain, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using backtracking + simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # First, try backtracking to get a good initial solution
    initial_chain = _backtrack_fold(sequence, dimension, max_time=3.0)
    
    if initial_chain is None:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    best_chain = initial_chain
    best_energy = _energy(sequence, best_chain)
    
    # Now apply simulated annealing to refine
    time_budget = _remaining_time() - 2.0
    
    # Estimate steps per run based on time budget
    # For n=100, each step takes ~0.0001s, so ~10000 steps per second
    steps_per_run = max(1000, int(5000 * (100 / n)))
    
    num_runs = 3
    for run in range(num_runs):
        if _remaining_time() < 1.0:
            break
        
        # Start from the current best
        start_chain = best_chain
        
        # Run annealing
        improved = _anneal(sequence, start_chain, steps_per_run, rng)
        improved_energy = _energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

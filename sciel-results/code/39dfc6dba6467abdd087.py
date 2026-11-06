"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import heapq

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Kill is at 210s wall-clock. Use 195s as a safe deadline.
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

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

def _beam_search(sequence, dimension, beam_width, rng, time_limit):
    """
    Beam search for HP lattice folding.
    State: (chain, occupied_set, energy_so_far)
    We keep the top 'beam_width' partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial state
    start = (0, 0)
    initial_chain = [start]
    initial_occupied = {start}
    initial_energy = 0
    
    # Heap of states: (energy, tiebreaker, chain, occupied)
    # We want min energy (most negative), so we push (energy, random_tiebreak, ...)
    # But we need to keep the best 'beam_width' states.
    # A min-heap is good for extracting the best, but we need to maintain a set of the best.
    # Alternatively, we can use a list and sort, or a heap where we pop the worst if size > beam_width.
    # Let's use a max-heap simulation with negative energy for min-heap behavior?
    # Actually, standard beam search:
    # 1. Expand all states in the beam.
    # 2. Collect all valid children.
    # 3. Select the top 'beam_width' children with lowest energy.
    
    beam = [(initial_energy, rng.random(), initial_chain, initial_occupied)]
    
    for step in range(1, n):
        if _time_remaining() < 5.0:
            break
        
        candidates = []
        current_residue = sequence[step]
        
        for energy, tiebreak, chain, occupied in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                new_energy = energy
                
                # Check for new contacts if current residue is H
                if current_residue == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find index of q in chain
                            # This is O(N) which is slow.
                            # Optimization: store index in occupied dict?
                            # occupied is a set. We need to know if the residue at q is H and index > step-1.
                            # Since we are growing, the only possible contact is with a previous H.
                            # We can maintain a dict of pos -> index for the current chain?
                            # That's expensive to copy.
                            
                            # Alternative: precompute H positions? No, chain changes.
                            # Let's optimize the contact check.
                            # We can keep a set of occupied H positions.
                            # Let's restructure state to include (chain, occupied_set, h_positions_set)
                            
                            pass # Placeholder for now, will restructure below
        
        # The above approach is too slow due to O(N) contact check or large state copies.
        # Let's use a more efficient state representation.
        # State: (chain_list, occupied_set, h_positions_set, energy)
        # We can't easily copy lists/sets.
        
        # Alternative: Backtracking with Beam Search (BFS-like)
        # Let's stick to a simpler Beam Search that tracks the chain as a tuple of coordinates.
        # To speed up contact calculation, we can incrementally update energy.
        
        # Let's restart the beam search implementation with better state management.
        pass

    # Fallback to greedy if beam search is too complex or slow
    return None

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    h_positions = set()
    if sequence[0] == 'H':
        h_positions.add(start)
        
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Check neighbours of site for H
                for q in _neighbours(site):
                    if q in h_positions:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        best_score, site = max(options)
        occupied[site] = i
        chain.append(site)
        if sequence[i] == 'H':
            h_positions.add(site)
    return chain

def _local_search(sequence, chain, max_iterations=100, rng=None):
    """
    Simple local search: try to move a single residue (if possible) or swap?
    Moving a single residue in a chain is hard because of connectivity.
    Instead, let's do "chain breaking and rejoining" or just re-growth from a specific point.
    Or, use a simple "kick" move: pick a random internal bond, break it, and re-attach the tail.
    This is complex.
    
    Let's stick to multiple greedy restarts with different random seeds and slightly different scoring.
    """
    return chain

def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. Multiple randomized greedy growths.
    2. Beam search (if implemented efficiently).
    """
    _init_time()
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Randomized Greedy
    # We have a time budget. Let's run as many as we can.
    # Estimate cost per greedy: O(N^2) roughly due to neighbour checks.
    # For N=100, N^2=10000. 10000 ops is fast.
    # Let's run 1000 greedy trials if time permits.
    
    base_seed = 12345 + len(sequence)
    rng = random.Random(base_seed)
    
    # Determine number of trials based on sequence length and time
    # Shorter sequences can afford more trials.
    # N=20: very fast. N=100: slower.
    # Let's just loop until time runs out, with a minimum of 100 trials.
    
    num_trials = 0
    min_trials = 100
    
    while True:
        remaining = _time_remaining()
        if remaining < 2.0:
            break
        
        # Increase trials for shorter sequences
        if n < 30:
            target_trials = 5000
        elif n < 50:
            target_trials = 1000
        else:
            target_trials = 200
            
        if num_trials >= target_trials:
            break
            
        # Vary the scoring parameters slightly for diversity
        # gain_weight, free_weight
        gain_w = 5.0 + rng.random() * 5.0
        free_w = 0.1 + rng.random() * 0.5
        
        # Custom greedy with these weights
        start = (0, 0)
        chain, occupied = [start], {start: 0}
        h_positions = set()
        if sequence[0] == 'H':
            h_positions.add(start)
            
        success = True
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in h_positions:
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain_w * gain + free_w * free + rng.random()
                options.append((score, site))
            if not options:
                success = False
                break
            best_score, site = max(options)
            occupied[site] = i
            chain.append(site)
            if sequence[i] == 'H':
                h_positions.add(site)
        
        if success:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        num_trials += 1
        
        # Check time more frequently for long sequences
        if n > 50 and num_trials % 10 == 0:
            if _time_remaining() < 2.0:
                break

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

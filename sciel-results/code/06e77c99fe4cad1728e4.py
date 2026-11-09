"""HP lattice protein folding with Simulated Annealing and Regrowth moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _get_neighbors(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _calc_energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
    if not chain:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _get_neighbors(*p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _regrow_suffix(sequence, prefix, start_idx, rng, temp):
    """
    Regrow the chain from start_idx to the end.
    Returns the full chain if successful, None if stuck.
    """
    n = len(sequence)
    if start_idx >= n:
        return list(prefix)
    
    current_chain = list(prefix)
    occupied = {p: i for i, p in enumerate(current_chain)}
    last_pos = current_chain[-1]
    
    for i in range(start_idx, n):
        neighbors = _get_neighbors(*last_pos)
        valid_moves = []
        
        for move in neighbors:
            if move in occupied:
                continue
            
            # Calculate score for this move
            score = 0.0
            if sequence[i] == 'H':
                # Count H-H contacts with existing chain
                for q in _get_neighbors(*move):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1.0
            
            # Count free space (heuristic to avoid trapping)
            free_space = 0
            for q in _get_neighbors(*move):
                if q not in occupied:
                    free_space += 1
            score += free_space * 0.1
            
            # Add noise
            score += rng.random() * temp
            
            valid_moves.append((score, move))
        
        if not valid_moves:
            return None
        
        # Pick the best move (or weighted random? Best is usually fine for regrowth)
        best_move = max(valid_moves, key=lambda x: x[0])[1]
        
        current_chain.append(best_move)
        occupied[best_move] = i
        last_pos = best_move
        
    return current_chain

def _initial_chain(sequence, rng):
    """Create an initial chain using greedy regrowth from start."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        neighbors = _get_neighbors(*chain[-1])
        valid_moves = []
        for move in neighbors:
            if move in occupied:
                continue
            valid_moves.append(move)
        
        if not valid_moves:
            return None # Failed to generate initial chain
        
        # Pick random move for diversity in initial population
        move = rng.choice(valid_moves)
        chain.append(move)
        occupied[move] = i
        
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Regrowth moves.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension] * n
    
    # Time management
    # The evaluator gives 240s for ALL sequences. 
    # We don't know how many sequences are left, so we assume this call is one of many.
    # However, the contract says "fold is called once per benchmark sequence... with a 240s CPU budget for all sequences together".
    # This implies the process runs once, or the budget is shared. 
    # Usually, these sandboxes run the script which loops over sequences. 
    # But here `fold` is the entry point. 
    # If `fold` is called in a loop by the evaluator, we need to be very fast per call.
    # If `fold` is called once and expected to handle the whole task? No, "fold(sequence, dimension) -> list".
    # The error says "timeout: child process ended without a result".
    # This suggests the total time for all calls exceeded the limit.
    # We should allocate a fixed small time budget per call, e.g., 5-10 seconds, to be safe.
    # Let's aim for 8 seconds per sequence to allow for ~30 sequences in 240s.
    
    start_time = time.time()
    time_limit = 8.0  # seconds per sequence
    
    rng = random.Random(42)
    
    # Generate multiple initial chains
    initial_chains = []
    for _ in range(5):
        c = _initial_chain(sequence, rng)
        if c:
            initial_chains.append(c)
            
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
        
    best_chain = initial_chains[0]
    best_energy = _calc_energy(sequence, best_chain)
    
    # Simulated Annealing for each initial chain
    # We will distribute the time among the initial chains
    time_per_chain = time_limit / len(initial_chains)
    
    for chain in initial_chains:
        current_chain = list(chain)
        current_energy = _calc_energy(sequence, current_chain)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        # Estimate steps: regrowth is O(N). N is up to 136.
        # O(N) is fast. We can do many steps.
        while time.time() - start_time < start_time + time_limit - 0.5:
            if time.time() > start_time + time_limit - 0.1:
                break
                
            # Pick a cut point
            # Regrowing the whole tail is expensive. Regrowing a shorter tail is faster but less global.
            # Let's pick a cut between N/3 and N-2.
            if n > 4:
                k = rng.randint(n // 3, n - 2)
            else:
                k = n - 2
                
            prefix = current_chain[:k+1]
            
            new_chain = _regrow_suffix(sequence, prefix, k + 1, rng, T)
            
            if new_chain is None:
                # Failed to regrow, cool down faster or just continue
                T *= cooling
                continue
                
            new_energy = _calc_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
            # Check time frequently
            if time.time() - start_time > start_time + time_limit - 1.0:
                break
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

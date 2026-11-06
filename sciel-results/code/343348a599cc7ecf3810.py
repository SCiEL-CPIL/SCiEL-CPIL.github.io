"""HP lattice protein folding on the square lattice using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Safety margin for 210s limit

def _neighbours(point):
    """Yield the 4 neighbors of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation.
    Energy = - (number of non-consecutive H-H contacts)
    """
    if not chain:
        return 0
    # Map position to index for O(1) lookup
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    # Only count each pair once (i < j)
                    if i < j:
                        contacts += 1
    return -contacts

def _is_valid(chain):
    """Check if the chain is a valid self-avoiding walk."""
    if len(chain) != len(set(chain)):
        return False
    for i in range(len(chain) - 1):
        x1, y1 = chain[i]
        x2, y2 = chain[i + 1]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
    return True

def _grow(sequence, dimension, rng):
    """Randomized greedy growth with H-H clustering bias.
    This implements the 'core set' idea by prioritizing placement of H residues
    near existing H residues to form a dense core.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Pre-compute H indices for faster checking
    h_indices = {i for i, res in enumerate(sequence) if res == 'H'}
    
    for i in range(1, len(sequence)):
        last_pos = chain[-1]
        options = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Calculate a score for this site
            score = 0.0
            
            # 1. Prefer sites that allow new H-H contacts
            if i in h_indices:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        # Check if the residue at j is H and not adjacent in sequence
                        if j in h_indices and abs(i - j) > 1:
                            score += 5.0 # Strong bonus for potential contact
                
            # 2. Prefer sites with more free space (to avoid trapping)
            free_space = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_space += 1
            score += 0.5 * free_space
            
            # 3. Random noise to explore
            score += rng.random() * 2.0
            
            options.append((score, site))
        
        if not options:
            return None
            
        # Select the best option
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _sa_optimize(sequence, initial_chain, rng, time_budget):
    """Simulated Annealing optimization using Reversal and Pivot moves."""
    if not initial_chain:
        return None
        
    chain = list(initial_chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Cooling schedule
    start_temp = 5.0
    end_temp = 0.1
    # Estimate number of steps we can take
    # Rough heuristic: 1000 steps per 0.1s? Let's just use time.
    
    deadline = time.time() + time_budget
    temp = start_temp
    
    while time.time() < deadline:
        if temp < end_temp:
            break
            
        # Generate a move
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.7:
            # Reversal move: Reverse a segment of the chain
            # Choose two random indices i, j with j > i
            n = len(chain)
            if n > 4:
                i = rng.randint(0, n - 4)
                j = rng.randint(i + 2, n - 1)
                
                # Extract segment
                segment = chain[i:j+1]
                # Reverse it
                reversed_segment = segment[::-1]
                
                # Check validity of the new chain
                # The connections at the boundaries must be valid
                # Boundary 1: chain[i-1] to reversed_segment[0]
                # Boundary 2: reversed_segment[-1] to chain[j+1]
                
                candidate = chain[:i] + reversed_segment + chain[j+1:]
                
                # Check self-avoidance and connectivity
                if _is_valid(candidate):
                    new_chain = candidate
        else:
            # Pivot move: Rotate a segment around a pivot
            n = len(chain)
            if n > 4:
                pivot_idx = rng.randint(1, n - 2)
                # Choose a direction to rotate
                # We can try rotating the segment chain[pivot_idx+1:] by 90 or 180 degrees
                # This is complex to implement correctly for SAWs without collisions.
                # Let's stick to Reversal and maybe a simple "Slide" if possible, 
                # but Reversal is the most effective for HP.
                # Let's try a "Crossover" like move: Swap two segments? 
                # No, let's just do more Reversals with different lengths.
                # Or try a small "Pivot" on a short segment.
                
                # Let's try reversing a shorter segment
                length = rng.randint(2, 5)
                i = rng.randint(0, n - length)
                j = i + length
                segment = chain[i:j]
                reversed_segment = segment[::-1]
                candidate = chain[:i] + reversed_segment + chain[j:]
                if _is_valid(candidate):
                    new_chain = candidate

        if new_chain is None:
            # Move was invalid, retry or skip
            # Reduce temperature slightly to keep moving
            temp *= 0.99
            continue

        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        # Geometric cooling
        temp *= 0.995
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    # Check global time limit
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Return a straight line as fallback
        return [[i, 0] for i in range(len(sequence))]
    
    # Allocate time for this sequence
    # We have ~200s total for 10 sequences, so ~20s per sequence.
    # Let's be conservative: 15s per sequence to be safe.
    seq_time_budget = 15.0
    
    # 1. Generate initial conformations using Greedy Growth
    # We want a diverse set of starting points.
    # The "graph" intuition is embedded in the growth bias.
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Generate a few initial chains
    num_initial = 10
    initial_chains = []
    for _ in range(num_initial):
        c = _grow(sequence, dimension, rng)
        if c:
            initial_chains.append(c)
            
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
    
    # 2. Optimize the best few initial chains with SA
    # Sort initial chains by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    # Pick the top 3-5 to optimize, or just the best one if time is tight.
    # Let's optimize the best 2 to balance diversity and depth.
    num_to_optimize = 2
    
    for c in initial_chains[:num_to_optimize]:
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Allocate half the remaining budget to each
        remaining = TIME_LIMIT - 5 - (time.time() - START_TIME)
        alloc = remaining / 2.0
        if alloc < 1.0:
            break
            
        optimized = _sa_optimize(sequence, c, rng, alloc)
        if optimized:
            e = _energy(sequence, optimized)
            if e < best_energy:
                best_energy = e
                best_chain = optimized
                
    # Fallback if SA failed or didn't run
    if best_chain is None:
        best_chain = initial_chains[0]
        
    # Ensure the result is a list of lists of integers
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

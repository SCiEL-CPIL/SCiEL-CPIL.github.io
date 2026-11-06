"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search growth + simulated annealing with kink moves and deadline guard.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 10.0  # 10s safety margin


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
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_grow(sequence, dimension, rng, beam_width=50):
    """Beam search chain growth: keep top beam_width partial chains by contact count."""
    # Each beam entry: (chain, occupied_set, current_contacts)
    start = (0,) * dimension
    beams = [([(start)], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        new_beams = []
        for chain, occupied, contacts in beams:
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                new_occupied = set(occupied)
                new_occupied.add(site)
                new_contacts = contacts
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_contacts += 1
                new_chain = chain + [site]
                new_beams.append((new_chain, new_occupied, new_contacts))
        
        if not new_beams:
            return None
        
        # Keep top beam_width by contacts (higher is better, so more negative energy)
        new_beams.sort(key=lambda x: -x[2])
        beams = new_beams[:beam_width]
    
    # Return the best complete chain
    best_chain = max(beams, key=lambda x: x[2])[0]
    return best_chain


def _kink_move(chain, sequence, dimension, rng):
    """Perform a kink move: pick a random internal segment and try to flip it."""
    n = len(chain)
    if n < 4:
        return None
    
    # Pick a random segment of length 2 or 3 to kink
    seg_len = rng.choice([2, 3])
    if n <= seg_len + 1:
        return None
    
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    # Try all possible new positions for the kinked segment
    best_energy = _energy(sequence, chain)
    best_chain = chain
    
    for _ in range(20):  # Try a few random kink positions
        new_chain = list(chain)
        # Pick a random direction for the kink
        axis = rng.randint(0, dimension - 1)
        step = rng.choice([-1, 1])
        
        # Shift the segment
        valid = True
        for i in range(start, end):
            new_pos = list(chain[i])
            new_pos[axis] += step
            new_pos = tuple(new_pos)
            # Check self-avoidance
            if new_pos in set(chain[:start]) | set(chain[end:]) or (i > start and new_pos in set(chain[start:i])):
                valid = False
                break
            new_chain[i] = new_pos
        
        if not valid:
            continue
        
        # Check unit steps
        valid = True
        for i in range(len(new_chain) - 1):
            diff = sum(abs(new_chain[i][d] - new_chain[i+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
    
    return best_chain


def _simulated_annealing(chain, sequence, dimension, rng, time_budget=5.0):
    """Simulated annealing with kink moves."""
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    T_start = 1.0
    T_end = 0.01
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        T = T_start * (T_end / T_start) ** ((time.time() - start_time) / time_budget)
        
        # Try a kink move
        new_chain = _kink_move(current_chain, sequence, dimension, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search + simulated annealing with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    
    time_budget = max(1.0, min(_time_remaining() / 26.0, 10.0))  # Per-sequence budget
    sa_time = max(0.5, time_budget * 0.6)
    beam_time = max(0.5, time_budget * 0.4)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam search to get a good initial chain
    start_phase = time.time()
    while time.time() - start_phase < beam_time:
        chain = _beam_grow(sequence, dimension, rng, beam_width=50)
        if chain is None:
            break
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        # Check if we've found a good solution
        if best_energy <= -len(sequence) // 4:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing to improve
    remaining = _time_remaining()
    if remaining > 1.0:
        improved_chain, improved_energy = _simulated_annealing(
            best_chain, sequence, dimension, rng, time_budget=min(sa_time, remaining - 0.5)
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Ensure self-avoidance and unit steps
    seen = set()
    for p in best_chain:
        if p in seen:
            # Fallback to straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
            break
        seen.add(p)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

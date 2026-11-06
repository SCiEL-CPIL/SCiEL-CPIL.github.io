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
    """Compute energy as negative number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_growth(sequence, dimension, beam_width, rng):
    """
    Beam search growth: keep top-beam_width partial chains at each step.
    Each partial chain is (chain_list, occupied_dict, energy_so_far).
    Returns the best complete chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial state
    start = (0,) * dimension
    occupied = {start: 0}
    chain = [start]
    energy = 0  # No contacts possible with 1 residue
    
    beams = [(chain, occupied, energy)]
    
    for i in range(1, n):
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute energy gain
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                new_energy = energy - gain  # energy is negative, so subtract gain
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            return None
        
        # Keep top beam_width by energy (most negative is best)
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    # Return the best complete chain
    best_chain, _, best_energy = beams[0]
    return best_chain


def _sa_optimize(sequence, chain, dimension, max_iters, init_temp, min_temp, rng, time_limit):
    """
    Simulated annealing to optimize a chain.
    Uses kink moves and 2-opt reversals.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    current_chain = list(chain)
    
    temp = init_temp
    cooling = (min_temp / init_temp) ** (1.0 / max(1, max_iters))
    
    where = {p: i for i, p in enumerate(current_chain)}
    
    # Precompute contact count for faster energy evaluation
    def _compute_contacts(chain):
        w = {p: i for i, p in enumerate(chain)}
        contacts = 0
        for i, p in enumerate(chain):
            if sequence[i] == "H":
                for q in _neighbours(p):
                    j = w.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
        return contacts
    
    current_contacts = -current_energy  # positive count
    best_contacts = current_contacts
    
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink move: pick a random internal residue and try to rotate it
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            # Find all valid positions for residue i
            candidates = []
            for site in _neighbours(p):
                if site == prev or site == nxt:
                    continue
                if site in where:
                    continue
                # Check that moving doesn't break self-avoidance
                # Also need to check that the new position doesn't conflict
                # with any other residue
                if site not in where:
                    candidates.append(site)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Compute energy change
            old_contacts = 0
            if sequence[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j != i and sequence[j] == "H":
                        if abs(j - i) > 1:
                            old_contacts += 1
            
            # Temporarily remove from where
            del where[p]
            where[new_pos] = i
            current_chain[i] = new_pos
            
            new_contacts = 0
            if sequence[i] == "H":
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and sequence[j] == "H":
                        if abs(j - i) > 1:
                            new_contacts += 1
            
            delta_contacts = new_contacts - old_contacts
            delta_energy = -delta_contacts  # energy = -contacts
            
            if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / max(temp, 1e-10)):
                current_contacts = -current_energy + delta_contacts
                current_energy = -current_contacts
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    best_contacts = current_contacts
            else:
                # Revert
                del where[new_pos]
                where[p] = i
                current_chain[i] = p
        else:
            # 2-opt reversal: pick two indices and reverse the segment
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 2)
            
            # Try reversing segment [i, j]
            old_segment = current_chain[i:j + 1]
            new_segment = list(reversed(old_segment))
            
            # Check if the reversal is valid (self-avoiding)
            valid = True
            # Check endpoints connectivity
            if i > 0:
                if new_segment[0] == current_chain[i - 1]:
                    valid = False
            if j < n - 1:
                if new_segment[-1] == current_chain[j + 1]:
                    valid = False
            
            # Check internal self-avoidance
            if valid:
                temp_where = dict(where)
                for k in range(i, j + 1):
                    del temp_where[current_chain[k]]
                    temp_where[new_segment[k - i]] = k
                # Check no conflicts
                for k in range(i, j + 1):
                    if new_segment[k - i] in where:
                        # This position was occupied by something else
                        # But we already removed all old positions
                        pass
                # Actually need to check if any new position conflicts with unmodified positions
                for k in range(i, j + 1):
                    pos = new_segment[k - i]
                    if pos in where and where[pos] != k:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Compute energy change for the reversal
            # This is complex; for simplicity, compute full energy
            new_chain = current_chain[:i] + new_segment + current_chain[j + 1:]
            new_energy = _energy(sequence, new_chain)
            delta_energy = new_energy - current_energy
            
            if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / max(temp, 1e-10)):
                # Accept
                current_chain = new_chain
                current_energy = new_energy
                where = {p: idx for idx, p in enumerate(current_chain)}
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    best_contacts = -current_energy
        
        # Cool down
        temp = max(temp * cooling, min_temp)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search growth + simulated annealing."""
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # Determine beam width and SA parameters based on chain length
    if n <= 48:
        beam_width = 5000
        sa_iters = 4000
        init_temp = 5.0
        min_temp = 0.003
        num_restarts = 12
    elif n <= 80:
        beam_width = 3000
        sa_iters = 2000
        init_temp = 4.0
        min_temp = 0.005
        num_restarts = 10
    else:
        beam_width = 1500
        sa_iters = 1000
        init_temp = 3.0
        min_temp = 0.01
        num_restarts = 8
    
    best_chain = None
    best_energy = 1  # Very high (energy is negative, so we want more negative)
    
    # Time budget per restart
    total_time_limit = _time_remaining()
    if total_time_limit <= 0:
        # Return a straight line as fallback
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    time_per_restart = total_time_limit / max(1, num_restarts)
    beam_time = time_per_restart * 0.3
    sa_time = time_per_restart * 0.7
    
    for restart in range(num_restarts):
        # Check time
        if _time_remaining() <= 15.0:  # 15s safety margin
            break
        
        # Beam growth
        chain = _beam_growth(sequence, dimension, beam_width, rng)
        if chain is None:
            continue
        
        # SA optimization
        opt_chain, opt_energy = _sa_optimize(
            sequence, chain, dimension, sa_iters, init_temp, min_temp, rng, sa_time
        )
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

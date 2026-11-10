"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None


class SpatialHash:
    """Spatial hash grid for O(1) neighbor lookups."""
    
    def __init__(self):
        self.grid = {}
        self.index = {}
    
    def add(self, pos, idx):
        self.grid[pos] = idx
        self.index[idx] = pos
    
    def remove(self, idx):
        pos = self.index.pop(idx)
        del self.grid[pos]
    
    def get(self, pos):
        return self.grid.get(pos)
    
    def neighbours(self, pos):
        """Yield (neighbor_pos, index) for all occupied neighbors."""
        x, y, z = pos
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nb = (x+dx, y+dy, z+dz)
            idx = self.grid.get(nb)
            if idx is not None:
                yield nb, idx
    
    def is_occupied(self, pos):
        return pos in self.grid
    
    def copy(self):
        sh = SpatialHash()
        sh.grid = dict(self.grid)
        sh.index = dict(self.index)
        return sh


def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    sh = SpatialHash()
    for i, p in enumerate(chain):
        sh.add(p, i)
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q, j in sh.neighbours(p):
                if j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy growth with simulated annealing-like acceptance."""
    start = (0, 0, 0)
    chain = [start]
    sh = SpatialHash()
    sh.add(start, 0)
    
    for i in range(1, len(sequence)):
        candidates = []
        for site in sh.neighbours(chain[-1]):
            site_pos = site[0]
            if sh.is_occupied(site_pos):
                continue
            
            gain = 0
            if sequence[i] == "H":
                for q, j in sh.neighbours(site_pos):
                    if j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            free = 0
            x, y, z = site_pos
            for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                if not sh.is_occupied((x+dx, y+dy, z+dz)):
                    free += 1
            
            score = gain + 0.1 * free + rng.random() * temperature
            candidates.append((score, site_pos))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        if rng.random() < 0.8:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        
        sh.add(site, i)
        chain.append(site)
    
    return chain


def _pivot_move(sequence, chain, sh, i, rng):
    """
    Perform a pivot move at index i.
    Rotates the suffix chain[i:] by 90 degrees around a random axis through chain[i-1].
    Returns (new_chain, new_sh, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0:
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1:
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else:
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity using spatial hash
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        if p in prefix_set:
            return None
    
    new_chain = chain[:i] + new_suffix
    
    # Build new spatial hash
    new_sh = SpatialHash()
    for idx, p in enumerate(new_chain):
        new_sh.add(p, idx)
    
    # Compute energy delta
    # Optimization: Only check residues involved in the move? 
    # Pivots change positions of i..n-1. 
    # Contacts can change for any H in i..n-1 interacting with any H in 0..n-1.
    # However, contacts between two residues both in the suffix that maintain their relative positions? 
    # No, the whole suffix rotates. Relative positions within suffix are preserved, but orientation changes.
    # Actually, the set of pairs (u,v) in the suffix that are neighbors on the lattice is preserved in terms of "distance 1", 
    # but their absolute positions change, so they might collide with prefix or each other (checked above).
    # The energy delta calculation in the original code recalculates contacts for the whole suffix.
    # This is O(N) per move.
    
    old_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q, j in sh.neighbours(p):
                if j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
                    
    new_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q, j in new_sh.neighbours(p):
                if j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
                    
    delta = new_contacts - old_contacts
    
    return new_chain, new_sh, delta


def _parallel_tempering(sequence, initial_chain, rng, deadline):
    """
    Parallel Tempering Monte Carlo search.
    """
    n = len(initial_chain)
    if n < 5:
        return initial_chain, _energy(sequence, initial_chain)
    
    # Setup replicas
    n_replicas = 5
    # Temperatures: Low to High
    # Energy scale is -contacts. Max energy ~ -N. 
    # T should be in range [1, 10] roughly.
    temps = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    # Initialize replicas with the best chain found so far (or copies of it)
    # To diversify, we could run a few more greedy grows here if time permits, 
    # but for simplicity, we start all from the same chain and let PT diverge.
    # Actually, starting from the same chain is okay, PT will mix them.
    
    replicas = []
    for k in range(n_replicas):
        sh = SpatialHash()
        for i, p in enumerate(initial_chain):
            sh.add(p, i)
        replicas.append({
            'chain': list(initial_chain),
            'sh': sh,
            'energy': _energy(sequence, initial_chain),
            'temp': temps[k]
        })
    
    best_chain = initial_chain
    best_energy = replicas[0]['energy']
    
    steps_per_cycle = 10
    swaps_per_cycle = 1
    
    while time.time() < deadline:
        # MC steps for each replica
        for k in range(n_replicas):
            rep = replicas[k]
            for _ in range(steps_per_cycle):
                i = rng.randint(1, n - 1)
                result = _pivot_move(sequence, rep['chain'], rep['sh'], i, rng)
                if result is not None:
                    new_chain, new_sh, delta = result
                    # Metropolis acceptance
                    if delta > 0:
                        rep['chain'] = new_chain
                        rep['sh'] = new_sh
                        rep['energy'] -= delta
                        if rep['energy'] < best_energy:
                            best_energy = rep['energy']
                            best_chain = list(rep['chain'])
                    else:
                        # delta is negative (energy increases, cost increases)
                        # E_new = E_old - delta (since delta = E_new - E_old in contacts? No.)
                        # _energy returns -contacts.
                        # delta = new_contacts - old_contacts.
                        # If delta > 0, contacts increase, energy decreases (good).
                        # If delta < 0, contacts decrease, energy increases (bad).
                        # Metropolis: accept if exp(-delta_E / T) > rng.
                        # delta_E = E_new - E_old.
                        # E_new = -(old_contacts + delta) = -old_contacts - delta = E_old - delta.
                        # So delta_E = -delta.
                        # Accept if exp(-(-delta) / T) = exp(delta / T) > rng.
                        # Since delta is negative here, exp(delta/T) < 1.
                        if rng.random() < math.exp(delta / rep['temp']):
                            rep['chain'] = new_chain
                            rep['sh'] = new_sh
                            rep['energy'] -= delta
                            if rep['energy'] < best_energy:
                                best_energy = rep['energy']
                                best_chain = list(rep['chain'])
        
        # Replica swaps
        for k in range(n_replicas - 1):
            if rng.random() < swaps_per_cycle: # Just do it every time for simplicity
                rep1 = replicas[k]
                rep2 = replicas[k+1]
                
                # Swap acceptance criterion
                # P = min(1, exp(- (T1 - T2) * (E1 - E2) / (T1 * T2))) ? 
                # Standard PT: 
                # P = min(1, exp( - (1/T1 - 1/T2) * (E1 - E2) ))
                # Let's use this.
                inv_T1 = 1.0 / rep1['temp']
                inv_T2 = 1.0 / rep2['temp']
                dE = rep1['energy'] - rep2['energy']
                exponent = -(inv_T1 - inv_T2) * dE
                
                if exponent >= 0 or rng.random() < math.exp(exponent):
                    # Swap
                    replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
                    
        # Check time
        if time.time() > deadline - 0.5:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    budget = min(remaining * 0.95, 20.0) # Increased budget for PT
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Initialization (Greedy Growth)
    # We need a valid chain.
    n_grows = max(5, int((budget - 1.0) * 2)) # More grows to ensure we find a good start
    n_grows = min(n_grows, 20)
    
    grow_deadline = time.time() + (budget - 1.0) * 0.2
    
    for _ in range(n_grows):
        if time.time() > grow_deadline:
            break
        
        # Try different temperatures for growth
        temp = 0.5 + rng.random() * 2.0
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: Linear chain (valid, energy 0)
        # MUST USE TUPLES
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Optimization (Parallel Tempering)
    if time.time() < deadline - 0.5:
        best_chain, best_energy = _parallel_tempering(sequence, best_chain, rng, deadline - 0.5)
    
    return [[p[0], p[1], p[2]] for p in best_chain]

# EVOLVE-BLOCK-END

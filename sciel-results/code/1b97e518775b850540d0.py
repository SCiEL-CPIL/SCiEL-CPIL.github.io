"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _neighbours(point):
    """Yield all 4-neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _neighbours_set(point):
    """Return set of 4-neighbours."""
    x, y = point
    return {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}

def _energy(sequence, chain):
    """Compute HP lattice energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _build_contact_priorities(sequence):
    """
    Build a priority-ordered list of H-H contact pairs.
    Returns: list of (priority, i, j) where priority = sequence distance.
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if abs(i - j) > 1:
                dist = abs(i - j)
                pairs.append((dist, i, j))
    pairs.sort(key=lambda x: -x[0])
    return pairs

def _backtrack_fold(sequence, dimension, time_budget=115.0):
    """
    Backtracking search with contact-map heuristics.
    """
    start_time = time.time()
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    contact_pairs = _build_contact_priorities(sequence)
    
    # Top K contacts for prioritization
    K = max(3, n // 4)
    top_contacts = contact_pairs[:K]
    
    # For each residue, set of top-priority partners
    top_partners = defaultdict(set)
    for dist, i, j in top_contacts:
        top_partners[i].add(j)
        top_partners[j].add(i)
    
    # Precompute neighbour sets for speed
    neigh_cache = {}
    def get_neigh_set(point):
        if point not in neigh_cache:
            neigh_cache[point] = _neighbours_set(point)
        return neigh_cache[point]
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(42 + n)
    
    def backtrack(pos_idx, chain, occupied):
        nonlocal best_chain, best_energy, start_time
        
        if time.time() - start_time > time_budget:
            return "timeout"
        
        if pos_idx == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
            return "complete"
        
        last_pos = chain[-1]
        current_res = sequence[pos_idx]
        
        candidates = []
        for site in get_neigh_set(last_pos):
            if site in occupied:
                continue
            
            score = 0.0
            
            # Score: satisfy top-priority contacts
            if current_res == 'H' and pos_idx in top_partners:
                for partner in top_partners[pos_idx]:
                    if partner < pos_idx:
                        partner_pos = chain[partner]
                        if site in get_neigh_set(partner_pos):
                            dist = abs(pos_idx - partner)
                            score += dist * 50
            
            # Score: local H-H contacts
            if current_res == 'H':
                for q in get_neigh_set(site):
                    j = occupied.get(q)
                    if j is not None and j > pos_idx - 1 and sequence[j] == 'H':
                        score += 5
            
            # Slight random perturbation
            score += rng.random() * 0.3
            
            candidates.append((score, site))
        
        if not candidates:
            return "dead_end"
        
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            chain.append(site)
            occupied[site] = pos_idx
            
            result = backtrack(pos_idx + 1, chain, occupied)
            
            chain.pop()
            del occupied[site]
            
            if result == "timeout":
                return "timeout"
            if result == "complete":
                return "complete"
        
        return "dead_end"
    
    max_attempts = 15
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        result = backtrack(1, chain, occupied)
        
        if result == "timeout":
            break
        if best_chain is not None:
            continue
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation using backtracking
    with contact-map constraint heuristics.
    """
    if len(sequence) <= 2:
        return [[0] * dimension for _ in range(len(sequence))]
    
    try:
        result = _backtrack_fold(sequence, dimension, time_budget=115.0)
        return [list(p) for p in result]
    except Exception:
        rng = random.Random(12345 + len(sequence))
        best = None
        best_energy = 0
        
        for _ in range(50):
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            success = True
            
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
                    options.append((gain + 0.1 * free + rng.random(), site))
                
                if not options:
                    success = False
                    break
                
                site = max(options)[1]
                occupied[site] = i
                chain.append(site)
            
            if success:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best = chain
        
        if best is None:
            best = [(i, 0) for i in range(len(sequence))]
        
        return [list(p) for p in best]

# EVOLVE-BLOCK-END

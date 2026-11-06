# EVOLVE-BLOCK-START
import random
import time
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
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _prepare_sequence(sequence):
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    # Pre-compute contact pairs (i, j) where i < j, abs(i-j) > 1
    pairs = []
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            idx_i = h_indices[i]
            idx_j = h_indices[j]
            if abs(idx_i - idx_j) > 1:
                pairs.append((idx_i, idx_j))
    
    # Map residue index to list of partners
    partners = defaultdict(list)
    for i, j in pairs:
        partners[i].append(j)
        partners[j].append(i)
        
    return pairs, partners

def _backtrack_fold(sequence, dimension, time_budget=100.0):
    start_time = time.time()
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    pairs, partners = _prepare_sequence(sequence)
    
    # We will run multiple independent search attempts
    # Each attempt uses a different random seed and heuristic weighting
    best_chain = None
    best_energy = 0
    
    # Time per attempt
    num_attempts = 10
    time_per_attempt = time_budget / num_attempts
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_budget:
            break
            
        attempt_start = time.time()
        rng = random.Random(12345 + attempt * 100 + n)
        
        # Heuristic weights
        w_contact = 10.0 + rng.random() * 5.0
        w_potential = 1.0 + rng.random() * 2.0
        w_openness = 0.1 + rng.random() * 0.1
        w_random = 0.01
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        current_energy = 0
        
        # To speed up contact checking, we keep track of placed H residues
        # and their positions.
        # However, checking neighbors in `occupied` is O(1) per neighbor.
        
        def backtrack(pos_idx):
            nonlocal best_chain, best_energy
            nonlocal chain, occupied, current_energy
            
            if time.time() - attempt_start > time_per_attempt:
                return False  # Timeout for this attempt
            
            if pos_idx == n:
                # Complete chain
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                return True
            
            last_pos = chain[-1]
            candidates = []
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Calculate heuristic
                score = 0
                immediate_gain = 0
                
                # 1. Immediate contacts
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            immediate_gain += 1
                
                score += immediate_gain * w_contact
                
                # 2. Potential contacts (adjacent to H residues that are close in sequence but not yet placed? 
                #    Or adjacent to already placed H residues that we haven't counted yet? 
                #    Actually, if we are H and neighbor is H, it's counted above if j < pos_idx-1.
                #    If j == pos_idx-1, it's a backbone bond, not a contact.
                #    So immediate_gain is correct for new contacts.
                
                # 3. Openness: prefer sites with more free neighbors to avoid dead ends
                free_neighbors = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_neighbors += 1
                score += free_neighbors * w_openness
                
                # 4. Randomness
                score += rng.random() * w_random
                
                candidates.append((score, site))
            
            if not candidates:
                return False
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            for _, site in candidates:
                # Place
                chain.append(site)
                occupied[site] = pos_idx
                
                # Update energy if this move creates a contact
                delta_energy = 0
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (becomes more negative)
                            
                current_energy += delta_energy
                
                if backtrack(pos_idx + 1):
                    chain.pop()
                    del occupied[site]
                    current_energy -= delta_energy
                    return True
                
                # Undo
                chain.pop()
                del occupied[site]
                current_energy -= delta_energy
                
                if time.time() - attempt_start > time_per_attempt:
                    return False
            
            return False
        
        # Start search
        backtrack(1)
        
        if best_chain is not None:
            # If we found a good solution, we might want to stop early if it's optimal?
            # But we don't know optimal. So continue to try to improve.
            pass

    if best_chain is None:
        # Fallback: simple greedy or straight line
        # Let's do a quick greedy to ensure valid output
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        rng = random.Random(999)
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                options.append((rng.random(), site))
            if not options:
                # Dead end, try to backtrack or just fail
                break
            _, site = max(options)
            occupied[site] = i
            chain.append(site)
        
        if len(chain) < n:
             # Failed to build full chain, use straight line as last resort
             # Note: Straight line is always valid for n>0
             best_chain = [(i, 0) for i in range(n)]
        else:
             best_chain = chain
             
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation using constraint-satisfaction
    backtracking with contact map heuristics.
    """
    if len(sequence) <= 2:
        return [[0] * dimension for _ in range(len(sequence))]
    
    # Use backtracking with contact map heuristics
    try:
        # Allocate 110s to leave buffer for overhead
        result = _backtrack_fold(sequence, dimension, time_budget=110.0)
        return result
    except Exception:
        # Fallback to simple greedy if backtracking fails
        rng = random.Random(12345 + len(sequence))
        best = None
        best_energy = 0
        
        for _ in range(100):
            # Simple greedy growth
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

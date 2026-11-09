"""HP lattice protein folding using CSP with backjumping."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_csp(sequence, dimension, time_limit=10.0):
    n = len(sequence)
    start_time = time.time()
    
    # State: placed positions for each residue index
    # We'll place residues in an order determined by heuristic
    # But the chain must be self-avoiding with unit steps, so we must place in chain order?
    # Actually no - for a self-avoiding walk, consecutive residues must be adjacent.
    # So we MUST place in chain order (residue 0, then 1, then 2, ...).
    # 
    # Wait, that's a key constraint. The walk must have unit steps between consecutive residues.
    # So residue i+1 must be adjacent to residue i. This means we can only place in chain order.
    #
    # So the CSP is: place residue 0, then 1, then 2, ..., n-1 in order,
    # where each new residue must be adjacent to the previous one and not already occupied.
    #
    # The heuristic can be in value ordering: for each position in the chain,
    # try candidate sites in order of expected contact gain.
    #
    # Conflict-directed backjumping: if placing residue i at site s leads to a dead end
    # (no valid placement for residue i+1), we can potentially backjump to an earlier
    # residue that caused the conflict.
    
    # For HP folding, the key is to make good choices early.
    # Let's use a beam search or iterative deepening with good heuristics.
    
    # Actually, let me think about this differently. Since we must place in chain order,
    # the search tree is determined by the choices at each step. The heuristic is:
    # at each step, prefer sites that create more H-H contacts.
    
    # Let's do a randomized restart approach with a smart greedy + local search,
    # but also try a more systematic search for shorter sequences.
    
    best_chain = None
    best_energy = 0  # We want most negative (most contacts)
    
    # For the CSP approach with backjumping:
    # We'll do depth-first search with the constraint that we place in chain order.
    # At each step, we try candidate sites ordered by contact gain.
    # If we get stuck, we backtrack.
    # Conflict-directed backjumping: if a dead end is caused by a specific earlier choice,
    # jump back to that choice rather than all the way to the previous level.
    
    # For simplicity and speed, let's combine:
    # 1. A fast greedy with randomized restarts (like current but better heuristic)
    # 2. For sequences up to ~40 residues, try a more systematic search
    
    # Let me implement a better greedy first, then enhance with local search.
    
    def _greedy_with_lookahead(sequence, dimension, rng):
        """Greedy chain growth with lookahead for contact potential."""
        start = (0,) * dimension
        chain = [start]
        occupied = {start: 0}
        
        for i in range(1, n):
            candidates = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                # Calculate immediate contact gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                gain += 1
                # Calculate future potential: how many free neighbors does this site have?
                # More free neighbors means more room for future growth
                free_count = sum(1 for q in _neighbours(site) if q not in occupied)
                # Also consider: if this is H, how many H residues are nearby (within distance 2)?
                # This gives a sense of potential for future contacts
                potential = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if sequence[j] == 'H':
                                potential += 2  # Strong signal
                            else:
                                potential += 0.5
                        else:
                            # Check if any H residue could potentially be placed here later
                            # This is expensive, so just count free space
                            potential += 0.1
                score = gain * 10 + potential + free_count * 0.1 + rng.random() * 0.5
                candidates.append((score, site))
            
            if not candidates:
                return None
            
            # Pick the best candidate
            candidates.sort(key=lambda x: -x[0])
            site = candidates[0][1]
            occupied[site] = i
            chain.append(site)
        
        return chain
    
    def _local_search(chain, sequence, max_iters=1000):
        """Simple local search: try to improve by swapping or moving single residues."""
        if chain is None:
            return None, 0
        
        current_energy = _compute_energy(sequence, chain)
        best_chain = chain[:]
        best_energy = current_energy
        n = len(sequence)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            improved = False
            # Try moving each residue (except ends) to a neighboring free site
            for i in range(1, n - 1):
                old_pos = chain[i]
                for new_pos in _neighbours(old_pos):
                    # Check if new_pos is free and doesn't conflict with neighbors
                    if new_pos in {chain[j] for j in range(n) if j != i}:
                        continue
                    # Check if the move maintains adjacency with i-1 and i+1
                    # new_pos must be adjacent to chain[i-1] and chain[i+1]
                    if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
                        continue
                    if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
                        continue
                    
                    # Try the move
                    chain[i] = new_pos
                    new_energy = _compute_energy(sequence, chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = chain[:]
                        improved = True
                    else:
                        chain[i] = old_pos
                        # Keep best_chain as is, but continue searching from current state
                        # Actually, for local search, we should revert if not improved
                        # But we're already reverting. Let's keep the best found so far.
                if improved:
                    chain = best_chain[:]
                    break
        
        return best_chain, best_energy
    
    # Run multiple greedy restarts with local search
    import random
    rng = random.Random(12345 + len(sequence))
    
    num_restarts = 50 if n <= 30 else 30 if n <= 50 else 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        chain = _greedy_with_lookahead(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        # Local search
        improved_chain, improved_energy = _local_search(chain, sequence, max_iters=200)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain[:]
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence using CSP-inspired search with local refinement."""
    n = len(sequence)
    
    # Time budget per sequence
    if n <= 20:
        time_limit = 12.0
    elif n <= 40:
        time_limit = 10.0
    elif n <= 60:
        time_limit = 8.0
    else:
        time_limit = 6.0
    
    chain, energy = _fold_csp(sequence, dimension, time_limit=time_limit)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END

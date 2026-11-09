"""Rewritten HP lattice protein folding solver using backtracking with heuristics."""
import random
import math


def _neighbours(point):
    """Yield 4-neighbours on the 2D square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute energy = -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _count_h_residues(sequence):
    return sequence.count("H")


def _backtrack_fold(sequence, dimension, rng, max_nodes=50000):
    """
    Backtracking search with heuristic ordering.
    Tries to place the chain one residue at a time, ordering candidate positions
    by a heuristic score that favors positions with high potential H-H contact density.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better), so start at 0 (no contacts)
    
    # For efficiency, we'll do a limited backtracking search.
    # Given the 120s budget for 10 sequences, we need to be careful.
    # Let's use a randomized restart approach with short backtracking runs.
    
    # Actually, let's implement a proper backtracking with a node limit per attempt,
    # and do multiple attempts with different random seeds/orderings.
    
    # Precompute H positions for quick lookup
    is_h = [c == 'H' for c in sequence]
    
    def backtrack(idx, chain, occupied, nodes_visited):
        nonlocal best_chain, best_energy, nodes_visited
        
        if nodes_visited[0] > max_nodes:
            return
        
        if idx == n:
            # Complete chain found
            # Compute energy
            where = {p: i for i, p in enumerate(chain)}
            contacts = 0
            for i, p in enumerate(chain):
                if is_h[i]:
                    for q in _neighbours(p):
                        j = where.get(q)
                        if j is not None and j > i + 1 and is_h[j]:
                            contacts += 1
            energy = -contacts
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            return
        
        last_pos = chain[-1]
        # Get candidate positions
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Heuristic score: prefer sites that create H-H contacts with earlier H residues
            score = 0
            if is_h[idx]:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < idx - 1 and is_h[j]:
                            score += 10  # Strong preference for creating contacts
            
            # Also prefer sites with more free space (to avoid dead ends)
            free_space = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_space * 0.1
            
            # Add random tie-breaker
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return
        
        # Sort by score descending (try best first)
        candidates.sort(key=lambda x: -x[0])
        
        for _, site in candidates:
            nodes_visited[0] += 1
            chain.append(site)
            occupied[site] = idx
            
            backtrack(idx + 1, chain, occupied, nodes_visited)
            
            chain.pop()
            del occupied[site]
            
            if nodes_visited[0] > max_nodes:
                return
    
    # Do multiple attempts with different random seeds
    attempts = 10
    for attempt in range(attempts):
        rng_attempt = random.Random(12345 + attempt * 7919 + len(sequence))
        best_energy_local = 0
        best_chain_local = None
        nodes_visited = [0]
        
        # Use the global best as the initial best
        best_energy = best_energy_local if best_chain_local is not None else 0
        best_chain = best_chain_local
        
        # Reset for this attempt
        best_energy_for_attempt = 0
        best_chain_for_attempt = None
        
        # We need to modify backtrack to use local best
        # Let's refactor: use a class or closure with local variables
        
        # Simpler: just run backtrack and update global best
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        nodes_visited = [0]
        
        # We'll use a helper that tracks local best
        local_best_energy = 0
        local_best_chain = None
        
        def bt(idx, ch, occ, nv):
            nonlocal local_best_energy, local_best_chain
            if nv[0] > max_nodes:
                return
            if idx == n:
                where = {p: i for i, p in enumerate(ch)}
                contacts = 0
                for i, p in enumerate(ch):
                    if is_h[i]:
                        for q in _neighbours(p):
                            j = where.get(q)
                            if j is not None and j > i + 1 and is_h[j]:
                                contacts += 1
                e = -contacts
                if e < local_best_energy:
                    local_best_energy = e
                    local_best_chain = list(ch)
                return
            
            last_pos = ch[-1]
            candidates = []
            for site in _neighbours(last_pos):
                if site in occ:
                    continue
                score = 0
                if is_h[idx]:
                    for q in _neighbours(site):
                        if q in occ:
                            j = occ[q]
                            if j < idx - 1 and is_h[j]:
                                score += 10
                free_space = sum(1 for q in _neighbours(site) if q not in occ)
                score += free_space * 0.1
                score += rng_attempt.random() * 0.5
                candidates.append((score, site))
            
            if not candidates:
                return
            
            candidates.sort(key=lambda x: -x[0])
            
            for _, site in candidates:
                nv[0] += 1
                ch.append(site)
                occ[site] = idx
                
                bt(idx + 1, ch, occ, nv)
                
                ch.pop()
                del occ[site]
                
                if nv[0] > max_nodes:
                    return
        
        bt(1, chain, occupied, nodes_visited)
        
        if local_best_chain is not None and (best_chain is None or local_best_energy < best_energy):
            best_chain = local_best_chain
            best_energy = local_best_energy
    
    if best_chain is not None:
        return best_chain
    return None


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy self-avoiding conformation on the 2D lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Try backtracking first
    result = _backtrack_fold(sequence, dimension, random.Random(12345 + len(sequence)), max_nodes=30000)
    
    if result is not None:
        # Convert tuples to lists
        return [list(p) for p in result]
    
    # Fallback: greedy growth (from original)
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 1
    
    for _ in range(200):
        start = (0, 0)
        chain, occupied = [start], {start: 0}
        success = True
        for i in range(1, n):
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
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
    
    if best is None:
        # Straight line fallback
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]

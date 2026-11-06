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

def _compute_contact_priorities(sequence):
    """
    Compute priority scores for H-H contact pairs based on:
    1. Distance in sequence (longer range = higher priority for topology)
    2. Number of potential geometric constraints
    """
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    if len(h_indices) < 2:
        return {}
    
    # Score each potential contact pair
    pair_scores = {}
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            idx_i = h_indices[i]
            idx_j = h_indices[j]
            dist = idx_j - idx_i
            if dist <= 1:
                continue
            # Priority: longer range contacts are more topologically important
            if dist < 4:
                score = 1.0
            elif dist < 8:
                score = 2.0
            elif dist < 15:
                score = 3.0
            elif dist < 25:
                score = 4.0
            else:
                score = 5.0
            pair_scores[(idx_i, idx_j)] = score
    
    # Sort by priority (descending)
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: -x[1])
    
    # Select top pairs as hard constraints
    max_hard = min(25, max(10, len(sorted_pairs) // 3))
    hard_pairs = [(p[0], p[1]) for p in sorted_pairs[:max_hard]]
    
    # Build constraint map: residue -> list of (partner, priority)
    constraints = defaultdict(list)
    for (i, j), score in sorted_pairs[:max_hard]:
        constraints[i].append((j, score))
        constraints[j].append((i, score))
    
    return constraints

def _backtrack_fold_csp(sequence, dimension, time_budget=100.0):
    """
    Constraint-satisfaction backtracking with hard constraints for high-priority H-H contacts.
    """
    start_time = time.time()
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    constraints = _compute_contact_priorities(sequence)
    
    best_chain = None
    best_energy = 0
    
    num_attempts = 20
    time_per_attempt = time_budget / num_attempts
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_budget:
            break
            
        attempt_start = time.time()
        rng = random.Random(54321 + attempt * 7919 + n * 13)
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        current_energy = 0
        
        # Track which hard constraints are satisfied
        satisfied_hard = set()
        
        def backtrack(pos_idx, depth=0):
            nonlocal best_chain, best_energy
            nonlocal chain, occupied, current_energy
            nonlocal satisfied_hard
            
            if time.time() - attempt_start > time_per_attempt:
                return False
            
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
                
                # Calculate heuristic score
                score = 0.0
                immediate_gain = 0
                
                # 1. Immediate H-H contacts
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            immediate_gain += 1
                
                score += immediate_gain * 20.0
                
                # 2. Hard constraint satisfaction
                hard_gain = 0
                if sequence[pos_idx] == 'H':
                    for partner, priority in constraints.get(pos_idx, []):
                        if partner < pos_idx - 1:
                            partner_pos = chain[partner]
                            is_adjacent = any(q == partner_pos for q in _neighbours(site))
                            if is_adjacent:
                                hard_gain += priority
                score += hard_gain * 10.0
                
                # 3. Openness: prefer sites with more free neighbors
                free_neighbors = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_neighbors += 1
                score += free_neighbors * 0.5
                
                # 4. Dead-end avoidance
                if sequence[pos_idx] == 'H':
                    potential = 0
                    for partner, priority in constraints.get(pos_idx, []):
                        if partner > pos_idx + 1:
                            potential += 1
                    score -= potential * 0.1
                
                # 5. Small random perturbation
                score += rng.random() * 0.1
                
                candidates.append((score, site, immediate_gain, hard_gain))
            
            if not candidates:
                return False
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            for _, site, gain, hard_gain in candidates:
                # Place
                chain.append(site)
                occupied[site] = pos_idx
                
                # Update energy
                delta_energy = 0
                new_contacts = set()
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            delta_energy -= 1
                            new_contacts.add((min(j, pos_idx), max(j, pos_idx)))
                
                current_energy += delta_energy
                
                # Update satisfied hard constraints
                newly_satisfied = set()
                if sequence[pos_idx] == 'H':
                    for partner, priority in constraints.get(pos_idx, []):
                        if partner < pos_idx - 1:
                            partner_pos = chain[partner]
                            is_adjacent = any(q == partner_pos for q in _neighbours(site))
                            if is_adjacent:
                                newly_satisfied.add((min(partner, pos_idx), max(partner, pos_idx)))
                
                old_satisfied = set(satisfied_hard)
                satisfied_hard.update(newly_satisfied)
                
                if backtrack(pos_idx + 1, depth + 1):
                    chain.pop()
                    del occupied[site]
                    current_energy -= delta_energy
                    satisfied_hard = old_satisfied
                    return True
                
                # Undo
                chain.pop()
                del occupied[site]
                current_energy -= delta_energy
                satisfied_hard = old_satisfied
                
                if time.time() - attempt_start > time_per_attempt:
                    return False
            
            return False
        
        # Start search
        backtrack(1)
        
        if best_chain is not None:
            pass

    if best_chain is None:
        # Fallback: simple greedy
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        rng = random.Random(999)
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
                break
            _, site = max(options)
            occupied[site] = i
            chain.append(site)
        
        if len(chain) < n:
            best_chain = [(i, 0) for i in range(n)]
        else:
            best_chain = chain
             
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation using constraint-satisfaction
    backtracking with hard constraints for high-priority H-H contacts.
    """
    if len(sequence) <= 2:
        return [[0] * dimension for _ in range(len(sequence))]
    
    try:
        result = _backtrack_fold_csp(sequence, dimension, time_budget=110.0)
        return result
    except Exception:
        # Fallback to simple greedy
        rng = random.Random(12345 + len(sequence))
        best = None
        best_energy = 0
        
        for _ in range(100):
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

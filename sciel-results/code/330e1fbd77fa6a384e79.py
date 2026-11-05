"""Rewritten program: Maximum Weight Independent Set via branch-and-bound on compatible (m,r) pairs."""
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # All possible (m, r) pairs with 20 <= m <= 200, 0 <= r < m
    # We want to select a subset that is pairwise disjoint and maximizes sum of 1/m
    
    # Key insight: for a given modulus m, we can only pick ONE residue r.
    # But different moduli can share residues if they're compatible.
    # Two pairs (m1, r1) and (m2, r2) are compatible iff r1 != r2 mod gcd(m1, m2)
    
    # Strategy: Since each modulus m contributes 1/m to the score regardless of which residue r we pick,
    # and the constraint is about residues being pairwise disjoint, we should:
    # 1. Try to include as many small moduli as possible (they have higher weight 1/m)
    # 2. For each included modulus, choose a residue that doesn't conflict with others
    
    # Greedy approach with local search:
    # - Sort moduli by 1/m descending (i.e., m ascending)
    # - For each modulus in order, try to find a valid residue
    # - Use randomized restarts and simulated annealing to escape local minima
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is compatible with all pairs in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def find_valid_residues(m_new, selected):
        """Find all residues r for modulus m_new that are compatible with selected."""
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected):
                valid.append(r)
        return valid
    
    def score_of(selected):
        """Compute the score (sum of 1/m) for a selection."""
        return sum(1.0 / m for m, r in selected)
    
    def greedy_fill(selected, moduli_order, rng):
        """Fill remaining moduli greedily, trying residues in random order."""
        result = list(selected)
        for m in moduli_order:
            if any(m == m_sel for m_sel, _ in result):
                continue  # Already selected
            valid_rs = find_valid_residues(m, result)
            if valid_rs:
                r = rng.choice(valid_rs)
                result.append((m, r))
        return result
    
    def local_search(selected, moduli_order, rng, max_iters=500):
        """Local search: try swapping residues or adding/removing moduli."""
        current = list(selected)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try a random modification
            move_type = rng.random()
            
            if move_type < 0.4 and len(current) > 0:
                # Change residue of a random selected modulus
                idx = rng.randint(0, len(current) - 1)
                m, r_old = current[idx]
                others = [c for i, c in enumerate(current) if i != idx]
                valid_rs = find_valid_residues(m, others)
                if valid_rs:
                    new_r = rng.choice(valid_rs)
                    if new_r != r_old:
                        new_current = current[:]
                        new_current[idx] = (m, new_r)
                        new_score = score_of(new_current)
                        if new_score >= current_score:
                            current = new_current
                            current_score = new_score
                            if current_score > best_score:
                                best = list(current)
                                best_score = current_score
                            
            elif move_type < 0.7:
                # Try adding a random unselected modulus
                selected_mods = {m for m, _ in current}
                candidates = [m for m in moduli_order if m not in selected_mods]
                if candidates:
                    m_new = rng.choice(candidates)
                    valid_rs = find_valid_residues(m_new, current)
                    if valid_rs:
                        r_new = rng.choice(valid_rs)
                        new_current = current + [(m_new, r_new)]
                        new_score = score_of(new_current)
                        if new_score >= current_score:
                            current = new_current
                            current_score = new_score
                            if current_score > best_score:
                                best = list(current)
                                best_score = current_score
                            
            elif move_type < 0.85 and len(current) > 1:
                # Remove a random modulus
                idx = rng.randint(0, len(current) - 1)
                new_current = current[:idx] + current[idx+1:]
                new_score = score_of(new_current)
                # Accept if it allows better future fills (accept with some probability)
                if rng.random() < 0.3:  # Accept removals with some probability
                    current = new_current
                    current_score = new_score
                    
            else:
                # Swap: remove one, add another
                if len(current) > 0:
                    idx = rng.randint(0, len(current) - 1)
                    m_remove = current[idx][0]
                    others = [c for i, c in enumerate(current) if i != idx]
                    selected_mods = {m for m, _ in others}
                    candidates = [m for m in moduli_order if m not in selected_mods]
                    if candidates:
                        m_add = rng.choice(candidates)
                        valid_rs = find_valid_residues(m_add, others)
                        if valid_rs:
                            r_add = rng.choice(valid_rs)
                            new_current = others + [(m_add, r_add)]
                            new_score = score_of(new_current)
                            if new_score >= current_score:
                                current = new_current
                                current_score = new_score
                                if current_score > best_score:
                                    best = list(current)
                                    best_score = current_score
        
        return best
    
    def simulated_annealing(selected, moduli_order, rng, start_temp=1.0, end_temp=0.01, max_iters=3000):
        """Simulated annealing to escape local minima."""
        current = list(selected)
        current_score = score_of(current)
        best = list(current)
        best_score = current_score
        
        num_iters = max_iters
        for i in range(num_iters):
            if time.time() > deadline:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (i / num_iters)
            
            # Generate a neighbor
            move_type = rng.random()
            neighbor = None
            
            if move_type < 0.4 and len(current) > 0:
                # Change residue
                idx = rng.randint(0, len(current) - 1)
                m, r_old = current[idx]
                others = [c for j, c in enumerate(current) if j != idx]
                valid_rs = find_valid_residues(m, others)
                if valid_rs:
                    new_r = rng.choice(valid_rs)
                    neighbor = current[:]
                    neighbor[idx] = (m, new_r)
                    
            elif move_type < 0.7:
                # Add a modulus
                selected_mods = {m for m, _ in current}
                candidates = [m for m in moduli_order if m not in selected_mods]
                if candidates:
                    m_new = rng.choice(candidates)
                    valid_rs = find_valid_residues(m_new, current)
                    if valid_rs:
                        r_new = rng.choice(valid_rs)
                        neighbor = current + [(m_new, r_new)]
                        
            elif move_type < 0.85 and len(current) > 1:
                # Remove a modulus
                idx = rng.randint(0, len(current) - 1)
                neighbor = current[:idx] + current[idx+1:]
                
            else:
                # Swap
                if len(current) > 0:
                    idx = rng.randint(0, len(current) - 1)
                    others = [c for j, c in enumerate(current) if j != idx]
                    selected_mods = {m for m, _ in others}
                    candidates = [m for m in moduli_order if m not in selected_mods]
                    if candidates:
                        m_add = rng.choice(candidates)
                        valid_rs = find_valid_residues(m_add, others)
                        if valid_rs:
                            r_add = rng.choice(valid_rs)
                            neighbor = others + [(m_add, r_add)]
            
            if neighbor is None:
                continue
            
            neighbor_score = score_of(neighbor)
            delta = neighbor_score - current_score
            
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score
        
        return best
    
    # Main search loop
    moduli_order = list(range(20, 201))  # Sorted by m ascending (highest weight first)
    
    best_solution = None
    best_score = 0.0
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts * 7919 + 13)
        
        # Start with greedy fill
        initial = []
        current = greedy_fill(initial, moduli_order, rng)
        current_score = score_of(current)
        
        # Local search
        improved = local_search(current, moduli_order, rng, max_iters=200)
        improved_score = score_of(improved)
        
        # Simulated annealing
        annealed = simulated_annealing(improved, moduli_order, rng, start_temp=0.5, end_temp=0.01, max_iters=1500)
        annealed_score = score_of(annealed)
        
        # Final local search
        final = local_search(annealed, moduli_order, rng, max_iters=300)
        final_score = score_of(final)
        
        if final_score > best_score:
            best_score = final_score
            best_solution = final
    
    # Convert to required format [[m, r], ...]
    if best_solution is None:
        # Fallback: simple greedy
        rng = random.Random(42)
        best_solution = greedy_fill([], moduli_order, rng)
    
    result = [[m, r] for m, r in sorted(best_solution, key=lambda x: x[0])]
    return result

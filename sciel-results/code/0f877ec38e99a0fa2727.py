"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Precompute 1/m as float for speed in B&B
    inv_float = {m: 1.0 / m for m in candidates}
    
    # Build conflict graph: nodes are (m, r) pairs, edges if they conflict
    # Two pairs (m1, r1) and (m2, r2) conflict if:
    #   - m1 == m2 (same modulus, not allowed), OR
    #   - r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # For B&B, we work with nodes indexed by (m, r)
    # But that's up to sum(m for m in 20..200) = ~20910 nodes, too many for full B&B
    # Instead, use a hybrid: B&B over moduli selection with residue optimization
    
    # Key insight: for a fixed set of moduli S, the max score is sum(1/m for m in S)
    # and we need to find residues that make them pairwise disjoint.
    # So the problem reduces to: find max-weight subset S of {20..200} such that
    # there exist residues r_m for each m in S that are pairwise disjoint.
    
    # For a given set S, check if valid residues exist:
    # This is a constraint satisfaction problem. For each pair (m1, m2) in S,
    # we need r1 != r2 mod gcd(m1, m2).
    
    # Strategy: Use branch-and-bound over moduli, with upper bound being
    # sum of 1/m for all remaining moduli (fractional knapsack style).
    # At each node, we have a partial assignment of residues.
    
    # Better approach: Greedy + local search with smarter moves, plus
    # a final B&B polish on the remaining moduli.
    
    def is_disjoint(m_new, r_new, existing):
        """Check if (m_new, r_new) is disjoint from all in existing list of (m, r)."""
        for (m_e, r_e) in existing:
            g = gcd(m_new, m_e)
            if r_new % g == r_e % g:
                return False
        return True
    
    def find_valid_residue(m, existing):
        """Find any valid residue for m given existing assignments."""
        for r in range(m):
            if is_disjoint(m, r, existing):
                return r
        return None
    
    # Phase 1: Greedy construction
    def greedy_init():
        selected = []
        # Sort by 1/m descending (i.e., m ascending)
        for m in sorted(candidates):
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        return selected
    
    best = greedy_init()
    best_score = sum(inv_float[m] for m, r in best)
    
    # Phase 2: Local search with multiple move types
    random.seed(42)
    
    def compute_score(state):
        return sum(inv_float[m] for m, r in state)
    
    def try_add(state, m, r):
        """Try to add (m, r) to state, return new state if valid, else None."""
        used_mods = set(x[0] for x in state)
        if m in used_mods:
            return None
        for (m_e, r_e) in state:
            g = gcd(m, m_e)
            if r % g == r_e % g:
                return None
        return state + [(m, r)]
    
    def try_remove(state, idx):
        return state[:idx] + state[idx+1:]
    
    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.9995
    import math
    
    while T > T_min and time.time() < deadline - 20:
        state = best
        score = best_score
        
        for _ in range(200):
            if time.time() > deadline - 20:
                break
            
            move_type = random.choice(['add', 'remove', 'change_r', 'swap'])
            
            if move_type == 'add':
                used_mods = set(m for m, r in state)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                # Prefer small moduli (high value)
                m_new = random.choice(available[:min(50, len(available))])
                for attempt in range(20):
                    r_new = random.randint(0, m_new - 1)
                    new_state = try_add(state, m_new, r_new)
                    if new_state:
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'remove':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
                delta = new_score - score
                if delta > 0 or random.random() < math.exp(delta / T):
                    state, score = new_state, new_score
            elif move_type == 'change_r':
                if not state:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                for attempt in range(20):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd(m_old, m_e)
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'swap':
                # Remove one element and try to add a different one
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                new_state_without = try_remove(state, idx)
                used_mods = set(m for m, r in new_state_without)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                # Try to add a small modulus
                candidates_sorted = sorted(available, key=lambda x: x)
                for m_new in candidates_sorted[:min(20, len(candidates_sorted))]:
                    for r_new in range(min(m_new, 10)):
                        new_state = try_add(new_state_without, m_new, r_new)
                        if new_state:
                            new_score = compute_score(new_state)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(delta / T):
                                state, score = new_state, new_score
                                break
                    if state != new_state_without:
                        break
        
        if score > best_score:
            best, best_score = state, score
        
        T *= cooling
    
    # Phase 3: Branch and bound polish
    # Try to add more moduli using B&B
    used_mods = set(m for m, r in best)
    remaining = [m for m in candidates if m not in used_mods]
    
    if remaining and time.time() < deadline - 5:
        # Sort remaining by 1/m descending
        remaining.sort()
        
        # B&B: try to add moduli in order, with backtracking
        def bb_search(selected, remaining_mods, current_score):
            nonlocal best, best_score
            if time.time() > deadline - 5:
                return
            if not remaining_mods:
                return
            # Upper bound: current score + sum of 1/m for all remaining
            ub = current_score + sum(inv_float[m] for m in remaining_mods)
            if ub <= best_score:
                return
            
            # Try adding the first modulus
            m = remaining_mods[0]
            rest = remaining_mods[1:]
            
            # Try all residues for m
            for r in range(m):
                if is_disjoint(m, r, selected):
                    new_selected = selected + [(m, r)]
                    new_score = current_score + inv_float[m]
                    if new_score > best_score:
                        best = new_selected
                        best_score = new_score
                    bb_search(new_selected, rest, new_score)
            
            # Also try skipping m
            bb_search(selected, rest, current_score)
        
        # Run B&B with limited depth to avoid explosion
        # Use a simpler approach: try adding each remaining modulus one at a time
        for m in remaining[:30]:  # Only try top 30 remaining
            if time.time() > deadline - 5:
                break
            r = find_valid_residue(m, best)
            if r is not None:
                new_state = best + [(m, r)]
                new_score = best_score + inv_float[m]
                if new_score > best_score:
                    best, best_score = new_state, new_score
    
    # Final local search pass
    if time.time() < deadline - 2:
        T = 0.1
        state = best
        score = best_score
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            move_type = random.choice(['add', 'remove', 'change_r'])
            if move_type == 'add':
                used_mods = set(m for m, r in state)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                m_new = random.choice(available[:min(30, len(available))])
                for attempt in range(15):
                    r_new = random.randint(0, m_new - 1)
                    new_state = try_add(state, m_new, r_new)
                    if new_state:
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'remove':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
                delta = new_score - score
                if delta > 0 or random.random() < math.exp(delta / T):
                    state, score = new_state, new_score
            elif move_type == 'change_r':
                if not state:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                for attempt in range(15):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd(m_old, m_e)
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            state, score = new_state, new_score
                        break
        if score > best_score:
            best, best_score = state, score
    
    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Precompute gcd table
    gcd_table = {}
    for m1 in candidates:
        for m2 in candidates:
            if (m1, m2) not in gcd_table:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
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
    
    def find_all_valid_residues(m, existing):
        """Find all valid residues for m given existing assignments."""
        valid = []
        for r in range(m):
            if is_disjoint(m, r, existing):
                valid.append(r)
        return valid
    
    def compute_score(state):
        return sum(1.0 / m for m, r in state)
    
    # Phase 1: Greedy construction with multiple strategies
    best = []
    best_score = 0.0
    
    # Strategy 1: Ascending moduli (small first, high weight)
    def greedy_ascending():
        selected = []
        for m in sorted(candidates):
            if time.time() > deadline - 60:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        return selected
    
    # Strategy 2: Greedy with residue selection that maximizes future flexibility
    def greedy_flexible():
        selected = []
        for m in sorted(candidates):
            if time.time() > deadline - 60:
                break
            valid_residues = find_all_valid_residues(m, selected)
            if not valid_residues:
                continue
            # Choose residue that minimizes conflicts with remaining moduli
            best_r = valid_residues[0]
            best_conflicts = float('inf')
            for r in valid_residues:
                # Count how many remaining moduli would be blocked by this choice
                conflicts = 0
                for m2 in candidates:
                    if m2 <= m:
                        continue
                    g = gcd(m, m2)
                    if r % g in [r2 % g for (m2, r2) in selected if m2 == m2]:
                        conflicts += 1
                # Simpler: just count how many residues of m2 are blocked
                blocked_count = 0
                for m2 in candidates:
                    if m2 <= m:
                        continue
                    g = gcd(m, m2)
                    # Number of residues r2 in [0, m2) with r2 % g == r % g
                    target = r % g
                    # Count r2 in [0, m2) such that r2 % g == target
                    count = m2 // g + (1 if m2 % g > target else 0)
                    # But we only care about whether it conflicts with existing
                    # Actually, just check if this residue class is already taken by existing
                    for (m_e, r_e) in selected:
                        if m_e == m2:
                            if r_e % g == target:
                                blocked_count += 1
                                break
                if blocked_count < best_conflicts:
                    best_conflicts = blocked_count
                    best_r = r
            selected.append((m, best_r))
        return selected
    
    # Strategy 3: Randomized greedy with multiple restarts
    def greedy_randomized(seed_val):
        rng = random.Random(seed_val)
        selected = []
        # Process moduli in random order but biased toward small
        order = list(candidates)
        # Weight toward small moduli
        for i in range(len(order)):
            # Pick from top portion
            k = max(1, len(order) - i)
            idx = rng.randrange(k)
            order[i], order[idx] = order[idx], order[i]
        
        for m in order:
            if time.time() > deadline - 60:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        return selected
    
    # Run multiple greedy strategies
    states = []
    
    s1 = greedy_ascending()
    sc1 = compute_score(s1)
    states.append((s1, sc1))
    
    s2 = greedy_flexible()
    sc2 = compute_score(s2)
    states.append((s2, sc2))
    
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        s = greedy_randomized(seed)
        sc = compute_score(s)
        states.append((s, sc))
    
    # Pick best from greedy
    for s, sc in states:
        if sc > best_score:
            best, best_score = s, sc
    
    # Phase 2: Local search with multiple move types
    random.seed(42)
    
    def try_add(state, m, r):
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
    
    # Local search: try all possible single improvements
    def local_search_improve(state, max_iter=10000):
        state = list(state)
        score = compute_score(state)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 20:
                break
            
            # Try adding any missing modulus
            used_mods = set(m for m, r in state)
            for m in candidates:
                if m in used_mods:
                    continue
                valid_residues = find_all_valid_residues(m, state)
                if valid_residues:
                    # Pick best residue
                    new_state = state + [(m, valid_residues[0])]
                    new_score = compute_score(new_state)
                    if new_score > score:
                        state = new_state
                        score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try changing residue of existing modulus
            for idx, (m, r) in enumerate(state):
                valid_residues = find_all_valid_residues(m, [x for i, x in enumerate(state) if i != idx])
                for r_new in valid_residues:
                    if r_new == r:
                        continue
                    new_state = state[:idx] + [(m, r_new)] + state[idx+1:]
                    # Score doesn't change with residue, so this only helps if it enables additions
                    # For now, skip pure residue changes unless they enable additions
                    break
            
            # Try removing a modulus and adding a different one
            for idx in range(len(state)):
                if time.time() > deadline - 20:
                    break
                m_remove, r_remove = state[idx]
                state_without = try_remove(state, idx)
                score_without = compute_score(state_without)
                used_mods = set(m for m, r in state_without)
                # Try adding a different modulus
                for m_new in candidates:
                    if m_new in used_mods:
                        continue
                    valid_residues = find_all_valid_residues(m_new, state_without)
                    if valid_residues:
                        new_state = state_without + [(m_new, valid_residues[0])]
                        new_score = compute_score(new_state)
                        if new_score > score:
                            state = new_state
                            score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        return state, score
    
    # Run local search from best
    best, best_score = local_search_improve(best)
    
    # Phase 3: Simulated annealing for further improvement
    T = 0.5
    T_min = 0.001
    cooling = 0.9995
    
    state = list(best)
    score = best_score
    
    while T > T_min and time.time() < deadline - 10:
        # Random move
        move_type = random.choice(['add', 'remove', 'change_r', 'swap'])
        
        new_state = None
        new_score = None
        
        if move_type == 'add':
            used_mods = set(m for m, r in state)
            available = [m for m in candidates if m not in used_mods]
            if available:
                m_new = random.choice(available)
                valid_residues = find_all_valid_residues(m_new, state)
                if valid_residues:
                    r_new = random.choice(valid_residues)
                    new_state = state + [(m_new, r_new)]
                    new_score = compute_score(new_state)
        elif move_type == 'remove':
            if len(state) > 1:
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
        elif move_type == 'change_r':
            if state:
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                others = [x for i, x in enumerate(state) if i != idx]
                valid_residues = find_all_valid_residues(m_old, others)
                if len(valid_residues) > 1:
                    r_new = random.choice(valid_residues)
                    if r_new != r_old:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
        elif move_type == 'swap':
            if len(state) >= 1:
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                state_without = try_remove(state, idx)
                used_mods = set(m for m, r in state_without)
                available = [m for m in candidates if m not in used_mods]
                if available:
                    m_new = random.choice(available)
                    valid_residues = find_all_valid_residues(m_new, state_without)
                    if valid_residues:
                        r_new = random.choice(valid_residues)
                        new_state = state_without + [(m_new, r_new)]
                        new_score = compute_score(new_state)
        
        if new_state is not None and new_score is not None:
            delta = new_score - score
            if delta > 0 or random.random() < math.exp(delta / T):
                state = new_state
                score = new_score
                if score > best_score:
                    best, best_score = list(state), score
        
        T *= cooling
    
    # Phase 4: Final greedy fill
    used_mods = set(m for m, r in best)
    remaining = sorted([m for m in candidates if m not in used_mods])
    
    for m in remaining:
        if time.time() > deadline - 5:
            break
        r = find_valid_residue(m, best)
        if r is not None:
            new_state = best + [(m, r)]
            new_score = best_score + 1.0 / m
            if new_score > best_score:
                best, best_score = new_state, new_score
    
    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END

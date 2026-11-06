"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    n = len(candidates)
    # Map modulus value to index for fast lookup
    mod_to_idx = {m: i for i, m in enumerate(candidates)}
    
    # Precompute 1/m as Fraction
    inv = {m: Fraction(1, m) for m in candidates}
    
    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i, m1 in enumerate(candidates):
        for j, m2 in enumerate(candidates):
            gcd_table[i][j] = gcd(m1, m2)

    # Conflict check: are (m1, r1) and (m2, r2) disjoint?
    # Disjoint iff r1 % gcd(m1, m2) != r2 % gcd(m1, m2)
    def is_disjoint_pair(m1, r1, m2, r2):
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def is_compatible_with_all(m_new, r_new, selected):
        for (m_old, r_old) in selected:
            if not is_disjoint_pair(m_new, r_new, m_old, r_old):
                return False
        return True

    # --- Phase 1: Greedy Initialization with Residue Optimization ---
    # We want to maximize sum(1/m). Smaller m is better.
    # However, adding a small m blocks many others due to large gcds.
    # Strategy: Iterate through moduli in ascending order (smallest m first).
    # For each m, find a residue r that is compatible with all previously selected.
    # Among valid residues, choose one that minimizes the "future blocking potential".
    # Future blocking potential is approximated by sum of 1/gcd(m, m_future) for m_future > m
    # weighted by the fraction of residues blocked.
    
    def compute_blocking_cost(m_new, r_new, selected_mods_set, all_candidates):
        cost = 0.0
        for m2 in all_candidates:
            if m2 in selected_mods_set or m2 == m_new:
                continue
            g = gcd(m_new, m2)
            # If we pick r_new, we block the class r_new % g for m2.
            # The number of residues in m2 that are blocked is roughly m2/g.
            # The "cost" is the loss of potential 1/m2 if m2 could have been picked.
            # A simple heuristic: cost += 1.0 / (g * m2) ? Or just 1/g?
            # Let's use 1/g as a proxy for how "constrained" m2 becomes.
            # Actually, if g is large, m2 has fewer "free" residue classes.
            cost += 1.0 / g
        return cost

    def greedy_initialization(order):
        selected = []
        selected_mods_set = set()
        for m in order:
            if time.time() > deadline - 10:
                break
            valid_residues = []
            for r in range(m):
                if is_compatible_with_all(m, r, selected):
                    valid_residues.append(r)
            
            if not valid_residues:
                continue
            
            # Choose residue with min blocking cost
            best_r = valid_residues[0]
            best_cost = float('inf')
            # Optimization: only check a subset if m is large
            # But m <= 200, so checking all is fast.
            for r in valid_residues:
                # Cost calculation is expensive if done for every r.
                # We can approximate.
                # Let's just pick the first one for speed in initial pass,
                # or the one that is "central" to reduce future conflicts?
                # Actually, let's stick to a simpler heuristic:
                # Residues with smaller r % g for common gcds?
                # Let's just pick random valid residue to diversify,
                # or the one that minimizes conflicts with *next* few moduli.
                pass
            
            # To keep it fast, let's just pick the first valid residue.
            # We will refine in local search.
            best_r = valid_residues[0]
            
            selected.append((m, best_r))
            selected_mods_set.add(m)
        
        return selected

    # Generate multiple initial states
    best_state = None
    best_score = Fraction(0)
    
    # Strategy 1: Ascending order
    state1 = greedy_initialization(candidates)
    score1 = sum(inv[m] for m, r in state1)
    if best_state is None or score1 > best_score:
        best_state = state1
        best_score = score1

    # Strategy 2: Descending order (might allow more small moduli later? No, small moduli are processed first in asc)
    # Descending order processes large moduli first, leaving room for small ones?
    # If we process large moduli first, they have low weight. We want small moduli.
    # So ascending is generally better for greedy.
    # Let's try a randomized shuffle.
    rng = random.Random(42)
    for i in range(10):
        if time.time() > deadline - 20:
            break
        order = list(candidates)
        rng.shuffle(order)
        state = greedy_initialization(order)
        score = sum(inv[m] for m, r in state)
        if score > best_score:
            best_state = state
            best_score = score

    # --- Phase 2: Local Search with Add/Remove/Swap ---
    # We will iteratively try to improve the score.
    # Moves:
    # 1. Add a missing modulus (if compatible) -> always improves score (since 1/m > 0)
    # 2. Remove a modulus -> decreases score, but might allow adding two smaller moduli
    # 3. Change residue of an existing modulus -> score unchanged, but might allow adding a new one
    
    def get_valid_residues(m, selected):
        valid = []
        for r in range(m):
            if is_compatible_with_all(m, r, selected):
                valid.append(r)
        return valid

    def local_search(state, time_limit):
        state = list(state)
        score = sum(inv[m] for m, r in state)
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            used_mods = set(m for m, r in state)
            
            # 1. Try adding any missing modulus
            # Sort missing moduli by 1/m descending (i.e., m ascending)
            missing = [m for m in candidates if m not in used_mods]
            missing.sort()
            
            added_any = False
            for m in missing:
                if time.time() > end_time:
                    break
                valid_r = get_valid_residues(m, state)
                if valid_r:
                    # Add it
                    state.append((m, valid_r[0]))
                    score += inv[m]
                    used_mods.add(m)
                    improved = True
                    added_any = True
                    # Restart the loop to re-evaluate other additions
                    break 
            
            if added_any:
                continue
            
            # 2. Try swapping: remove one, add one or two
            # We look for a modulus m_rem in state such that removing it allows adding m_add
            # where 1/m_add > 1/m_rem (i.e., m_add < m_rem)
            # Or adding two moduli where sum(1/m_add1 + 1/m_add2) > 1/m_rem
            
            # To keep it fast, we only try removing the largest moduli (lowest weight)
            # and try to add the smallest missing moduli.
            
            # Sort state by m descending (remove large ones first)
            indices = list(range(len(state)))
            indices.sort(key=lambda i: state[i][0], reverse=True)
            
            swap_improved = False
            for idx in indices:
                if time.time() > end_time:
                    break
                
                m_rem, r_rem = state[idx]
                # Temporarily remove
                state_without = state[:idx] + state[idx+1:]
                score_without = score - inv[m_rem]
                used_mods_without = used_mods - {m_rem}
                
                # Find best addition
                missing_now = [m for m in candidates if m not in used_mods_without]
                missing_now.sort()
                
                best_gain = Fraction(0)
                best_add = None
                best_add_r = None
                best_add2 = None
                best_add2_r = None
                
                # Try adding single
                for m_add in missing_now[:10]: # Only check top 10 smallest
                    valid_r = get_valid_residues(m_add, state_without)
                    if valid_r:
                        gain = inv[m_add]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = m_add
                            best_add_r = valid_r[0]
                            best_add2 = None # Reset pair
                            best_add2_r = None
                        break # Since sorted, first valid is best single
                    
                # If single didn't improve, try pairs?
                # This is expensive. Let's stick to single swap for now, 
                # but allow adding a smaller one if it fits.
                
                if best_add is not None:
                    # Check if gain > 0 (always true) and if it's better than removing?
                    # We are replacing m_rem with m_add.
                    # Net change = inv[m_add] - inv[m_rem]
                    if best_gain > 0:
                        # We need to ensure the new state is valid (it is, by construction)
                        # Apply swap
                        state_without.append((best_add, best_add_r))
                        state = state_without
                        score = score_without + best_gain
                        used_mods = used_mods_without | {best_add}
                        improved = True
                        swap_improved = True
                        break # Restart loop after a swap
                
            if not swap_improved:
                # 3. Try changing residue to allow future additions?
                # This is complex. Let's rely on the add/swap loop.
                # If no improvement, we are stuck.
                # Try a random restart of local search?
                break
        
        return state, score

    # Run local search with a time budget
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        best_state, best_score = local_search(best_state, remaining_time)

    # --- Phase 3: Final Polish - Greedy Fill ---
    # Ensure we haven't left any easy wins on the table
    used_mods = set(m for m, r in best_state)
    missing = [m for m in candidates if m not in used_mods]
    missing.sort()
    
    for m in missing:
        if time.time() > deadline - 2:
            break
        valid_r = get_valid_residues(m, best_state)
        if valid_r:
            best_state.append((m, valid_r[0]))
            best_score += inv[m]
            used_mods.add(m)

    # Convert to required format
    result = [[m, r] for m, r in best_state]
    return result
# EVOLVE-BLOCK-END

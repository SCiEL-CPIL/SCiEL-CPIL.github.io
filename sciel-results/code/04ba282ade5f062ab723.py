"""Optimized program: greedy initialization + local search (add/swap)."""
import math
import time

# EVOLVE-BLOCK-START

def _is_compatible(m1, r1, m2, r2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _find_valid_residue(m, others):
    """
    Find a residue r for modulus m such that (m, r) is compatible with all (m2, r2) in others.
    Returns r or None.
    """
    for r in range(m):
        valid = True
        for m2, r2 in others:
            if not _is_compatible(m, r, m2, r2):
                valid = False
                break
        if valid:
            return r
    return None

def _score(chosen):
    return sum(1.0 / m for m, _ in chosen)

def _local_search(chosen, deadline):
    """
    Perform local search with Add and Swap moves.
    chosen: list of [m, r]
    """
    chosen_set = {(m, r) for m, r in chosen}
    moduli_in_set = {m for m, _ in chosen}
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Move 1: Add
        # Try to add any modulus not in the set, prioritizing smaller moduli (higher value)
        for m in range(20, 201):
            if time.time() > deadline:
                break
            if m in moduli_in_set:
                continue
            r = _find_valid_residue(m, chosen)
            if r is not None:
                chosen.append([m, r])
                moduli_in_set.add(m)
                improved = True
                # Restart add loop to catch chain reactions? 
                # Or just continue? Restarting is safer for finding more adds.
                # But simple continue might miss if order matters. 
                # Let's break and restart the while loop for efficiency of state update.
                break
        
        if improved:
            continue

        # Move 2: Swap
        # Try to replace a larger modulus with a smaller one to increase score.
        # Sort chosen by modulus descending to remove the least valuable ones first.
        # We iterate over current members.
        # To avoid modifying list while iterating, we make a copy or list of indices.
        # We want to swap (m_out, r_out) with (m_in, r_in) where m_in < m_out.
        
        # Iterate through moduli in the set, largest first (least valuable).
        sorted_members = sorted(chosen, key=lambda x: x[0], reverse=True)
        
        for m_out, r_out in sorted_members:
            if time.time() > deadline:
                break
            
            # Try all potential m_in smaller than m_out not in set
            for m_in in range(20, m_out):
                if m_in in moduli_in_set:
                    continue
                
                # Check if m_in can be added if m_out is removed
                others = [p for p in chosen if p[0] != m_out]
                r_in = _find_valid_residue(m_in, others)
                if r_in is not None:
                    # Perform swap
                    # Remove (m_out, r_out)
                    chosen = [p for p in chosen if p[0] != m_out]
                    moduli_in_set.discard(m_out)
                    
                    # Add (m_in, r_in)
                    chosen.append([m_in, r_in])
                    moduli_in_set.add(m_in)
                    
                    improved = True
                    break # Restart while loop to re-evaluate
            
            if improved:
                break

    return chosen

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under the 150 s wall-clock limit
    
    # 1. Initial Greedy Construction
    # Use the existing greedy logic but maybe slightly optimized or randomized?
    # The existing greedy is deterministic and fast.
    moduli = list(range(20, 201))
    chosen = []
    for m in moduli:
        if time.time() > deadline:
            break
        best_r = None
        best_conf = m + 1
        for r in range(m):
            conf = 0
            for (m2, r2) in chosen:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    conf += 1
                    if conf >= best_conf:
                        break
            if conf < best_conf:
                best_conf = conf
                best_r = r
                if best_conf == 0:
                    break
        if best_conf == 0:
            chosen.append([m, best_r])
    
    # 2. Local Search
    if time.time() < deadline:
        chosen = _local_search(chosen, deadline)
        
    return chosen

# EVOLVE-BLOCK-END

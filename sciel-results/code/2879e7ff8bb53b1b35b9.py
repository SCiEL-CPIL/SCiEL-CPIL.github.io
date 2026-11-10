import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    all_moduli = list(range(20, 201))
    
    def get_gcd(m1, m2):
        return math.gcd(m1, m2)

    def find_valid_residue(m_new, selected):
        """
        Find a residue r for m_new such that for all (m_sel, r_sel) in selected:
        r_new != r_sel (mod gcd(m_new, m_sel))
        """
        # Optimization: Pre-compute constraints if selected is large?
        # For now, simple loop.
        # We can speed this up by checking constraints one by one.
        # If no residue works, return -1.
        
        # To make it faster, we can iterate r from 0 to m_new-1.
        # Average m_new is ~110. Max 200.
        # Selected size might be up to ~50-100.
        # 200 * 100 = 20,000 ops per check. This is fast enough in Python if not called millions of times.
        
        for r in range(m_new):
            valid = True
            for m_sel, r_sel in selected:
                g = get_gcd(m_new, m_sel)
                if r % g == r_sel % g:
                    valid = False
                    break
            if valid:
                return r
        return -1

    # Phase 1: Greedy Construction
    # Sort moduli by 1/m descending (i.e., m ascending) to maximize score immediately.
    # This creates a dense set of small moduli.
    current_selected = []
    used_moduli = set()
    
    # Try to add moduli in increasing order of m
    for m in all_moduli:
        if time.time() > deadline:
            break
        r = find_valid_residue(m, current_selected)
        if r != -1:
            current_selected.append((m, r))
            used_moduli.add(m)

    # Phase 2: Local Search (Add and Swap)
    # We try to improve the score by:
    # 1. Adding any unused modulus if a valid residue exists.
    # 2. Swapping an existing modulus m_old with an unused modulus m_new < m_old.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add new moduli
        # Iterate through unused moduli, smallest first (highest potential gain)
        unused_moduli = [m for m in all_moduli if m not in used_moduli]
        unused_moduli.sort()
        
        for m_new in unused_moduli:
            if time.time() > deadline:
                break
            r_new = find_valid_residue(m_new, current_selected)
            if r_new != -1:
                current_selected.append((m_new, r_new))
                used_moduli.add(m_new)
                improved = True
                # Restart add loop to try more additions
                break
        
        # 2. If no additions, try swaps
        if not improved:
            # We want to replace larger m_old with smaller m_new.
            # Iterate existing moduli in descending order of m (worst first)
            # Iterate unused moduli in ascending order of m (best first)
            
            # To avoid O(N^2) full scan every time, we can just check a few promising swaps.
            # But N is small (< 180), so O(N^2) is ~30k checks. Each check is O(N) for feasibility.
            # 30k * 100 = 3M ops. Might be tight in Python if looped many times.
            # Let's limit the swap attempts per iteration.
            
            # Create list of (m, r) for existing
            existing_list = sorted(current_selected, key=lambda x: -x[0]) # Largest m first
            
            # Create list of unused
            unused_list = [m for m in all_moduli if m not in used_moduli]
            unused_list.sort() # Smallest m first
            
            swap_found = False
            for m_old, r_old in existing_list:
                if swap_found:
                    break
                if time.time() > deadline:
                    break
                
                for m_new in unused_list:
                    if m_new >= m_old:
                        break # Only swap if m_new is smaller (better score)
                    
                    # Check if m_new can fit into the set if we remove m_old
                    # Temporarily remove m_old
                    temp_selected = [(m, r) for m, r in current_selected if m != m_old]
                    
                    r_new = find_valid_residue(m_new, temp_selected)
                    if r_new != -1:
                        # Perform swap
                        # Rebuild current_selected
                        current_selected = temp_selected
                        current_selected.append((m_new, r_new))
                        used_moduli.discard(m_old)
                        used_moduli.add(m_new)
                        improved = True
                        swap_found = True
                        break
            
            if not swap_found:
                # If no simple swap works, we might be stuck in a local optimum.
                # A more advanced move (2-swap) is too expensive.
                # We can try random perturbations if time allows, but for now, break.
                break

    return [[m, r] for m, r in current_selected]

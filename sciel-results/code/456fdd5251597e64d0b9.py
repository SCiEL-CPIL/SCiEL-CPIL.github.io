# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    random.seed(42)
    
    # 1. Precompute gcds and incompatibility graph
    # Nodes are 20..200
    nodes = list(range(20, 201))
    n = len(nodes)
    # index mapping
    idx = {m: i for i, m in enumerate(nodes)}
    
    # incompatibility[m1][m2] = True if gcd(m1, m2) == 1
    # We want to select a set where no two are incompatible.
    # This is a Maximum Weight Independent Set problem.
    # Weight is 1/m.
    
    # For speed, we can just check gcd on the fly or precompute a bitmask?
    # 181 nodes. 181x181 matrix is fine.
    coprime = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(nodes[i], nodes[j]) == 1:
                coprime[i][j] = True
                coprime[j][i] = True

    # Greedy initialization: Sort by weight (1/m) descending.
    # Try to add if compatible with all currently selected.
    order = sorted(range(n), key=lambda i: -1.0/nodes[i])
    
    def get_score(selected_indices):
        return sum(1.0 / nodes[i] for i in selected_indices)
    
    def is_compatible(i, selected_indices):
        for j in selected_indices:
            if coprime[i][j]:
                return False
        return True

    # Initial Greedy Set
    selected = []
    for i in order:
        if is_compatible(i, selected):
            selected.append(i)
            
    best_selected = list(selected)
    best_score = get_score(best_selected)

    # Local Search on the Subset
    # Moves:
    # 1. Add a node if compatible.
    # 2. Swap a node: remove one, add another if compatible with rest.
    # 3. Random restarts? Maybe not needed if local search is good.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        # Try to add
        added = False
        # Pick random candidates not in selected
        candidates = [i for i in range(n) if i not in set(best_selected)]
        if candidates:
            # Try a few random additions
            for _ in range(10):
                i = random.choice(candidates)
                if is_compatible(i, best_selected):
                    new_sel = best_selected + [i]
                    new_score = get_score(new_sel)
                    if new_score > best_score:
                        best_selected = new_sel
                        best_score = new_score
                        added = True
                        break
        
        if added:
            continue
            
        # Try to swap
        # Pick a node to remove and a node to add
        if best_selected:
            i_out = random.choice(best_selected)
            rest = [x for x in best_selected if x != i_out]
            
            # Find best i_in that is compatible with rest and has higher weight
            # Or just random
            candidates_in = [i for i in range(n) if i not in set(rest)]
            if candidates_in:
                # Sort candidates_in by weight? Or just random?
                # Random is faster.
                for _ in range(20):
                    i_in = random.choice(candidates_in)
                    if is_compatible(i_in, rest):
                        new_sel = rest + [i_in]
                        new_score = get_score(new_sel)
                        if new_score > best_score:
                            best_selected = new_sel
                            best_score = new_score
                            break

    # 2. Assign Residues
    # Now we have best_selected (indices).
    # Convert to moduli.
    moduli = [nodes[i] for i in best_selected]
    
    # Initialize residues randomly
    residues = {}
    for m in moduli:
        residues[m] = random.randint(0, m-1)
        
    def check_residues_valid(res_map):
        mods = list(res_map.keys())
        for i in range(len(mods)):
            m1 = mods[i]
            r1 = res_map[m1]
            for j in range(i+1, len(mods)):
                m2 = mods[j]
                r2 = res_map[m2]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    # Local Search on Residues
    # If invalid, find a conflict and change one of the residues.
    valid = check_residues_valid(residues)
    max_res_iter = 10000
    for _ in range(max_res_iter):
        if time.time() > deadline:
            break
        if valid:
            break
            
        # Find a conflict
        conflict = None
        mods = list(residues.keys())
        for i in range(len(mods)):
            m1 = mods[i]
            r1 = residues[m1]
            for j in range(i+1, len(mods)):
                m2 = mods[j]
                r2 = residues[m2]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    conflict = (m1, m2)
                    break
            if conflict:
                break
                
        if not conflict:
            valid = True
            break
            
        m1, m2 = conflict
        # Try to change residue of m1 or m2
        # Pick one randomly
        if random.random() < 0.5:
            m_to_change = m1
        else:
            m_to_change = m2
            
        # Try all residues for m_to_change
        found_valid = False
        for new_r in range(m_to_change):
            residues[m_to_change] = new_r
            if check_residues_valid(residues):
                found_valid = True
                valid = True
                break
        if not found_valid:
            # If no single change works, try random
            residues[m_to_change] = random.randint(0, m_to_change - 1)

    # Final Validation and Return
    result = [[m, residues[m]] for m in moduli]
    
    # Double check validity
    if not check_residues_valid({m: r for m, r in result}):
        # Fallback to a simple valid set: all evens, residues 0?
        # Wait, if residues are all 0, and gcd>1, 0==0 is true -> INVALID.
        # We need distinct residues mod gcd.
        # For evens, gcd is at least 2.
        # If we set r_m = m % 2? No, all even m have r=0.
        # We need to ensure r1 != r2 mod gcd.
        
        # If the local search failed, we return the best valid one found or an empty set?
        # The prompt says "invalid result ... scores 0".
        # So we must ensure validity.
        
        # Let's try a simpler deterministic assignment for the fallback:
        # Sort moduli.
        # For each m, pick smallest r that doesn't conflict with previous.
        fallback_res = {}
        valid_fallback = True
        sorted_mods = sorted(moduli)
        for m in sorted_mods:
            found = False
            for r in range(m):
                ok = True
                for m2, r2 in fallback_res.items():
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    fallback_res[m] = r
                    found = True
                    break
            if not found:
                valid_fallback = False
                break
        
        if valid_fallback:
            result = [[m, fallback_res[m]] for m in moduli]
        else:
            # Extremely unlikely to fail, but if it does, return a single valid class
            # Smallest m in moduli
            if moduli:
                m = min(moduli)
                result = [[m, 0]]
            else:
                result = []

    return result
# EVOLVE-BLOCK-END

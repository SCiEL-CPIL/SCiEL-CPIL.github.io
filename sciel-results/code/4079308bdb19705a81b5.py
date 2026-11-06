import time
import math
import random
from math import gcd

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def _is_compatible(m, r, selected):
    """Check if (m, r) is compatible with all entries in selected."""
    for (m2, r2) in selected:
        g = gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _find_valid_residues(m, selected):
    """Find all valid residues for modulus m given selected set."""
    valid = []
    for r in range(m):
        if _is_compatible(m, r, selected):
            valid.append(r)
    return valid

def _greedy_construct(deadline, random_seed=None):
    """Greedy construction with optional randomization."""
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    moduli = list(range(20, 201))
    
    for m in moduli:
        if time.time() > deadline:
            break
        valid_residues = _find_valid_residues(m, selected)
        if valid_residues:
            r = random.choice(valid_residues) if random_seed is not None else valid_residues[0]
            selected.append((m, r))
    
    return selected

def _local_improve(chosen, deadline):
    """Iteratively add missing moduli and swap residues to free capacity."""
    chosen = [list(c) for c in chosen]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        current_moduli = set(m for m, r in chosen)
        
        # Try adding any missing modulus
        for m_new in range(20, 201):
            if m_new in current_moduli:
                continue
            if time.time() > deadline:
                break
            
            valids = _find_valid_residues(m_new, chosen)
            if valids:
                r_new = valids[0]
                chosen.append([m_new, r_new])
                current_moduli.add(m_new)
                improved = True
                continue
            
            # Try swapping residues to make room for m_new
            found_change = False
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                m_i, r_i = chosen[i]
                others = [c for j, c in enumerate(chosen) if j != i]
                valid_for_i = _find_valid_residues(m_i, others)
                
                if not valid_for_i:
                    continue
                
                for r_i_new in valid_for_i:
                    if r_i_new == r_i:
                        continue
                    temp_selected = [list(c) for c in chosen]
                    temp_selected[i][1] = r_i_new
                    
                    if _find_valid_residues(m_new, temp_selected):
                        chosen[i][1] = r_i_new
                        valids_new = _find_valid_residues(m_new, chosen)
                        chosen.append([m_new, valids_new[0]])
                        current_moduli.add(m_new)
                        improved = True
                        found_change = True
                        break
                if found_change:
                    break
    
    return chosen

def _residue_reoptimize(chosen, deadline):
    """Try different residue assignments to free capacity for more moduli."""
    chosen = [list(c) for c in chosen]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        current_moduli = set(m for m, r in chosen)
        
        # For each modulus, try all alternative residues
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            m_i, r_i = chosen[i]
            others = [c for j, c in enumerate(chosen) if j != i]
            valid_for_i = _find_valid_residues(m_i, others)
            
            if not valid_for_i:
                continue
            
            for r_new in valid_for_i:
                if r_new == r_i:
                    continue
                
                # Try this assignment
                chosen[i][1] = r_new
                
                # Try to add any missing modulus
                added = False
                for m_new in range(20, 201):
                    if m_new in current_moduli:
                        continue
                    if time.time() > deadline:
                        break
                    valids = _find_valid_residues(m_new, chosen)
                    if valids:
                        chosen.append([m_new, valids[0]])
                        current_moduli.add(m_new)
                        improved = True
                        added = True
                        break
                
                if not added:
                    # Revert
                    chosen[i][1] = r_i
    
    return chosen

def _simulate_annealing(classes, deadline, initial_temp=0.5, cooling=0.999):
    """Simulated annealing with add/remove/swap moves."""
    current = {m: r for m, r in classes}
    current_score = _score(classes)
    best = {m: r for m, r in classes}
    best_score = current_score
    
    temp = initial_temp
    all_moduli = list(range(20, 201))
    
    while time.time() < deadline and temp > 0.001:
        move_type = random.random()
        
        if move_type < 0.4 and len(current) >= 2:
            # Swap residues between two moduli
            items = list(current.items())
            (m1, r1), (m2, r2) = random.sample(items, 2)
            
            new_current = dict(current)
            new_current[m1] = r2
            new_current[m2] = r1
            
            # Check validity
            valid = True
            for m, r in new_current.items():
                for m2_, r2_ in new_current.items():
                    if m >= m2_:
                        continue
                    g = gcd(m, m2_)
                    if r % g == r2_ % g:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_score = _score([[m, r] for m, r in new_current.items()])
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current = new_current
                    current_score = new_score
                    if current_score > best_score:
                        best = dict(current)
                        best_score = current_score
        
        elif move_type < 0.7:
            # Add a new modulus
            available = [m for m in all_moduli if m not in current]
            if available:
                m = random.choice(available)
                valids = _find_valid_residues(m, list(current.items()))
                if valids:
                    r = random.choice(valids)
                    new_current = dict(current)
                    new_current[m] = r
                    new_score = _score([[m, r] for m, r in new_current.items()])
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current = new_current
                        current_score = new_score
                        if current_score > best_score:
                            best = dict(current)
                            best_score = current_score
        
        else:
            # Remove a modulus
            if current:
                m_to_remove = random.choice(list(current.keys()))
                new_current = dict(current)
                del new_current[m_to_remove]
                new_score = _score([[m, r] for m, r in new_current.items()])
                delta = new_score - current_score
                if random.random() < math.exp(delta / temp) * 0.5:
                    current = new_current
                    current_score = new_score
        
        temp *= cooling
    
    return [[m, r] for m, r in best.items()], best_score

def _final_polish(classes, deadline):
    """Final pass: systematically try to add all possible moduli and optimize residues."""
    chosen = [list(c) for c in classes]
    
    # Multiple passes of local improvement
    for _ in range(3):
        if time.time() > deadline:
            break
        chosen = _local_improve(chosen, deadline)
        chosen = _residue_reoptimize(chosen, deadline)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin under 150s limit
    
    # Phase 1: Greedy construction (deterministic)
    best_classes = _greedy_construct(deadline)
    best_score = _score(best_classes)
    
    # Phase 2: Local improvement
    if time.time() < deadline - 30:
        improved = _local_improve(best_classes, deadline - 30)
        if _score(improved) > best_score:
            best_classes = improved
            best_score = _score(best_classes)
    
    # Phase 3: Residue reoptimization
    if time.time() < deadline - 20:
        reopt = _residue_reoptimize(best_classes, deadline - 20)
        if _score(reopt) > best_score:
            best_classes = reopt
            best_score = _score(best_classes)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 15:
        sa_classes, sa_score = _simulate_annealing(best_classes, deadline - 15)
        if sa_score > best_score:
            best_classes = sa_classes
            best_score = sa_score
    
    # Phase 5: Final polish
    if time.time() < deadline - 2:
        polished = _final_polish(best_classes, deadline - 2)
        if _score(polished) > best_score:
            best_classes = polished
            best_score = _score(polished)
    
    return best_classes

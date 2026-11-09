import time
import math
from fractions import Fraction

# Global variables for B&B
_best_score = 0.0
_best_classes = []
_start_time = 0
_deadline = 0
_weights = []
_moduli = []
_residues = []
_conflict_masks = []
_mod_masks = {}
_node_index = {}
N_NODES = 0

def _disjoint(m1, r1, m2, r2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _init_graph():
    global N_NODES, _weights, _moduli, _residues, _conflict_masks, _mod_masks, _node_index
    
    # Generate nodes
    nodes = []
    for m in range(20, 201):
        for r in range(m):
            nodes.append((m, r))
    
    N_NODES = len(nodes)
    _weights = [1.0 / m for m, r in nodes]
    _moduli = [m for m, r in nodes]
    _residues = [r for m, r in nodes]
    _conflict_masks = [0] * N_NODES
    
    # Map (m, r) to index
    _node_index = {}
    for i, (m, r) in enumerate(nodes):
        _node_index[(m, r)] = i
        
    # Mod masks
    _mod_masks = {}
    for m in range(20, 201):
        mask = 0
        for r in range(m):
            idx = _node_index[(m, r)]
            mask |= (1 << idx)
        _mod_masks[m] = mask
        
    # Build conflict masks
    # Group indices by modulus
    mod_indices = {}
    for i in range(N_NODES):
        m = _moduli[i]
        if m not in mod_indices:
            mod_indices[m] = []
        mod_indices[m].append(i)
        
    # Iterate pairs of moduli
    for m1 in range(20, 201):
        idx_m1 = mod_indices[m1]
        for m2 in range(m1, 201):
            idx_m2 = mod_indices[m2]
            g = math.gcd(m1, m2)
            
            if m1 == m2:
                # All conflict with each other
                # Create mask for all indices in idx_m1
                mask_m1 = 0
                for idx in idx_m1:
                    mask_m1 |= (1 << idx)
                for idx in idx_m1:
                    _conflict_masks[idx] |= mask_m1
            else:
                # Group indices of m2 by residue mod g
                groups = {}
                for idx in idx_m2:
                    r2 = _residues[idx]
                    key = r2 % g
                    if key not in groups:
                        groups[key] = 0
                    groups[key] |= (1 << idx)
                    
                for idx1 in idx_m1:
                    r1 = _residues[idx1]
                    key = r1 % g
                    if key in groups:
                        _conflict_masks[idx1] |= groups[key]
                        # Symmetry: need to update idx2 as well
                        # But we are iterating m1 <= m2, so we need to add to idx2's mask
                        # We can do this by iterating the bits in groups[key]?
                        # Or just store the mask and add it to all idx2 in that group.
                        # Since we already have the mask, we can just OR it to idx1.
                        # For idx2, we need to OR idx1's bit.
                        # It is faster to just OR the mask to idx1, and then
                        # iterate the bits of the mask to update idx2?
                        # No, the mask `groups[key]` contains many bits.
                        # We need to set bit idx1 in all masks for idx2 in groups[key].
                        
                        # Optimization: 
                        # _conflict_masks[idx1] |= groups[key]
                        # for idx2 in bits_of(groups[key]):
                        #     _conflict_masks[idx2] |= (1 << idx1)
                        
                        # This inner loop is expensive.
                        # Alternative: Store conflicts in a list of lists first, then convert to masks?
                        # Or, just do the double loop for m1 != m2 case?
                        # The double loop is $m_1 \times m_2$.
                        # Sum of $m_1 m_2$ is ~200M.
                        # The bit manipulation approach above is faster for the mask creation,
                        # but the symmetry update is the issue.
                        
                        # Let's stick to the double loop for simplicity and correctness, 
                        # but optimize with precomputed groups.
                        
                        # Actually, let's just use the double loop for m1 != m2 
                        # but use the group lookup to skip non-conflicts.
                        pass 

    # Re-implement conflict building with double loop but optimized
    _conflict_masks = [0] * N_NODES
    for m1 in range(20, 201):
        idx_m1 = mod_indices[m1]
        for m2 in range(m1, 201):
            idx_m2 = mod_indices[m2]
            g = math.gcd(m1, m2)
            
            if m1 == m2:
                mask_m1 = 0
                for idx in idx_m1:
                    mask_m1 |= (1 << idx)
                for idx in idx_m1:
                    _conflict_masks[idx] |= mask_m1
            else:
                # Group idx_m2 by r2 % g
                groups = {}
                for idx2 in idx_m2:
                    r2 = _residues[idx2]
                    key = r2 % g
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(idx2)
                    
                for idx1 in idx_m1:
                    r1 = _residues[idx1]
                    key = r1 % g
                    if key in groups:
                        # idx1 conflicts with all idx2 in groups[key]
                        mask_to_add = 0
                        for idx2 in groups[key]:
                            mask_to_add |= (1 << idx2)
                            _conflict_masks[idx2] |= (1 << idx1)
                        _conflict_masks[idx1] |= mask_to_add

def _greedy_solve():
    global _best_score, _best_classes
    avail = (1 << N_NODES) - 1
    score = 0.0
    selected = []
    
    # Sort indices by weight descending
    sorted_indices = sorted(range(N_NODES), key=lambda i: _weights[i], reverse=True)
    
    for i in sorted_indices:
        if avail & (1 << i):
            selected.append(i)
            score += _weights[i]
            avail &= ~_conflict_masks[i]
            avail &= ~(1 << i)
            
    if score > _best_score:
        _best_score = score
        _best_classes = [[_moduli[i], _residues[i]] for i in selected]
        
    return score, selected

def _bound(avail):
    bound = 0.0
    for m in range(20, 201):
        if avail & _mod_masks[m]:
            bound += 1.0 / m
    return bound

def _branch_and_bound(avail, score, selected):
    global _best_score, _best_classes, _deadline
    
    if time.time() > _deadline:
        return
        
    if avail == 0:
        if score > _best_score:
            _best_score = score
            _best_classes = [[_moduli[i], _residues[i]] for i in selected]
        return
        
    # Upper bound
    b = _bound(avail)
    if score + b <= _best_score:
        return
        
    # Find a node to branch on
    # Pick the one with highest weight in avail
    # To speed up, we can iterate sorted_indices again
    # But we need to pass sorted_indices or make it global
    global _sorted_indices
    u = -1
    for i in _sorted_indices:
        if avail & (1 << i):
            u = i
            break
            
    if u == -1:
        return
        
    # Branch 1: Include u
    new_avail = avail & ~_conflict_masks[u] & ~(1 << u)
    _branch_and_bound(new_avail, score + _weights[u], selected + [u])
    
    # Branch 2: Exclude u
    new_avail = avail & ~(1 << u)
    _branch_and_bound(new_avail, score, selected)

def construct():
    global _start_time, _deadline, _best_score, _best_classes, _sorted_indices
    _start_time = time.time()
    _deadline = _start_time + 115.0  # 5s margin
    
    _init_graph()
    _sorted_indices = sorted(range(N_NODES), key=lambda i: _weights[i], reverse=True)
    
    _best_score = 0.0
    _best_classes = []
    
    # Greedy
    _greedy_solve()
    
    # B&B
    initial_avail = (1 << N_NODES) - 1
    _branch_and_bound(initial_avail, 0.0, [])
    
    if not _best_classes:
        # Fallback
        _best_classes = [[20, 0]]
        
    return _best_classes

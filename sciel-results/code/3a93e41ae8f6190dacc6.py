"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Compute constraint degree: how many other moduli share a factor > 1
    constraint = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_table[i][j] > 1:
                constraint[i] += 1
                constraint[j] += 1

    def is_disjoint_idx(r1, i1, r2, i2):
        """Check if residue class r1 mod moduli[i1] is disjoint from r2 mod moduli[i2]."""
        g = gcd_table[i1][i2]
        if g == 1:
            return False  # coprime moduli always intersect
        return (r1 - r2) % g != 0

    def can_add_idx(r_new, m_idx_new, selected_indices, selected_residues):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected."""
        for idx, (r_sel, m_idx_sel) in enumerate(zip(selected_residues, selected_indices)):
            g = gcd_table[m_idx_sel][m_idx_new]
            if g == 1:
                return False
            if (r_new - r_sel) % g == 0:
                return False
        return True

    def greedy_construct(order, rng):
        """Greedy: process moduli in given order, pick a valid residue for each."""
        selected_indices = []
        selected_residues = []
        for m_idx in order:
            m = moduli[m_idx]
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                if can_add_idx(r, m_idx, selected_indices, selected_residues):
                    selected_indices.append(m_idx)
                    selected_residues.append(r)
                    break
        return selected_indices, selected_residues

    def try_add_one(selected_indices, selected_residues, rng):
        """Try to add any one missing modulus with a valid residue. Returns (new_indices, new_residues) or None."""
        selected_set = set(selected_indices)
        unused = [i for i in range(n) if i not in selected_set]
        rng.shuffle(unused)
        
        for m_idx in unused:
            m = moduli[m_idx]
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                if can_add_idx(r, m_idx, selected_indices, selected_residues):
                    return selected_indices + [m_idx], selected_residues + [r]
        return None

    def kick_and_repair(selected_indices, selected_residues, kick_size, rng):
        """Remove kick_size random elements, then greedily try to re-add them and others."""
        if len(selected_indices) < kick_size:
            kick_size = max(0, len(selected_indices) - 1)
        if kick_size == 0:
            return selected_indices, selected_residues
        
        # Pick random indices to remove
        remove_indices = rng.sample(range(len(selected_indices)), kick_size)
        remove_set = set(remove_indices)
        
        new_indices = [idx for i, idx in enumerate(selected_indices) if i not in remove_set]
        new_residues = [r for i, r in enumerate(selected_residues) if i not in remove_set]
        
        # Now try to add back removed ones and any other unused
        for _ in range(3):  # a few passes
            result = try_add_one(new_indices, new_residues, rng)
            if result is None:
                break
            new_indices, new_residues = result
        
        return new_indices, new_residues

    def local_search(selected_indices, selected_residues, max_iter, rng):
        """Iteratively try to add more moduli, with kicks when stuck."""
        best_indices = selected_indices[:]
        best_residues = selected_residues[:]
        best_len = len(best_indices)
        
        cur_indices = selected_indices[:]
        cur_residues = selected_residues[:]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add
            result = try_add_one(cur_indices, cur_residues, rng)
            if result is not None:
                cur_indices, cur_residues = result
                if len(cur_indices) > best_len:
                    best_indices = cur_indices[:]
                    best_residues = cur_residues[:]
                    best_len = len(cur_indices)
            else:
                # Stuck, do a kick
                kick_size = 1 if rng.random() < 0.6 else (2 if rng.random() < 0.8 else 3)
                cur_indices, cur_residues = kick_and_repair(cur_indices, cur_residues, kick_size, rng)
        
        return best_indices, best_residues

    # Build diverse orderings
    rng = random.Random(42)
    
    order_most_constrained = sorted(range(n), key=lambda i: (-constraint[i], -moduli[i]))
    order_least_constrained = sorted(range(n), key=lambda i: (constraint[i], moduli[i]))
    order_desc = list(range(n - 1, -1, -1))
    order_asc = list(range(n))
    
    all_orders = [order_most_constrained, order_least_constrained, order_desc, order_asc]
    
    # Add some random orders
    for _ in range(6):
        random_order = list(range(n))
        rng.shuffle(random_order)
        all_orders.append(random_order)
    
    best_indices = []
    best_residues = []
    best_len = 0
    
    for order in all_orders:
        if time.time() > deadline - 5.0:
            break
        
        s_indices, s_residues = greedy_construct(order, rng)
        
        # Local search with limited iterations per order
        s_indices, s_residues = local_search(s_indices, s_residues, max_iter=800, rng=rng)
        
        if len(s_indices) > best_len:
            best_indices = s_indices[:]
            best_residues = s_residues[:]
            best_len = len(s_indices)
    
    # Final intensive local search on the best solution
    if time.time() < deadline - 2.0:
        best_indices, best_residues = local_search(best_indices, best_residues, max_iter=3000, rng=rng)
    
    # Convert to required format
    result = [[best_residues[i], moduli[best_indices[i]]] for i in range(len(best_indices))]
    
    # Ensure nonempty
    if not result:
        return [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END

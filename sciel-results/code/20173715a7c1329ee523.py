import sys
import math

def construct():
    moduli = list(range(2, 129))
    max_mod = 128
    
    # Create index mapping
    pairs = []
    index_map = {}
    idx = 0
    for m in moduli:
        for r in range(m):
            pairs.append((m, r))
            index_map[(m, r)] = idx
            idx += 1
    N = len(pairs)
    
    # Precompute GCDs
    m_indices = {m: i for i, m in enumerate(moduli)}
    num_mods = len(moduli)
    gcd_mat = [[0]*num_mods for _ in range(num_mods)]
    for i in range(num_mods):
        for j in range(num_mods):
            gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
            
    # Precompute residue bitsets for each modulus and each divisor g
    # residue_bits[m][g][k] is a bitset of indices in pairs corresponding to (m, r)
    # where r % g == k.
    # Note: The bit positions are the global indices 0..N-1.
    
    # First, map local residue index to global index?
    # We can just build the mask by iterating r.
    
    # Get divisors for each modulus
    divisors = {m: [] for m in moduli}
    for m in moduli:
        for d in range(1, m + 1):
            if m % d == 0:
                divisors[m].append(d)
                
    residue_bits = {}
    all_bits_for_mod = {}
    for m in moduli:
        # all_bits_for_mod[m] has bits set for all (m, r)
        mask = 0
        for r in range(m):
            mask |= (1 << index_map[(m, r)])
        all_bits_for_mod[m] = mask
        
        # For each divisor g of m
        for g in divisors[m]:
            residue_bits[(m, g)] = {}
            for k in range(g):
                mask_k = 0
                for r in range(k, m, g):
                    mask_k |= (1 << index_map[(m, r)])
                residue_bits[(m, g)][k] = mask_k
                
    # Build compat bitsets
    compat = [0] * N
    for i in range(N):
        m_i, r_i = pairs[i]
        mi_idx = m_indices[m_i]
        mask = 0
        for m_j in moduli:
            if m_j == m_i:
                continue
            mj_idx = m_indices[m_j]
            g = gcd_mat[mi_idx][mj_idx]
            k = r_i % g
            # Bad residues are those where r_j % g == k
            bad_mask = residue_bits[(m_j, g)][k]
            # Good residues are all bits for m_j except bad_mask
            good_mask = all_bits_for_mod[m_j] ^ bad_mask
            mask |= good_mask
        compat[i] = mask
        
    # Search for Max Clique
    search_moduli = list(range(2, 129))
    mod_nodes = []
    for m in search_moduli:
        nodes = []
        for r in range(m):
            nodes.append(index_map[(m, r)])
        mod_nodes.append(nodes)
        
    best_count = 0
    best_solution = []
    
    def search(idx, current_mask, current_count, current_nodes):
        nonlocal best_count, best_solution
        
        if current_count + (len(search_moduli) - idx) <= best_count:
            return
            
        if idx == len(search_moduli):
            if current_count > best_count:
                best_count = current_count
                best_solution = current_nodes.copy()
            return
            
        candidates = mod_nodes[idx]
        valid_cands = []
        for v in candidates:
            if (current_mask >> v) & 1:
                valid_cands.append(v)
                
        # Skip
        search(idx + 1, current_mask, current_count, current_nodes)
        
        # Pick
        for v in valid_cands:
            new_mask = current_mask & compat[v]
            current_nodes.append(v)
            search(idx + 1, new_mask, current_count + 1, current_nodes)
            current_nodes.pop()
            
    initial_mask = (1 << N) - 1
    search(0, initial_mask, 0, [])
    
    result = []
    for v in best_solution:
        m, r = pairs[v]
        result.append([r, m])
        
    return result

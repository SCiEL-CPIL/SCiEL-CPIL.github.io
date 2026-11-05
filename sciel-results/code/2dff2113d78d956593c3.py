import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Precompute count of residues in [0, m) for each modulus m, for each modulus g, for each residue k mod g
    # count_res[i][g][k] = count of r in [0, moduli[i]) such that r % g == k
    # g ranges from 1 to max(moduli[i])
    max_m = max(moduli)
    count_res = [[[0] * g for k in range(g)] for g in range(max_m + 1)] # This is wrong structure
    
    # Correct structure:
    # For each modulus index i, for each possible gcd value g (1..moduli[i]),
    # we need an array of size g.
    # Let's just compute on the fly or store in a dict/list of lists.
    
    # Simpler: Precompute for each pair (i, j) the gcd g.
    # And for each j, and each g, the counts.
    
    # Let's precompute counts for each modulus j
    # counts[j] is a list of lists. counts[j][g] is a list of size g.
    counts = [None] * n_mods
    for j in range(n_mods):
        m_j = moduli[j]
        counts[j] = [None] * (m_j + 1)
        for g in range(1, m_j + 1):
            # Count residues 0..m_j-1 mod g
            arr = [0] * g
            for r in range(m_j):
                arr[r % g] += 1
            counts[j][g] = arr

    best_count = 0
    best_result = []
    
    def is_valid(m_idx, r, selected_mods, selected_res):
        for k, sm_idx in enumerate(selected_mods):
            g = gcds[m_idx][sm_idx]
            if g > 1:
                if r % g == selected_res[k] % g:
                    return False
        return True

    def calculate_damage(m_idx, r, selected_mods, selected_res, all_indices):
        # all_indices: set of indices not yet selected
        # We want to sum counts[unselected_idx][g][r % g] for all unselected
        # This is expensive if done naively.
        # Optimization: Maintain a running sum?
        # For each unselected u, damage contribution is counts[u][gcd(m,u)][r % gcd(m,u)]
        # We can precompute for each m_idx and r, the vector of damages to all u?
        # No, r changes.
        
        # Let's just sum.
        # Number of unselected is small in later stages.
        damage = 0
        # We need to know which are unselected.
        # selected_mods is a list. We can make a set for O(1) lookup?
        # Or just iterate all and check if in selected.
        # Since n=127, iterating all is fine.
        for u_idx in all_indices:
            if u_idx in selected_mods:
                continue
            g = gcds[m_idx][u_idx]
            if g > 1:
                damage += counts[u_idx][g][r % g]
        return damage

    def greedy_pass(seed):
        random.seed(seed)
        indices = list(range(n_mods))
        random.shuffle(indices)
        
        selected_mods = []
        selected_res = []
        selected_set = set()
        
        for m_idx in indices:
            m = moduli[m_idx]
            valid_residues = []
            for r in range(m):
                if is_valid(m_idx, r, selected_mods, selected_res):
                    valid_residues.append(r)
            
            if not valid_residues:
                continue
                
            if len(valid_residues) == 1:
                r = valid_residues[0]
            else:
                # Calculate damage for each valid residue
                # To speed up, we can sample? No, m <= 128, so max 128 residues.
                # Damage calc is O(N) per residue.
                # 128 * 127 = 16k ops. Fine.
                min_damage = float('inf')
                best_r = valid_residues[0]
                for r in valid_residues:
                    d = calculate_damage(m_idx, r, selected_mods, selected_res, indices)
                    if d < min_damage:
                        min_damage = d
                        best_r = r
                r = best_r
            
            selected_mods.append(m_idx)
            selected_res.append(r)
            selected_set.add(m_idx)
            
        result = []
        for i, m_idx in enumerate(selected_mods):
            result.append([selected_res[i], moduli[m_idx]])
        return result

    seed = 0
    while time.time() < deadline:
        res = greedy_pass(seed)
        if len(res) > best_count:
            best_count = len(res)
            best_result = res
        seed += 1
        
    return best_result if best_result else [[1, 2]]
# EVOLVE-BLOCK-END

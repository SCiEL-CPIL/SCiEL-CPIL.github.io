import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    moduli_range = list(range(2, 129))
    n_moduli = len(moduli_range)
    
    # Precompute gcd matrix
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_table[i][j] = gcd(moduli_range[i], moduli_range[j])

    # Candidate cliques: sets of moduli where every pair has gcd > 1
    # Largest cliques are likely multiples of small primes.
    
    candidates = []
    
    # Multiples of 2
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 2 == 0])
    # Multiples of 3
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 3 == 0])
    # Multiples of 4
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 4 == 0])
    # Multiples of 5
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 5 == 0])
    # Multiples of 6
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 6 == 0])
    # Multiples of 7
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 7 == 0])
    
    # Sort by size descending to prioritize larger cliques
    candidates.sort(key=lambda x: len(x), reverse=True)
    
    def solve_clique(clique_indices, time_limit=60.0):
        """
        Attempt to find valid residues for a given set of moduli (clique).
        Uses backtracking with MRV heuristic and a time limit.
        Returns list of [r, m] pairs or None.
        """
        if not clique_indices:
            return []
        
        # Calculate constraint strength for each variable in the clique
        # Strength = sum of gcd(m, n) for all other n in clique
        strengths = {}
        for i in clique_indices:
            s = 0
            for j in clique_indices:
                if i != j:
                    s += gcd_table[i][j]
            strengths[i] = s
            
        # Order variables by strength descending (MRV-like: most constrained first)
        ordered_vars = sorted(clique_indices, key=lambda x: strengths[x], reverse=True)
        
        residues = {}
        start_solve = time.time()
        
        def is_compatible(new_r, new_m_idx, assigned):
            """
            Check if new_r is compatible with all assigned residues.
            assigned: dict {idx: residue}
            """
            for idx, res in assigned.items():
                g = gcd_table[new_m_idx][idx]
                if new_r % g == res % g:
                    return False
            return True

        def backtrack(pos):
            if time.time() - start_solve > time_limit:
                return False
            
            if pos == len(ordered_vars):
                return True
                
            m_idx = ordered_vars[pos]
            m = moduli_range[m_idx]
            
            # Try residues
            for r in range(m):
                if is_compatible(r, m_idx, residues):
                    residues[m_idx] = r
                    if backtrack(pos + 1):
                        return True
                    del residues[m_idx]
                    
            return False

        if backtrack(0):
            result = []
            for idx in ordered_vars:
                result.append([residues[idx], moduli_range[idx]])
            return result
        return None

    best = []
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        # Give more time to larger cliques
        time_limit = 50.0 if len(cand) > 50 else 20.0
        res = solve_clique(cand, time_limit)
        if res and len(res) > len(best):
            best = res
            
    # Fallback: if no clique worked, try a general greedy on full set
    if not best:
        residues = {}
        selected = []
        selected_idx = []
        
        def is_compatible_general(new_r, new_m_idx):
            for idx in selected_idx:
                g = gcd_table[new_m_idx][idx]
                if new_r % g == residues[idx] % g:
                    return False
            return True

        for m_idx in range(n_moduli):
            m = moduli_range[m_idx]
            for r in range(m):
                if is_compatible_general(r, m_idx):
                    residues[m_idx] = r
                    selected.append([r, m])
                    selected_idx.append(m_idx)
                    break
        best = selected

    if not best:
        best = [[1, 2]]

    return best

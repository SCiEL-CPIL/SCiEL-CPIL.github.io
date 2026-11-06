import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    moduli_range = list(range(20, 201))
    
    def is_feasible(selected_moduli, node_limit=10000):
        if not selected_moduli:
            return True
        
        n = len(selected_moduli)
        residues = [None] * n
        
        gcd_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = gcd(selected_moduli[i], selected_moduli[j])
                gcd_matrix[i][j] = g
                gcd_matrix[j][i] = g
        
        nodes_visited = 0
        
        def backtrack(idx):
            nonlocal nodes_visited
            if time.time() > deadline:
                return False
            nodes_visited += 1
            if nodes_visited > node_limit:
                return False
            
            if idx == n:
                return True
            
            m = selected_moduli[idx]
            
            for r in range(m):
                valid = True
                for j in range(idx):
                    if residues[j] is None:
                        continue
                    g = gcd_matrix[idx][j]
                    if (r - residues[j]) % g == 0:
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            
            return False
        
        return backtrack(0)

    def calc_score(selected_moduli):
        return sum(1.0 / m for m in selected_moduli)

    selected = []
    for m in moduli_range:
        if time.time() > deadline:
            break
        if is_feasible(selected + [m]):
            selected.append(m)
    
    current_score = calc_score(selected)
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for m_in in moduli_range:
            if time.time() > deadline:
                break
            if m_in in selected_set:
                continue
            
            for m_out in sorted(selected, reverse=True):
                if 1.0 / m_in <= 1.0 / m_out:
                    continue
                
                new_selected = [m for m in selected if m != m_out] + [m_in]
                if is_feasible(new_selected):
                    selected = new_selected
                    selected_set = set(selected)
                    current_score = calc_score(selected)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            for m_in in moduli_range:
                if time.time() > deadline:
                    break
                if m_in in selected_set:
                    continue
                if is_feasible(selected + [m_in]):
                    selected.append(m_in)
                    selected_set = set(selected)
                    current_score = calc_score(selected)
                    improved = True
                    break
    
    final_moduli = selected
    residues = [None] * len(final_moduli)
    n = len(final_moduli)
    
    if n > 0:
        gcd_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = gcd(final_moduli[i], final_moduli[j])
                gcd_matrix[i][j] = g
                gcd_matrix[j][i] = g
        
        def backtrack_get_res(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            m = final_moduli[idx]
            for r in range(m):
                valid = True
                for j in range(idx):
                    if residues[j] is None:
                        continue
                    g = gcd_matrix[idx][j]
                    if (r - residues[j]) % g == 0:
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack_get_res(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        if not backtrack_get_res(0):
            pass
    
    result = []
    for m, r in zip(final_moduli, residues):
        if r is not None:
            result.append([m, r])
            
    return result

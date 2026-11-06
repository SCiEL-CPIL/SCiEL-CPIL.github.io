import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    moduli = list(range(20, 201))
    moduli.sort(key=lambda x: 1.0/x, reverse=True)
    
    n = len(moduli)
    weights = [1.0/m for m in moduli]
    
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
    best_score = 0.0
    best_set = []
    
    # Greedy initialization
    greedy_set = []
    for i in range(n):
        if time.time() > deadline:
            break
        candidate_set = greedy_set + [moduli[i]]
        if is_feasible(candidate_set):
            greedy_set.append(moduli[i])
    if greedy_set:
        best_score = sum(1.0/m for m in greedy_set)
        best_set = list(greedy_set)
        
    # Memoization for feasibility
    # Key: tuple of moduli in the set
    feasibility_cache = {}
    
    def check_feasibility_cached(m_set):
        key = tuple(sorted(m_set))
        if key in feasibility_cache:
            return feasibility_cache[key]
        res = is_feasible(list(key))
        feasibility_cache[key] = res
        return res
        
    def dfs(idx, current_set, current_score):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx == n:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
            
        # Option 1: Exclude
        dfs(idx + 1, current_set, current_score)
        
        # Option 2: Include
        new_set = current_set + [moduli[idx]]
        if check_feasibility_cached(new_set):
            dfs(idx + 1, new_set, current_score + weights[idx])

    dfs(0, [], 0.0)
    
    residues = find_residues(best_set)
    result = [[m, r] for m, r in zip(best_set, residues)]
    return result

def is_feasible(moduli_set):
    if not moduli_set:
        return True
        
    indices = sorted(range(len(moduli_set)), key=lambda i: moduli_set[i])
    n = len(moduli_set)
    assigned = [None] * n
    
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli_set[indices[i]], moduli_set[indices[j]])
            gcds[j][i] = gcds[i][j]
            
    def backtrack(k):
        if k == n:
            return True
            
        idx = indices[k]
        m_k = moduli_set[idx]
        
        for v in range(m_k):
            conflict = False
            for j in range(k):
                idx_j = indices[j]
                d = gcds[idx][idx_j]
                if assigned[idx_j] is not None and v % d == assigned[idx_j] % d:
                    conflict = True
                    break
            if not conflict:
                assigned[idx] = v
                if backtrack(k + 1):
                    return True
        assigned[idx] = None
        return False
        
    return backtrack(0)

def find_residues(moduli_set):
    if not moduli_set:
        return []
        
    indices = sorted(range(len(moduli_set)), key=lambda i: moduli_set[i])
    n = len(moduli_set)
    assigned = [None] * n
    
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli_set[indices[i]], moduli_set[indices[j]])
            gcds[j][i] = gcds[i][j]
            
    def backtrack(k):
        if k == n:
            return True
            
        idx = indices[k]
        m_k = moduli_set[idx]
        
        for v in range(m_k):
            conflict = False
            for j in range(k):
                idx_j = indices[j]
                d = gcds[idx][idx_j]
                if assigned[idx_j] is not None and v % d == assigned[idx_j] % d:
                    conflict = True
                    break
            if not conflict:
                assigned[idx] = v
                if backtrack(k + 1):
                    return True
        assigned[idx] = None
        return False
        
    backtrack(0)
    return [assigned[i] for i in range(n)]

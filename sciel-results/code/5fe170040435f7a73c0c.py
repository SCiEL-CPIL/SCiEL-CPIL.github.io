import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocessing
    pos_counts = [0] * n
    neg_counts = [0] * n
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
    
    assignment = [None] * n
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if assignment[v] is not None:
                continue
            if pos_counts[v] > 0 and neg_counts[v] == 0:
                assignment[v] = True
                changed = True
            elif neg_counts[v] > 0 and pos_counts[v] == 0:
                assignment[v] = False
                changed = True
    
    new_clauses = []
    for c in clauses:
        new_c = []
        satisfied = False
        for lit in c:
            v = abs(lit) - 1
            if assignment[v] is not None:
                if assignment[v] == (lit > 0):
                    satisfied = True
                    break
                else:
                    continue
            else:
                new_c.append(lit)
        if not satisfied:
            if len(new_c) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            new_clauses.append(new_c)
    
    clauses = new_clauses
    
    if not clauses:
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                final_assignment[i] = random.random() < 0.5
        return final_assignment
    
    var_map = {}
    new_n = 0
    for v in range(n):
        if assignment[v] is None:
            var_map[v] = new_n
            new_n += 1
    
    remapped_clauses = []
    for c in clauses:
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            new_v = var_map[v]
            new_lit = new_v + 1
            if lit < 0:
                new_lit = -new_lit
            new_c.append(new_lit)
        remapped_clauses.append(new_c)
    
    m = len(remapped_clauses)
    
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    for i, c in enumerate(remapped_clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    var_to_clauses = [[] for _ in range(new_n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
    var_to_clauses_np = [np.array(vtc, dtype=np.int32) for vtc in var_to_clauses]
    
    def count_unsat(assignment_np):
        tv = assignment_np[clause_vars]
        lt = np.where(clause_signs > 0, tv, ~tv)
        csat = lt.any(axis=1)
        return int(np.sum(~csat)), ~csat
    
    def get_gains_for_clause(assignment_np, ci):
        gains = np.zeros(3, dtype=np.int32)
        for j in range(3):
            v = clause_vars[ci, j]
            vtc = var_to_clauses_np[v]
            if len(vtc) == 0:
                continue
            cv = clause_vars[vtc]
            cs = clause_signs[vtc]
            tv = assignment_np[cv]
            lt = np.where(cs > 0, tv, ~tv)
            csat = lt.any(axis=1)
            # Find position of v in each clause
            pos = (cv == v)
            for idx in range(len(vtc)):
                if not csat[idx]:
                    gains[j] += 1
                else:
                    # Check if v is the only true literal
                    if pos[idx].sum() == 1:
                        other_true = lt[idx].sum() == 1 and lt[idx][pos[idx].nonzero()[0][0]]
                        if other_true:
                            gains[j] -= 1
        return gains
    
    rng = random.Random(42)
    curr = np.array([rng.random() < 0.5 for _ in range(new_n)], dtype=bool)
    curr_unsat, _ = count_unsat(curr)
    best_unsat = curr_unsat
    best_assignment = curr.copy()
    
    if best_unsat == 0:
        final_assignment = [False] * n
        rev_map = {v: k for k, v in var_map.items()}
        for new_v, val in enumerate(best_assignment):
            final_assignment[rev_map[new_v]] = bool(val)
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
        return final_assignment
    
    def luby():
        level = 1
        while True:
            for _ in range(2 ** (level - 1)):
                yield 1
            level += 1
            if level > 20:
                level = 1
    
    luby_gen = luby()
    walk_prob = 0.3
    
    while time.time() < deadline:
        steps = next(luby_gen)
        step_deadline = time.time() + min(steps * 0.0005, (deadline - time.time()) * 0.4)
        
        for _ in range(steps):
            if time.time() > step_deadline or time.time() > deadline:
                break
            
            curr_unsat, unsat_mask = count_unsat(curr)
            
            if curr_unsat == 0:
                best_assignment = curr.copy()
                best_unsat = 0
                break
            
            if curr_unsat < best_unsat:
                best_unsat = curr_unsat
                best_assignment = curr.copy()
            
            if best_unsat == 0:
                break
            
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            
            if rng.random() < walk_prob:
                j = rng.randrange(3)
            else:
                gains = get_gains_for_clause(curr, ci)
                max_g = gains.max()
                candidates = np.where(gains == max_g)[0]
                j = int(candidates[rng.randrange(len(candidates))])
            
            v = int(clause_vars[ci, j])
            curr[v] = not curr[v]
        
        if best_unsat == 0:
            break
        
        if best_unsat > 15:
            walk_prob = min(0.8, walk_prob * 1.03)
        else:
            walk_prob = max(0.1, walk_prob * 0.97)
    
    final_assignment = [False] * n
    rev_map = {v: k for k, v in var_map.items()}
    for new_v, val in enumerate(best_assignment):
        final_assignment[rev_map[new_v]] = bool(val)
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
    
    return final_assignment

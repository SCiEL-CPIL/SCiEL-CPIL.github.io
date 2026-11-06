import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    moduli = list(range(20, 201))
    sorted_moduli = sorted(moduli, key=lambda m: -1.0/m)
    
    def is_compatible(m_new, r_new, existing):
        for m, r in existing:
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True
    
    def try_assign(m, existing):
        for r in range(m):
            if is_compatible(m, r, existing):
                return r
        return -1
    
    def score(assignment):
        return sum(1.0 / m for m, r in assignment)
    
    # Greedy initial
    assignment = []
    for m in sorted_moduli:
        r = try_assign(m, assignment)
        if r >= 0:
            assignment.append((m, r))
    
    def local_search(assignment, time_limit):
        assignment = list(assignment)
        cur_score = score(assignment)
        end = time.time() + time_limit
        while time.time() < end:
            improved = False
            in_set = set(m for m, r in assignment)
            
            for m in sorted_moduli:
                if m in in_set:
                    continue
                r = try_assign(m, assignment)
                if r >= 0:
                    assignment.append((m, r))
                    cur_score = score(assignment)
                    improved = True
                    break
            
            if not improved:
                for i in range(len(assignment)):
                    m_i, r_i = assignment[i]
                    for r_new in range(m_i):
                        if r_new == r_i:
                            continue
                        if not is_compatible(m_i, r_new, [(mm, rr) for j, (mm, rr) in enumerate(assignment) if j != i]):
                            continue
                        temp = list(assignment)
                        temp[i] = (m_i, r_new)
                        in_set2 = set(mm for mm, rr in temp)
                        for m_add in sorted_moduli:
                            if m_add in in_set2:
                                continue
                            r_add = try_assign(m_add, temp)
                            if r_add >= 0:
                                cand = temp + [(m_add, r_add)]
                                cs = score(cand)
                                if cs > cur_score:
                                    assignment = cand
                                    cur_score = cs
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return assignment, cur_score
    
    assignment, _ = local_search(assignment, 50)
    assignment, _ = local_search(assignment, 40)
    assignment, _ = local_search(assignment, 30)
    
    return [[m, r] for m, r in assignment]

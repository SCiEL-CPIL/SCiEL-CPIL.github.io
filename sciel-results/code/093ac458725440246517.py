import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 0.0
    best_result = []
    
    # Strategy 1: Greedy in order of m, smallest valid residue
    if time.time() < deadline:
        assigned = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            forbidden = set()
            for (m_i, r_i) in assigned:
                d = math.gcd(m, m_i)
                base = r_i % d
                for k in range(m // d):
                    forbidden.add((base + k * d) % m)
            r = 0
            while r < m and r in forbidden:
                r += 1
            if r < m:
                assigned.append((m, r))
        score = sum(1.0 / m for (m, r) in assigned)
        if score > best_score:
            best_score = score
            best_result = [[m, r] for (m, r) in assigned]
    
    # Strategy 2: Common factor K approach
    if time.time() < deadline:
        for K in range(1, 201):
            if time.time() > deadline:
                break
            multiples = [m for m in range(20, 201) if m % K == 0]
            if not multiples:
                continue
            k_pick = min(K, len(multiples))
            selected = sorted(multiples)[:k_pick]
            # Assign residues 0 to k_pick-1
            classes = [[m, i] for i, m in enumerate(selected)]
            # Verify disjointness
            valid = True
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    m1, r1 = classes[i]
                    m2, r2 = classes[j]
                    d = math.gcd(m1, m2)
                    if r1 % d == r2 % d:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                score = sum(1.0 / m for (m, r) in classes)
                if score > best_score:
                    best_score = score
                    best_result = classes
    
    if not best_result:
        # Fallback: simple valid construction
        best_result = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best_result

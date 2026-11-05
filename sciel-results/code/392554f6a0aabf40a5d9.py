import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 128
    # Precompute LCMs
    lcm_table = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i * j // gcd(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    S = set(range(1, N + 1))
    
    def find_bad_triples(s_set):
        triples = []
        s_list = sorted(s_set)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                lab = lcm_table[a][b]
                for k in range(j + 1, n):
                    c = s_list[k]
                    if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                        triples.append((a, b, c))
        return triples

    # Pruning Phase
    while time.time() < deadline:
        bad_triples = find_bad_triples(S)
        if not bad_triples:
            break
            
        counts = {}
        for a, b, c in bad_triples:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            counts[c] = counts.get(c, 0) + 1
            
        max_count = max(counts.values())
        candidates = [x for x, c in counts.items() if c == max_count]
        # Remove the smallest value among those with max count
        remove_val = min(candidates)
        S.remove(remove_val)
        
    # Polish Phase: Try to add back removed elements
    def can_add(x, current_set):
        if x in current_set:
            return True
        s_list = sorted(current_set)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                lab = lcm_table[a][b]
                lax = lcm_table[a][x]
                lxb = lcm_table[b][x]
                if lab == lax == lxb:
                    return False
        return True

    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x not in S:
            if can_add(x, S):
                S.add(x)
                
    return list(S)

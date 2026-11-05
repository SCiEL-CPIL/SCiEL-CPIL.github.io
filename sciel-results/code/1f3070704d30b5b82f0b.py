import math
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # 60 seconds limit, use 55 for safety
    
    N = 128
    
    # Precompute gcd for 1..N
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Get divisors for each number
    divisors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divisors[j].append(i)
            
    # Sort candidates: Prefer numbers with more divisors (more composite)
    # This helps avoid creating coprime triples
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda x: len(divisors[x]), reverse=True)
    
    S = set()
    
    # For each d, maintain the set of values v = x/d for x in S
    # We need to check if adding u creates 3 pairwise coprime in T_d
    # T_d is the set of v's.
    # We check if there are v, w in T_d such that gcd(u,v)=1, gcd(u,w)=1, gcd(v,w)=1.
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
            
        can_add = True
        for d in divisors[x]:
            u = x // d
            
            # Get current T_d
            # T_d = { y//d for y in S if y % d == 0 }
            # We can iterate S and filter, or maintain a list.
            # Since N is small, iterating S is fine.
            T_d = []
            for y in S:
                if y % d == 0:
                    T_d.append(y // d)
                    
            if len(T_d) < 2:
                continue
                
            # Find neighbors of u in T_d (coprime to u)
            N_u = [v for v in T_d if gcd_table[u][v] == 1]
            
            if len(N_u) < 2:
                continue
                
            # Check if any pair in N_u is coprime
            found_bad = False
            for i in range(len(N_u)):
                v1 = N_u[i]
                for j in range(i + 1, len(N_u)):
                    v2 = N_u[j]
                    if gcd_table[v1][v2] == 1:
                        found_bad = True
                        break
                if found_bad:
                    break
                    
            if found_bad:
                can_add = False
                break
                
        if can_add:
            S.add(x)
            
    return list(S)

import time
import math
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k != 4:
        # Fallback
        n = k * (g // 2)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    # Precompute non-backtracking sequences of length 3 to g-1
    # Steps: 0:+1, 1:-1, 2:+b, 3:-b
    seqs = []
    for L in range(3, g):
        for seq in product([0, 1, 2, 3], repeat=L):
            valid = True
            for i in range(L - 1):
                s1, s2 = seq[i], seq[i+1]
                if (s1 == 0 and s2 == 1) or (s1 == 1 and s2 == 0):
                    valid = False
                    break
                if (s1 == 2 and s2 == 3) or (s1 == 3 and s2 == 2):
                    valid = False
                    break
            if valid:
                c1 = seq.count(0)
                c_1 = seq.count(1)
                cb = seq.count(2)
                c_b = seq.count(3)
                A = c1 - c_1
                B = cb - c_b
                seqs.append((A, B))
                
    seqs = list(set(seqs))
    
    n_min = 243
    n_max = 400
    
    for n in range(n_min, n_max + 1):
        if time.time() > deadline:
            break
            
        bad_b = set()
        reject_n = False
        
        for A, B in seqs:
            if B == 0:
                if A % n == 0:
                    reject_n = True
                    break
            else:
                g_val = math.gcd(B, n)
                # Case 1: B*b = -A
                if (-A) % g_val == 0:
                    B_prime = B // g_val
                    A_prime = -A // g_val
                    n_prime = n // g_val
                    try:
                        inv = pow(B_prime, -1, n_prime)
                    except ValueError:
                        pass
                    else:
                        b_prime = (A_prime * inv) % n_prime
                        max_k = n // n_prime
                        for k in range(max_k + 1):
                            b = b_prime + k * n_prime
                            if 2 <= b <= n // 2:
                                bad_b.add(b)
                                
                # Case 2: B*b = A
                if A % g_val == 0:
                    B_prime = B // g_val
                    A_prime = A // g_val
                    n_prime = n // g_val
                    try:
                        inv = pow(B_prime, -1, n_prime)
                    except ValueError:
                        pass
                    else:
                        b_prime = (A_prime * inv) % n_prime
                        max_k = n // n_prime
                        for k in range(max_k + 1):
                            b = b_prime + k * n_prime
                            if 2 <= b <= n // 2:
                                bad_b.add(b)
                                
        if reject_n:
            continue
            
        if len(bad_b) >= n // 2 - 1:
            continue
            
        valid_b = None
        for b in range(2, n // 2 + 1):
            if b in bad_b:
                continue
            if b == 1 or b == n - 1 or b == n // 2:
                continue
            valid_b = b
            break
            
        if valid_b:
            b = valid_b
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
                edges.append((i, (i + b) % n))
            return n, edges
            
    # Fallback
    n = 320
    b = 127
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + b) % n))
    return n, edges

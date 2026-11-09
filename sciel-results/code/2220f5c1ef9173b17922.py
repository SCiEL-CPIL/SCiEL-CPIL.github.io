import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p1, p2 = 5, 41
    
    # CRT constants: x = a*p2 + b*p1*(p2*inv(p2 mod p1, p1)) mod m
    # p2 mod p1 = 41 mod 5 = 1, inv(1,5) = 1
    # So x = a*41 + b*5*(41*1) = a*41 + b*205 ≡ a*41 + b*0? Wait, let me recompute.
    # x ≡ a (mod 5), x ≡ b (mod 41)
    # x = a + 5*k, need a + 5k ≡ b (mod 41) => 5k ≡ b-a (mod 41) => k ≡ (b-a)*inv(5,41) (mod 41)
    # inv(5,41): 5*33 = 165 = 4*41+1, so inv=33
    # k = (b-a)*33 mod 41
    # x = a + 5*((b-a)*33 % 41)
    # Let me verify: a=1,b=2: k=(2-1)*33%41=33, x=1+5*33=166. 166%5=1✓, 166%41=166-4*41=166-164=2✓
    
    inv_5_41 = 33  # 5*33=165=4*41+1
    
    def crt(a, b):
        """Combine a mod 5 and b mod 41 to get x mod 205"""
        k = ((b - a) * inv_5_41) % 41
        return (a + 5 * k) % 205
    
    # For each modulus, find forbidden differences
    def get_forbidden(p):
        forb = set()
        for x in range(1, p):
            forb.add(pow(x, 2, p))
        forb.discard(0)
        return forb
    
    forb5 = get_forbidden(5)
    forb41 = get_forbidden(41)
    
    # For Z_5: residues are 0,1,2,3,4. Squares mod 5: 1^2=1, 2^2=4, 3^2=4, 4^2=1. So forb5={1,4}
    # Forbidden differences: ±1, ±4 ≡ ±1, ±4. So can't have consecutive or diff 4.
    # Valid sets in Z_5: {0,2}, {0,3}, {1,3}, {1,4}, {2,4}, etc. Max size is 2.
    
    # For Z_41: find max independent set in graph where edges connect x,y if (x-y)%41 in forb41
    # This is a graph with 41 vertices. We can use a simple approach: try all subsets or use greedy+local search.
    # Since 41 is small, we can be more thorough.
    
    def find_max_valid_set(p, forbidden_set, max_tries=1000, time_limit=30):
        """Find large valid set in Z_p using greedy + local search"""
        start_time = time.time()
        best = set()
        best_bool = np.zeros(p, dtype=bool)
        
        # Precompute conflict matrix
        conflict = np.zeros((p, p), dtype=bool)
        for i in range(p):
            for j in range(p):
                if i != j and (i - j) % p in forbidden_set:
                    conflict[i][j] = True
                    conflict[j][i] = True
        
        def can_add(v, S_bool):
            return not np.any(S_bool & conflict[v])
        
        def greedy(order):
            S = set()
            S_bool = np.zeros(p, dtype=bool)
            for i in order:
                if not S_bool[i] and can_add(i, S_bool):
                    S.add(i)
                    S_bool[i] = True
            return S, S_bool
        
        def local_search(S, S_bool):
            improved = True
            while improved:
                improved = False
                # Try add 1
                for v in range(p):
                    if not S_bool[v] and can_add(v, S_bool):
                        S.add(v)
                        S_bool[v] = True
                        improved = True
                        break
                if improved:
                    continue
                # Try remove 1, add 2
                S_list = list(S)
                for u in S_list:
                    S_bool[u] = False
                    S.remove(u)
                    valid = [v for v in range(p) if not S_bool[v] and can_add(v, S_bool)]
                    found = False
                    for i in range(len(valid)):
                        for j in range(i+1, len(valid)):
                            if not conflict[valid[i]][valid[j]]:
                                S.add(valid[i])
                                S_bool[valid[i]] = True
                                S.add(valid[j])
                                S_bool[valid[j]] = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        improved = True
                        break
                    S.add(u)
                    S_bool[u] = True
            return S, S_bool
        
        # Try multiple random orders
        for trial in range(max_tries):
            if time.time() - start_time > time_limit:
                break
            order = list(range(p))
            random.shuffle(order)
            S, S_bool = greedy(order)
            S, S_bool = local_search(S, S_bool)
            if len(S) > len(best):
                best = S.copy()
                best_bool = S_bool.copy()
        
        return best, best_bool
    
    # Find max valid sets in Z_5 and Z_41
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # For Z_5, just enumerate all subsets (2^5=32)
    best5 = set()
    for mask in range(32):
        S5 = set(i for i in range(5) if mask & (1 << i))
        valid = True
        for a in S5:
            for b in S5:
                if a != b and (a - b) % 5 in forb5:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(S5) > len(best5):
            best5 = S5.copy()
    
    # For Z_41, use greedy + local search
    best41, _ = find_max_valid_set(41, forb41, max_tries=2000, time_limit=400)
    
    # Combine via CRT: for each a in best5, b in best41, add crt(a,b) to result
    result = set()
    for a in best5:
        for b in best41:
            result.add(crt(a, b))
    
    # Verify the result
    forb205 = set()
    for x in range(1, 205):
        forb205.add(pow(x, 2, 205))
    forb205.discard(0)
    
    valid = True
    for a in result:
        for b in result:
            if a != b:
                d = (a - b) % 205
                if d in forb205:
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple greedy if CRT combination failed (shouldn't happen)
        S = set()
        S_bool = np.zeros(205, dtype=bool)
        for i in range(205):
            if can_add(i, S_bool, forb205):
                S.add(i)
                S_bool[i] = True
        return np.array(list(S), dtype=np.int64)
    
    return np.array(list(result), dtype=np.int64)

def can_add(v, S_bool, forbidden_set):
    """Check if v can be added to set represented by S_bool"""
    for i in range(len(S_bool)):
        if S_bool[i] and (v - i) % len(S_bool) in forbidden_set:
            return False
    return True

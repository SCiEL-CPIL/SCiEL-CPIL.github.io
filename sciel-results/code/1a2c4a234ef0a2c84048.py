import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 128
    S = set(range(64, N + 1))
    best_S = S.copy()
    best_size = len(S)
    
    def is_valid_full(S):
        S_list = list(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                for k in range(j+1, n):
                    c = S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True

    def get_witness(x, S):
        for b in S:
            if b == x: continue
            sb = x + b
            if sb > N or sb not in S: continue
            for c in S:
                if c == x or c == b: continue
                sc = x + c
                if sc > N or sc not in S: continue
                sbc = b + c
                if sbc > N: continue
                if sbc in S:
                    return (x, b, c)
        
        for a in S:
            if a >= x: continue
            b = x - a
            if b <= 0 or b == a or b not in S: continue
            for c in S:
                if c == a or c == b: continue
                sac = a + c
                if sac > N or sac not in S: continue
                sbc = b + c
                if sbc > N: continue
                if sbc in S:
                    return (a, b, c)
                    
        return None

    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb
            if S:
                r1 = random.choice(list(S))
                S.remove(r1)
                r2 = random.randint(1, N)
                if r2 not in S:
                    S.add(r2)
            no_improve_count = 0
            max_no_improve += 100
            continue

        candidates = [x for x in range(1, N+1) if x not in S]
        if not candidates:
            break
        
        x = random.choice(candidates)
        witness = get_witness(x, S)
        
        if witness is None:
            S.add(x)
            no_improve_count = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
        else:
            a, b, c = witness
            potential_removals = set()
            if a != x: potential_removals.add(a)
            if b != x: potential_removals.add(b)
            if c != x: potential_removals.add(c)
            for s in [a+b, a+c, b+c]:
                if s != x and s in S:
                    potential_removals.add(s)
            
            found_repair = False
            # Limit repair attempts to avoid timeout
            for r in random.sample(list(potential_removals), min(5, len(potential_removals))):
                if r not in S: continue
                
                S.remove(r)
                S.add(x)
                
                if is_valid_full(S):
                    no_improve_count = 0
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                    found_repair = True
                    break
                else:
                    S.remove(x)
                    S.add(r)
            
            if not found_repair:
                no_improve_count += 1

    if not is_valid_full(best_S):
        return list(range(65, 129))
        
    return list(best_S)

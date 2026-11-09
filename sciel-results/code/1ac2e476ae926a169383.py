import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    random.seed(42)
    
    def score(s):
        s_set = set(s)
        count = 0
        for d in range(1, 256):
            for a in s:
                if a + d in s_set and a + 2 * d in s_set:
                    count += 1
                    break
        return count
    
    best_set = None
    best_score = -1
    
    # Try several initializations
    inits = [
        list(range(1, 33)),
        random.sample(range(1, 513), 32),
        [1 + 16 * i for i in range(32)],
        sorted(random.sample(range(1, 513), 32)),
        list(range(1, 17)) + list(range(497, 513)),
    ]
    
    for s0 in inits:
        if time.time() > deadline - 15:
            break
        
        s = s0[:]
        cur_score = score(s)
        if cur_score > best_score:
            best_score = cur_score
            best_set = s[:]
        
        T = 5.0
        T_min = 0.01
        cooling = 0.99998
        
        while time.time() < deadline - 3:
            r = random.random()
            if r < 0.4:
                i = random.randint(0, 31)
                nv = random.randint(1, 512)
                if nv in s:
                    continue
                ns = s[:]
                ns[i] = nv
            elif r < 0.7:
                i, j = random.sample(range(32), 2)
                ns = s[:]
                ns[i], ns[j] = ns[j], ns[i]
            else:
                i = random.randint(0, 31)
                delta = random.randint(-50, 50)
                nv = s[i] + delta
                if nv < 1 or nv > 512 or nv in s:
                    continue
                ns = s[:]
                ns[i] = nv
            
            ns_score = score(ns)
            
            if ns_score >= cur_score:
                s = ns
                cur_score = ns_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = s[:]
            else:
                if random.random() < pow(2.718281828, -(cur_score - ns_score) / T):
                    s = ns
                    cur_score = ns_score
            
            T = max(T_min, T * cooling)
    
    return best_set

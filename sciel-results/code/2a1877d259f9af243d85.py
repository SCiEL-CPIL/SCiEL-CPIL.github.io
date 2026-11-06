import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 4096
    
    def can_add(x, S, P, T):
        # S: list of elements
        # P: set of pair sums (a+b for a<=b in S)
        # T: set of triple sums
        
        # 1. Check if 2x is in P (collision 3x = x+a+b => 2x=a+b)
        if 2*x in P:
            return False
        
        # 2. Check if x+s is in P for any s in S
        # Collision: 2x+a = x+c+d => x+a = c+d
        for s in S:
            if x + s in P:
                return False
        
        # 3. Check if 3x is in T
        if 3*x in T:
            return False
        
        # 4. Check if 2x+s is in T for any s in S
        for s in S:
            if 2*x + s in T:
                return False
        
        # 5. Check if x+p is in T for any p in P
        # Collision: x+a+b = c+d+e (where c,d,e in S)
        # Or x+a+b = x+c+d (handled by 2)
        # Or x+a+b = 2x+c (handled by 2: x+a = x+c => a=c)
        # Or x+a+b = 3x (handled by 1: a+b = 2x)
        # So we just need to check against existing T
        for p in P:
            if x + p in T:
                return False
        
        return True

    def add_element(x, S, P, T):
        # Update P
        P.add(2*x)
        for s in S:
            P.add(x + s)
        
        # Update T
        T.add(3*x)
        for s in S:
            T.add(2*x + s)
        
        # Update T with x + p for p in OLD P
        # We need to iterate over P before adding new pairs?
        # Or just iterate over S to form pairs?
        # Using P is faster if we snapshot it.
        # But P is a set, we can just iterate it.
        # However, we just added new pairs to P.
        # We need x + (old pair sums).
        # Let's re-calculate or use a list of old pairs?
        # Actually, x + (x+s) = 2x+s (already added).
        # x + (2x) = 3x (already added).
        # x + (s1+s2) where s1,s2 in old S.
        # These are exactly the triples (x, s1, s2).
        # We can generate them by iterating i, j in S.
        # But we already added 2x+s.
        # So we just need to add x + s1 + s2 for s1 <= s2 in S.
        
        # To be safe and simple, let's just iterate S for pairs.
        # It's O(k^2).
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                T.add(x + a + b)
                
        S.append(x)

    def build_greedy(seed):
        random.seed(seed)
        S = []
        P = set()
        T = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, P, T):
                add_element(x, S, P, T)
                
                if time.time() > deadline - 5:
                    break
                    
        return S

    best_set = []
    best_size = 0
    
    num_restarts = 200
    base_seed = 42
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_set = build_greedy(base_seed + i)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Optional: If we hit a high size, we could try local search, 
        # but randomized greedy is often sufficient for this constraint size.
        
    return best_set

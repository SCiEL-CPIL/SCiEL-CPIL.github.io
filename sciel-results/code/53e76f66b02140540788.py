import random
import time

def _try_greedy(order):
    S = []
    S_3_sums = set()
    # pair_sums_mod[a] = set of (u+v) % a for u,v in S \ {a}, u != v
    pair_sums_mod = {}
    
    for x in order:
        valid = True
        
        # Check 1: Does x divide any existing sum of 3 distinct elements in S?
        if S_3_sums:
            for s in S_3_sums:
                if s % x == 0:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Check 2: For each existing a in S, does a divide any NEW sum involving x?
        # Condition: Exists u, v in S \ {a} such that (x + u + v) % a == 0
        # Equivalent to: (-x % a) is in pair_sums_mod[a]
        
        if S:
            for a in S:
                target = (-x) % a
                if target in pair_sums_mod[a]:
                    valid = False
                    break
        
        if valid:
            # Add x to S
            S.append(x)
            
            # Update S_3_sums with new sums involving x
            # New sums are x + u + v for all u, v in S_old
            old_S = S[:-1]
            for i in range(len(old_S)):
                u = old_S[i]
                for j in range(i + 1, len(old_S)):
                    v = old_S[j]
                    S_3_sums.add(x + u + v)
            
            # Update pair_sums_mod
            # 1. For existing a in old_S, add new pairs (x, u) for u in old_S \ {a}
            for a in old_S:
                if a not in pair_sums_mod:
                    pair_sums_mod[a] = set()
                for u in old_S:
                    if u != a:
                        pair_sums_mod[a].add((x + u) % a)
            
            # 2. Initialize pair_sums_mod[x] with pairs from old_S
            pair_sums_mod[x] = set()
            for i in range(len(old_S)):
                u = old_S[i]
                for j in range(i + 1, len(old_S)):
                    v = old_S[j]
                    pair_sums_mod[x].add((u + v) % x)
                    
    return S

def construct():
    best_set = []
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    # 1. Baseline: Descending order
    order_desc = list(range(192, 0, -1))
    best_set = _try_greedy(order_desc)
    
    # 2. Randomized Restarts
    # Shuffle and try
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        # Randomize the order. 
        # Heuristic: Maybe keep some structure? 
        # Simple random shuffle is a good start.
        order = list(range(1, 193))
        random.shuffle(order)
        
        # Optional: Bias towards larger numbers? 
        # We can sort by random weight + value?
        # Let's just use pure random for now to escape local minima.
        
        current_set = _try_greedy(order)
        if len(current_set) > len(best_set):
            best_set = current_set
            # If we found a very good set, we might stop early? 
            # Max is 48. If we hit 48, we can return.
            if len(best_set) == 48:
                break

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))

import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n

    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_subset(family[i], family[j]) or is_subset(family[j], family[i]):
                    count += 1
        return count

    def local_search(family, max_iters, deadline):
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        no_improve = 0
        threshold = 5000
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            i = random.randint(0, m-1)
            new_mask = random.randint(0, N-1)
            
            if new_mask == family[i]:
                continue
            if new_mask in family[:i] + family[i+1:]:
                continue
            
            old_mask = family[i]
            
            lost = 0
            gained = 0
            for j in range(m):
                if j == i:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1
            
            delta = gained - lost
            
            if delta > 0:
                family[i] = new_mask
                current_score += delta
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                no_improve += 1
                
            if no_improve > threshold:
                break

        return best_family, best_score

    def construct_star_structure():
        # Structure: {0}
        # L1: {0, i} for i=1..11 (11 sets)
        # L2: {0, i, j} for i<j, i,j in 1..11 (55 sets)
        # L3: {0, i, j, k} ...
        
        # Let's target 40 sets.
        # 1 (L0) + 11 (L1) = 12 sets.
        # We have 28 slots left for L2.
        # If we take 28 sets from L2 (which are all comparable to L0 and L1 if they contain 0),
        # Total comparable pairs:
        # L0-L1: 11
        # L0-L2: 28
        # L1-L2: Each L2 set {0,i,j} is comparable to L1 sets {0,i} and {0,j}.
        # So 2 * 28 = 56 pairs? 
        # Wait, are L1 sets comparable to each other? No.
        # Are L2 sets comparable to each other? No (same size).
        # So total = 11 + 28 + 56 = 95.
        
        # Can we do better?
        # What if we use L3 sets?
        # L3 set {0,i,j,k} is comparable to:
        # L0: 1
        # L1: 3 ({0,i}, {0,j}, {0,k})
        # L2: 3 ({0,i,j}, {0,i,k}, {0,j,k})
        # Total 7 comparables per L3 set.
        # L2 set has 3 comparables (L0, 2 L1s).
        # So L3 is more "efficient" per set if we have enough support.
        
        # Let's try a mix.
        # 1 L0
        # 11 L1
        # x L2
        # y L3
        # 1 + 11 + x + y = 40 => x + y = 28
        
        # Pairs:
        # L0-L1: 11
        # L0-L2: x
        # L0-L3: y
        # L1-L2: 2x (each L2 connects to 2 L1s)
        # L1-L3: 3y (each L3 connects to 3 L1s)
        # L2-L3: 3y (each L3 connects to 3 L2s, assuming those L2s are present)
        
        # We need to ensure the L2s required by L3s are present.
        # If we pick L3s randomly, we need specific L2s.
        # To maximize, we should pick L2s that are shared by many L3s.
        
        # Let's just generate a random star structure and let local search fix it.
        # Or generate a specific one.
        
        sets = [0] # L0: {0}
        l1 = []
        for i in range(1, n):
            sets.append(1 | (1 << i))
            l1.append(1 | (1 << i))
            
        # We need 28 more sets.
        # Let's pick 28 L2 sets that form a dense subgraph or just random.
        # Actually, let's pick L2 sets that share indices to allow L3s.
        # But for initial construction, random star is fine.
        
        candidates_l2 = []
        for i in range(1, n):
            for j in range(i+1, n):
                candidates_l2.append(1 | (1 << i) | (1 << j))
        
        random.shuffle(candidates_l2)
        
        # Pick 28
        selected_l2 = candidates_l2[:28]
        sets.extend(selected_l2)
        
        return sets

    best_overall = None
    best_score_overall = 0
    
    # Restart 0: Star structure
    if time.time() < deadline - 10:
        fam0 = construct_star_structure()
        fam, score = local_search(fam0, 200000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
            
    # Restart 1: Chain (original)
    if time.time() < deadline - 10:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(42)
        extra = random.sample(remaining, m - len(chain))
        fam1 = chain + extra
        fam, score = local_search(fam1, 200000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
            
    # Restarts 2-8: Random
    for restart in range(2, 9):
        if time.time() > deadline - 10:
            break
        
        random.seed(100 + restart)
        family = random.sample(range(N), m)
        
        fam, score = local_search(family, 200000, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result

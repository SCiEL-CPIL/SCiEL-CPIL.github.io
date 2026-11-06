import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 512
    K = 32
    
    # Precompute: for each d in 1..255, list of bitmasks representing (a, a+d, a+2d)
    # A 3-AP with difference d exists iff there's some a where all three bits are set
    # For scoring: count d where any (1<<(a-1)) & (1<<(a+d-1)) & (1<<(a+2d-1)) is subset of mask
    
    # For each d, precompute the list of triple bitmasks
    # triple_mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2d-1))
    # A 3-AP exists for d iff (mask & triple_mask) == triple_mask for some triple
    
    ap_triples = {}
    for d in range(1, N // 2 + 1):
        triples = []
        for a in range(1, N - 2 * d + 1):
            # bits at positions a-1, a+d-1, a+2d-1
            bm = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a + 2 * d - 1))
            triples.append(bm)
        ap_triples[d] = triples
    
    def score_mask(mask):
        """
        mask: Python int with bits set for elements in the set
        Returns: number of distinct d > 0 such that a 3-AP with difference d exists
        """
        count = 0
        for d in range(1, N // 2 + 1):
            for bm in ap_triples[d]:
                if (mask & bm) == bm:
                    count += 1
                    break
        return count
    
    def greedy_init(seed=42):
        """
        Greedily build a set of K elements to maximize score using bitmask.
        """
        random.seed(seed)
        
        mask = 0
        S = set()
        
        # Add first element randomly
        first = random.randint(1, N)
        S.add(first)
        mask |= (1 << (first - 1))
        
        for _ in range(K - 1):
            best_val = -1
            best_score = -1
            
            for val in range(1, N + 1):
                if val in S:
                    continue
                
                new_mask = mask | (1 << (val - 1))
                sc = score_mask(new_mask)
                
                if sc > best_score:
                    best_score = sc
                    best_val = val
            
            if best_val != -1:
                S.add(best_val)
                mask |= (1 << (best_val - 1))
            else:
                candidates = [v for v in range(1, N + 1) if v not in S]
                if candidates:
                    val = random.choice(candidates)
                    S.add(val)
                    mask |= (1 << (val - 1))
        
        return list(S), mask

    # Initial best set
    best_set, best_mask = greedy_init(seed=42)
    best_score = score_mask(best_mask)
    
    # Simulated Annealing with bitmask
    current_set = list(best_set)
    current_mask = best_mask
    current_score = best_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9998
    
    steps = 0
    while time.time() < deadline and T > T_min:
        # Generate a neighbor: swap one element in S with one not in S
        idx = random.randint(0, K - 1)
        old_val = current_set[idx]
        
        # Pick a random new_val not in current_set
        new_val = random.randint(1, N)
        while new_val in current_set:
            new_val = random.randint(1, N)
        
        # Apply swap to mask
        new_mask = (current_mask & ~(1 << (old_val - 1))) | (1 << (new_val - 1))
        
        new_score = score_mask(new_mask)
        delta = new_score - current_score
        
        if delta > 0 or (delta >= -5 and random.random() < math.exp(delta / T)):
            # Accept
            current_set[idx] = new_val
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
                best_mask = current_mask
        else:
            # Reject: mask unchanged
            pass
        
        T *= cooling
        steps += 1
        
        # Reheat if stuck
        if T < 1.0 and random.random() < 0.002:
            T = 10.0
            current_set = best_set.copy()
            current_mask = best_mask
            current_score = best_score

    return best_set

# Quick self-test
if __name__ == "__main__":
    s = time.time()
    res = construct()
    print(f"Time: {time.time() - s:.2f}s")
    print(f"Result: {res}")
    print(f"Len: {len(res)}")
    print(f"Distinct: {len(set(res)) == 32}")
    print(f"Range: {min(res) >= 1 and max(res) <= 512}")
    
    # Calculate score for verification
    mask = 0
    for x in res:
        mask |= (1 << (x - 1))
    sc = 0
    for d in range(1, 256):
        found = False
        for a in range(1, 512 - 2*d + 1):
            bm = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            if (mask & bm) == bm:
                found = True
                break
        if found:
            sc += 1
    print(f"Score: {sc}")

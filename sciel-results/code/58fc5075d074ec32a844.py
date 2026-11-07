import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Bitset representation: U[c] is a Python int where bit i is set if sum i is achievable at cardinality c
    # Max possible sum: 60 * 256 = 15360, so we need bits 0..15360
    MAX_SUM = 15360
    U = [0] * 61
    U[0] = 1  # sum 0 at cardinality 0
    
    used = set()
    best_result = []
    
    # Structured candidate orderings to try
    # 1. Small to large
    # 2. Powers of 2 first, then others
    # 3. Random restarts with different seeds
    
    def try_add(x, U, used):
        """Try to add x to the current set. Returns (success, new_U) if successful."""
        if x in used:
            return False, U
        
        # Find current max cardinality
        current_max_k = 0
        for k in range(60, -1, -1):
            if U[k]:
                current_max_k = k
                break
        
        # Check collision: for each c, new_sums = U[c] << x must not intersect U[k] for k != c+1
        for c in range(current_max_k + 1):
            if not U[c]:
                continue
            new_sums = U[c] << x
            # Check against all U[k] where k != c+1
            for k in range(current_max_k + 2):
                if k == c + 1:
                    continue
                if U[k] and (new_sums & U[k]):
                    return False, U
        
        # Valid: update U
        new_U = U[:]
        for c in range(current_max_k, -1, -1):
            if U[c]:
                new_sums = U[c] << x
                new_U[c + 1] |= new_sums
        return True, new_U
    
    def greedy_build(candidate_order, U, used, deadline):
        """Greedy build using given candidate order."""
        U_local = U[:]
        used_local = set(used)
        
        for x in candidate_order:
            if time.time() > deadline:
                break
            if x in used_local:
                continue
            
            success, new_U = try_add(x, U_local, used_local)
            if success:
                U_local = new_U
                used_local.add(x)
        
        return U_local, used_local
    
    # Strategy 1: Greedy small-to-large
    candidates_small = list(range(1, 257))
    U1, used1 = greedy_build(candidates_small, U, used, deadline)
    best_result = list(used1)
    
    # Strategy 2: Powers of 2 first, then small numbers
    powers = [2**i for i in range(0, 8) if 2**i <= 256]
    remaining = [x for x in range(1, 257) if x not in set(powers)]
    candidates_pow = powers + remaining
    if time.time() < deadline:
        U2, used2 = greedy_build(candidates_pow, U, used, deadline)
        if len(used2) > len(best_result):
            best_result = list(used2)
    
    # Strategy 3: Random restarts with different orderings
    if time.time() < deadline - 5:
        rng = random.Random(42)
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            # Create a random ordering with some structure
            base = list(range(1, 257))
            # Shuffle but keep some small numbers early
            small = [x for x in base if x <= 50]
            large = [x for x in base if x > 50]
            rng.shuffle(small)
            rng.shuffle(large)
            candidates_rand = small[:10] + large + small[10:]
            
            U3, used3 = greedy_build(candidates_rand, U, used, deadline)
            if len(used3) > len(best_result):
                best_result = list(used3)
    
    # Strategy 4: Try adding numbers in geometric progression
    if time.time() < deadline - 3:
        for ratio in [2, 3, 4]:
            if time.time() > deadline - 2:
                break
            geo = []
            v = 1
            while v <= 256:
                geo.append(v)
                v *= ratio
            rest = [x for x in range(1, 257) if x not in set(geo)]
            rng = random.Random(123)
            rng.shuffle(rest)
            candidates_geo = geo + rest
            U4, used4 = greedy_build(candidates_geo, U, used, deadline)
            if len(used4) > len(best_result):
                best_result = list(used4)
    
    result = [int(x) for x in best_result]
    result.sort()
    
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    print(res)

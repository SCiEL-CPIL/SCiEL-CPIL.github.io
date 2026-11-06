import time
import random

N = 256

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin

    def is_valid(x, S, sums):
        if 2 * x in sums:
            return False
        for s in S:
            if x + s in sums:
                return False
        return True

    def is_maximal(S, sums):
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            if is_valid(y, S, sums):
                return False
        return True

    while time.time() < deadline:
        # Strategy 1: Greedy with Blocking Score
        S = []
        sums = set()
        candidates = set(range(1, N + 1))
        
        while candidates:
            valid_candidates = [x for x in candidates if is_valid(x, S, sums)]
            if not valid_candidates:
                break
            
            best_x = None
            max_block = -1
            
            for x in valid_candidates:
                # Estimate blocking power: count y in candidates that become blocked
                block_count = 0
                # 2y = x + s'  => y = (x+s')/2
                for s_prime in S:
                    y = (x + s_prime) / 2
                    if y == int(y) and y in candidates and y != x:
                        block_count += 1
                
                # y + s = 2x => y = 2x - s
                for s in S:
                    y = 2 * x - s
                    if y in candidates and y != x:
                        block_count += 1
                        
                # y + s = x + s' => y = x + s' - s
                # This implies y - s' = x - s
                # We can precompute differences? No, S is small.
                # Just iterate
                for s in S:
                    for s_prime in S:
                        y = x + s_prime - s
                        if y in candidates and y != x:
                            block_count += 1
                            break # One collision is enough to block y? 
                            # Actually, if y+s collides with ANY sum, y is blocked.
                            # So if we find one, y is blocked. We count y once.
                            # But we are iterating s, s'. We should just check if y is blocked.
                            # The condition is: exists s, s' such that y+s = x+s' AND y+s is a sum in S U {x}?
                            # No, y+s must collide with a sum in S U {x}.
                            # The sums in S U {x} are sums(S) U {2x} U {x+s' for s' in S}.
                            # We already checked 2y and y+s vs sums(S) (y was valid).
                            # So we only need to check collisions with NEW sums: 2x and x+s'.
                            # 1. y+s = 2x => y = 2x-s (checked above)
                            # 2. y+s = x+s' => y = x+s'-s (checking here)
                            # If either holds, y is blocked.
                            # So we should count y if EITHER condition holds.
                            # My loop above counts y multiple times if multiple s,s' work.
                            # We should use a set of blocked y's.
                
                # Refine: Use a set for blocked y's to avoid double counting
                # But for speed, a rough count might suffice to rank x.
                
                if block_count > max_block:
                    max_block = block_count
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            sums.add(2 * best_x)
            for s in S[:-1]:
                sums.add(best_x + s)
            candidates.remove(best_x)
            
            if is_maximal(S, sums):
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S[:]
                break

        # Strategy 2: Random Greedy (if time remains)
        if time.time() < deadline:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            S = []
            sums = set()
            for x in candidates:
                if is_valid(x, S, sums):
                    S.append(x)
                    sums.add(2 * x)
                    for s in S[:-1]:
                        sums.add(best_x + s) # BUG: best_x vs x
                    # Fix:
                    # sums.add(x + s)
                    
                    if is_maximal(S, sums):
                        if len(S) < best_size:
                            best_size = len(S)
                            best_set = S[:]
                        break

    return best_set

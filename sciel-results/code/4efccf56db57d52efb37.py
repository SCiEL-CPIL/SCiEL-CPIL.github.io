import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    OFFSET = 25000
    # Max sum magnitude ~ 25000. Range 50000.
    # Bitset size 50001 is enough.
    
    def greedy(order):
        S = []
        dp_S = [0] * 50
        dp_S[0] = 1
        
        possible_sums = []
        
        for y in order:
            if time.time() > deadline:
                break
                
            valid = True
            
            # Check y is not mean of S
            # Sums in dp_S are positive.
            for k in range(2, len(S) + 1):
                target = k * y
                # Max sum in dp_S[k] is k * 512.
                # If target > k * 512, it can't be there.
                # But we can just check the bit.
                if target < 50000: # Safe bound
                    if (dp_S[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check a in S
            for i in range(len(S)):
                a = S[i]
                diff = a - y
                idx = diff + OFFSET
                if 0 <= idx < 50001:
                    if (possible_sums[i] >> idx) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add y
            S.append(y)
            
            # Update dp_S
            for k in range(len(S), 0, -1):
                dp_S[k] |= (dp_S[k-1] << y)
                
            # Update possible_sums
            for i in range(len(S) - 1):
                a = S[i]
                val = y - a
                old_bs = possible_sums[i]
                
                if val > 0:
                    new_bs = old_bs | (old_bs << val) | (1 << (val + OFFSET))
                elif val < 0:
                    new_bs = old_bs | (old_bs >> (-val)) | (1 << (val + OFFSET))
                else:
                    new_bs = old_bs | (1 << OFFSET)
                    
                possible_sums[i] = new_bs
                
            possible_sums.append(0)
            
        return S

    # Try a few orders
    best = []
    
    # 1..512
    res = greedy(range(1, 513))
    if len(res) > len(best): best = res
    
    # 512..1
    res = greedy(range(512, 0, -1))
    if len(res) > len(best): best = res
    
    # Random
    import random
    for _ in range(10):
        if time.time() > deadline - 5.0: break
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        res = greedy(rand_order)
        if len(res) > len(best): best = res
        
    return best

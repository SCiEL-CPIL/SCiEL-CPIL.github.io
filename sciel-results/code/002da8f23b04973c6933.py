import time

def construct():
    deadline = time.time() + 85  # 90s limit, 5s margin
    
    best_set = []
    
    # Strategy 1: Greedy Top-Down
    def greedy_top_down():
        S = []
        sums = [0] * 61
        sums[0] = 1
        for x in range(256, 0, -1):
            if time.time() > deadline:
                return S
            valid = True
            new_sums = sums[:]  # Copy
            for i in range(1, 61):
                if sums[i-1] == 0:
                    continue
                shifted = sums[i-1] << x
                new_sums[i] |= shifted
                for j in range(61):
                    if i == j:
                        continue
                    if (shifted & sums[j]) != 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
                sums = new_sums
                if len(S) == 60:
                    break
        return S

    # Strategy 2: AP Search (from current program)
    def ap_search():
        best_k = 0
        best_a = 1
        for k in range(60, 0, -1):
            a_max = 257 - k
            if a_max < 1:
                continue
            for a in range(1, a_max + 1):
                valid = True
                for i in range(1, k):
                    max_i_sum = i * (a + k - 1) - i * (i - 1) // 2
                    min_ip1_sum = (i + 1) * a + i * (i + 1) // 2
                    if max_i_sum >= min_ip1_sum:
                        valid = False
                        break
                if valid:
                    best_k = k
                    best_a = a
                    break
            if best_k > 0:
                break
        if best_k == 0:
            return []
        return list(range(best_a, best_a + best_k))

    # Run both
    set1 = greedy_top_down()
    if time.time() < deadline - 10:
        set2 = ap_search()
    else:
        set2 = []
        
    if len(set1) > len(set2):
        result = set1
    else:
        result = set2
        
    # Verify result
    if not result:
        return list(range(1, 10)) # Fallback
        
    # Check validity of result
    # This is slow, so only do it if we have time
    if time.time() < deadline - 5:
        sums = [set() for _ in range(len(result)+1)]
        sums[0].add(0)
        for x in result:
            for i in range(len(result), 0, -1):
                for s in sums[i-1]:
                    sums[i].add(s+x)
        # Check disjointness
        valid = True
        for i in range(len(result)+1):
            for j in range(i+1, len(result)+1):
                if sums[i] & sums[j]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to AP if greedy failed verification
            if len(set2) > 0:
                result = set2
            else:
                result = list(range(1, 10))
                
    return result

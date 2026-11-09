import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)
    
    # Known good: score 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    def search_for_B(A, max_val, current_best):
        a1, a2, a3 = A[0], A[1], A[2]
        best_B = None
        best_s = current_best
        
        # Iterate b1, b2, b3 with b1 < b2 < b3
        for b1 in range(1, max_val + 1):
            s11 = a1 + b1
            s21 = a2 + b1
            s31 = a3 + b1
            # These three must be pairwise coprime
            if math.gcd(s11, s21) != 1:
                continue
            if math.gcd(s11, s31) != 1:
                continue
            if math.gcd(s21, s31) != 1:
                continue
            
            for b2 in range(b1 + 1, max_val + 1):
                s12 = a1 + b2
                s22 = a2 + b2
                s32 = a3 + b2
                
                # Check against column 1
                valid = True
                for ns in (s12, s22, s32):
                    for ps in (s11, s21, s31):
                        if ns == ps:
                            if ps != 1:
                                valid = False
                                break
                        elif math.gcd(ns, ps) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check among themselves
                if math.gcd(s12, s22) != 1 or math.gcd(s12, s32) != 1 or math.gcd(s22, s32) != 1:
                    continue
                
                for b3 in range(b2 + 1, max_val + 1):
                    s13 = a1 + b3
                    s23 = a2 + b3
                    s33 = a3 + b3
                    
                    # Check against all previous
                    ok = True
                    for ns in (s13, s23, s33):
                        for ps in (s11, s21, s31, s12, s22, s32):
                            if ns == ps:
                                if ps != 1:
                                    ok = False
                                    break
                            elif math.gcd(ns, ps) != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    # Check among new
                    if math.gcd(s13, s23) != 1 or math.gcd(s13, s33) != 1 or math.gcd(s23, s33) != 1:
                        continue
                    
                    all_sums = set([s11, s21, s31, s12, s22, s32, s13, s23, s33])
                    score = len(all_sums)
                    if score > best_s:
                        best_s = score
                        best_B = [b1, b2, b3]
                        if best_s >= 9:
                            return best_B, best_s
        return best_B, best_s
    
    # Strategy: Try A values that are small and well-separated
    # to give B more room to create distinct coprime sums
    
    # Phase 1: Very small A, wide B range
    for a1 in range(1, 10):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 15):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 25):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 40, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    if time.time() > deadline:
        return [list(best_A), list(best_B)]
    
    # Phase 2: Small-medium A
    for a1 in range(1, 20):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 30):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 45):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 55, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    if time.time() > deadline:
        return [list(best_A), list(best_B)]
    
    # Phase 3: Medium A
    for a1 in range(1, 35):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 70):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 80, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    if time.time() > deadline:
        return [list(best_A), list(best_B)]
    
    # Phase 4: Larger A
    for a1 in range(1, 50):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 70):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 95):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 120, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    return [list(best_A), list(best_B)]

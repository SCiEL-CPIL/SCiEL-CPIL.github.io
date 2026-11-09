import time
import random

def has_3_ap(S):
    """Check if set S contains a 3-term arithmetic progression."""
    S = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            c = 2 * b - a
            if c in S and c != a:
                return True
    return False

def construct():
    # The goal is to minimize the largest Sidon subset size.
    # A subset of size 3 is Sidon iff it is not a 3-term arithmetic progression.
    # If we choose a set of 12 integers with no 3-term AP,
    # then no subset of size 3 is Sidon.
    # Since any subset of size 2 is always Sidon, the max Sidon size will be 2.
    # This is the theoretical minimum for any set of >= 2 distinct integers.
    
    # Construct a set of 12 integers in [0, 128] with no 3-term AP.
    # Numbers with only 0s and 1s in base 3 form such a set.
    base3_set = []
    for i in range(1024): # 3^6 = 729, 3^7=2187. We need up to 128.
        # Generate number with digits 0 or 1 in base 3
        val = 0
        power = 1
        temp = i
        valid = True
        for _ in range(7): # 3^6 = 729 > 128, so 7 digits enough
            digit = temp % 2
            temp //= 2
            val += digit * power
            power *= 3
            if power > 128:
                break
        if val <= 128:
            base3_set.append(val)
        if power > 128 and temp == 0:
            break
            
    # Remove duplicates and sort
    base3_set = sorted(list(set(base3_set)))
    
    # We need exactly 12. The set generated above should have at least 12.
    # Count:
    # 0, 1
    # 3, 4
    # 9, 10, 12, 13
    # 27, 28, 30, 31
    # 81, 82, 84, 85, 90, 91, 93, 94
    # 1 + 1 + 2 + 4 + 8 = 16 numbers.
    
    if len(base3_set) < 12:
        # Fallback to random search if generation failed (unlikely)
        while True:
            cand = random.sample(range(129), 12)
            if not has_3_ap(cand):
                return cand
    else:
        return base3_set[:12]

import time
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    def check_valid(S):
        """Check if S is valid: for every a<b<=c in S, a does not divide b+c."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c in S_list[j:]:
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, a):
        """Check if adding a to S maintains validity."""
        if a in S:
            return True
        S_list = sorted(S)
        larger = [x for x in S_list if x > a]
        for i, b in enumerate(larger):
            for c in larger[i:]:
                if (b + c) % a == 0:
                    return False
        return True

    # Start with the known valid set [171, 256]
    base = set(range(171, 257))
    
    best_set = set(base)
    best_size = len(best_set)

    # Strategy 1: Greedy ascending
    S1 = set(base)
    for a in range(1, 171):
        if time.time() > deadline - 5:
            break
        if can_add(S1, a):
            S1.add(a)
    if len(S1) > best_size:
        best_set = set(S1)
        best_size = len(S1)

    # Strategy 2: Greedy descending
    S2 = set(base)
    for a in range(170, 0, -1):
        if time.time() > deadline - 5:
            break
        if can_add(S2, a):
            S2.add(a)
    if len(S2) > best_size:
        best_set = set(S2)
        best_size = len(S2)

    # Strategy 3: Greedy ascending then descending
    S3 = set(base)
    for a in range(1, 171):
        if time.time() > deadline - 5:
            break
        if can_add(S3, a):
            S3.add(a)
    for a in range(170, 0, -1):
        if time.time() > deadline - 5:
            break
        if can_add(S3, a):
            S3.add(a)
    if len(S3) > best_size:
        best_set = set(S3)
        best_size = len(S3)

    # Strategy 4: Try different base thresholds
    for threshold in range(128, 172):
        if time.time() > deadline - 10:
            break
        S4 = set(range(threshold, 257))
        # Verify base is valid
        if not check_valid(S4):
            continue
        # Greedy ascending
        for a in range(1, threshold):
            if time.time() > deadline - 5:
                break
            if can_add(S4, a):
                S4.add(a)
        if len(S4) > best_size:
            best_set = set(S4)
            best_size = len(S4)

    # Strategy 5: Local search - try removing elements to allow adding more
    if time.time() < deadline - 15:
        S5 = set(best_set)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S5_list = sorted(S5)
            for i, a in enumerate(S5_list):
                if time.time() > deadline - 10:
                    break
                # Try removing a and see if we can add more
                S_trial = set(S5)
                S_trial.discard(a)
                added = 0
                for b in range(1, 257):
                    if b not in S_trial and b != a:
                        if can_add(S_trial, b):
                            S_trial.add(b)
                            added += 1
                if len(S_trial) > len(S5):
                    S5 = S_trial
                    improved = True
                    break
        if len(S5) > best_size:
            best_set = set(S5)
            best_size = len(S5)

    # Final verification
    S_final = sorted(best_set)
    valid = True
    for i, a in enumerate(S_final):
        for j in range(i + 1, len(S_final)):
            b = S_final[j]
            for c in S_final[j:]:
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to known valid set
        return list(range(171, 257))

    return S_final

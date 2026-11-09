import random
import time
import sys

# Precompute all 4-term arithmetic progressions in [1, 512]
N = 512
APs = []
# ap[x] will store indices of APs containing x
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        idx = len(APs)
        APs.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

def is_valid_set(S_set):
    """Check if S_set contains any 4-AP."""
    for ap in APs:
        if all(x in S_set for x in ap):
            return False
    return True

def can_add(S_set, x):
    """Check if adding x to S_set keeps it 4-AP free."""
    for idx in aps_containing[x]:
        ap = APs[idx]
        # Check if the other 3 elements are in S_set
        count = 0
        for y in ap:
            if y != x and y in S_set:
                count += 1
        if count == 3:
            return False
    return True

def find_conflicts(S_set, x):
    """Return list of AP indices that would be completed if x is added."""
    conflicts = []
    for idx in aps_containing[x]:
        ap = APs[idx]
        count = 0
        for y in ap:
            if y != x and y in S_set:
                count += 1
        if count == 3:
            conflicts.append(idx)
    return conflicts

def check_swap_validity(S_set, x, y):
    """
    Check if (S_set - {y}) U {x} is valid.
    S_set is known to be valid.
    We only need to check APs containing x or y.
    APs containing y but not x: 
       In S_set, they were not fully contained. 
       Removing y ensures they are not fully contained in the new set.
       Adding x doesn't affect them if x is not in them.
       If x IS in them, they are handled in "APs containing x".
    So we only need to check APs containing x.
    """
    new_S = S_set - {y}
    new_S.add(x)
    
    # Check APs containing x
    for idx in aps_containing[x]:
        ap = APs[idx]
        if all(z in new_S for z in ap):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    # Initial construction: Randomized Greedy
    S = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(S, x):
            S.add(x)
    
    # Local Search
    best_size = len(S)
    no_improvement_count = 0
    max_no_improvement = 500 # Restart if stuck
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Restart with a new random greedy set
            S = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(S, x):
                    S.add(x)
            no_improvement_count = 0
            continue
            
        # Pick a random x not in S
        complement = [x for x in range(1, N + 1) if x not in S]
        if not complement:
            break
            
        x = random.choice(complement)
        
        if can_add(S, x):
            S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            conflicts = find_conflicts(S, x)
            if conflicts:
                # Pick a random conflict AP
                ap_idx = random.choice(conflicts)
                ap = APs[ap_idx]
                
                # Pick a random y from the AP (excluding x) to remove
                # y must be in S
                possible_y = [z for z in ap if z != x and z in S]
                if possible_y:
                    y = random.choice(possible_y)
                    
                    # Check if swap is valid
                    if check_swap_validity(S, x, y):
                        # Perform swap
                        S.remove(y)
                        S.add(x)
                        # Size is unchanged, but structure changed
                        no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
    return sorted(S)

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")

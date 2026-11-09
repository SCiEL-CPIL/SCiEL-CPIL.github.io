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

def fill_greedy(S_set, candidates):
    """Try to add as many elements from candidates as possible."""
    added = 0
    for x in candidates:
        if x not in S_set and can_add(S_set, x):
            S_set.add(x)
            added += 1
    return added

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    
    best_S = set()
    best_size = 0
    
    def greedy_construct():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        return S
    
    # Multiple greedy restarts to find a good initial set
    for _ in range(10):
        if time.time() > deadline:
            break
        S = greedy_construct()
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    S = set(best_S)
    
    # Local Search with fill phase
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            # Restart with a new random greedy set
            S = greedy_construct()
            no_improvement_count = 0
            # Try to fill after restart
            complement = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(complement)
            fill_greedy(S, complement)
            continue
            
        # Pick a random x not in S
        complement = [x for x in range(1, N + 1) if x not in S]
        if not complement:
            break
            
        x = random.choice(complement)
        
        if can_add(S, x):
            S.add(x)
            # Try to fill more after adding
            remaining = [y for y in complement if y != x and y not in S]
            random.shuffle(remaining)
            fill_greedy(S, remaining)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
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
                        
                        # After swap, try to fill with other elements
                        # that might now be addable
                        complement_now = [z for z in range(1, N + 1) if z not in S]
                        random.shuffle(complement_now)
                        fill_greedy(S, complement_now)
                        
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
    return sorted(best_S)

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

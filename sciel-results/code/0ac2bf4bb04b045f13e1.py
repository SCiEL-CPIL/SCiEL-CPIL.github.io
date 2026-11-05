import time
import random
import sys

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s margin

# Precomputation
# 3-APs: list of (a, b, c)
# For fast gain calculation, for each x, store the other two elements for each 3-AP involving x.
# We can just iterate d.

# 4-APs: list of (a, b, c, d)
four_aps = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            four_aps.append((a, b, c, e))

# For each element, list of 4-AP indices it belongs to
elem_to_4aps = {x: [] for x in range(1, N + 1)}
for i, ap in enumerate(four_aps):
    for x in ap:
        elem_to_4aps[x].append(i)

def count_3aps(S):
    """Count 3-APs in set S."""
    in_set = [False] * (N + 1)
    for x in S:
        in_set[x] = True
    count = 0
    for b in range(1, N + 1):
        if not in_set[b]:
            continue
        # Count pairs (a, c) in S such that a + c = 2b
        # a = 2b - c
        # c ranges from max(1, 2b-N) to min(N, 2b-1)
        # a must be in S and a != b
        # Actually, just iterate c
        for c in range(1, N + 1):
            if c == b:
                continue
            a = 2 * b - c
            if 1 <= a <= N and a != b and in_set[a] and in_set[c]:
                # This counts each 3-AP twice? No.
                # A 3-AP is (a, b, c) with a < b < c.
                # If we fix b, and find a, c, we have one 3-AP.
                # But we iterate c from 1 to N.
                # If a < b < c, then a = 2b - c < b implies c > b.
                # If a > b > c, then a = 2b - c > b implies c < b.
                # So for each b, each valid pair (a,c) corresponds to one 3-AP.
                # Wait, if we count all c, we count both (a,c) and (c,a)?
                # No, a is determined by c.
                # If c > b, then a < b. The 3-AP is (a,b,c).
                # If c < b, then a > b. The 3-AP is (c,b,a).
                # So each 3-AP is counted exactly once for its middle term b.
                count += 1
    return count

def has_4ap(S):
    """Check if S contains any 4-AP."""
    in_set = [False] * (N + 1)
    for x in S:
        in_set[x] = True
    for a, b, c, d in four_aps:
        if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
            return True
    return False

def get_gain(x, in_set):
    """Calculate the increase in 3-AP count if x is added to the set represented by in_set."""
    gain = 0
    # 1. x is the middle term
    # a + c = 2x, a, c in S, a != x, c != x
    for c in range(1, N + 1):
        if c == x:
            continue
        a = 2 * x - c
        if 1 <= a <= N and a != x and in_set[a] and in_set[c]:
            gain += 1
    # 2. x is the first term (a)
    # x, b, c => x + c = 2b => c = 2b - x
    # b, c in S
    for b in range(1, N + 1):
        if b == x:
            continue
        c = 2 * b - x
        if 1 <= c <= N and c != x and in_set[b] and in_set[c]:
            gain += 1
    # 3. x is the last term (c)
    # a, b, x => a + x = 2b => a = 2b - x
    # a, b in S
    for b in range(1, N + 1):
        if b == x:
            continue
        a = 2 * b - x
        if 1 <= a <= N and a != x and in_set[a] and in_set[b]:
            gain += 1
    return gain

def get_loss(x, in_set):
    """Calculate the decrease in 3-AP count if x is removed from the set represented by in_set."""
    # This is the same as the gain if x were added to S \ {x}
    # But in_set currently contains x.
    # We need to count 3-APs involving x where the other two elements are in S.
    # This is exactly the same logic as get_gain, but we must be careful.
    # get_gain counts pairs (a,c) in S such that a+c=2x etc.
    # If we remove x, we lose all 3-APs that include x.
    # The number of such 3-APs is the number of pairs (a,c) in S\{x} such that {a,c,x} is a 3-AP.
    # This is identical to the calculation in get_gain, assuming in_set represents S.
    # However, in get_gain, we check `in_set[a] and in_set[c]`.
    # If we are removing x, the other elements a,c are still in S.
    # So the count is the same.
    loss = 0
    # 1. x is the middle term
    for c in range(1, N + 1):
        if c == x:
            continue
        a = 2 * x - c
        if 1 <= a <= N and a != x and in_set[a] and in_set[c]:
            loss += 1
    # 2. x is the first term
    for b in range(1, N + 1):
        if b == x:
            continue
        c = 2 * b - x
        if 1 <= c <= N and c != x and in_set[b] and in_set[c]:
            loss += 1
    # 3. x is the last term
    for b in range(1, N + 1):
        if b == x:
            continue
        a = 2 * b - x
        if 1 <= a <= N and a != x and in_set[a] and in_set[b]:
            loss += 1
    return loss

def check_4ap_violation(x, in_set):
    """Check if adding x to S (represented by in_set) creates a 4-AP.
    Assumes x is NOT in in_set."""
    for idx in elem_to_4aps[x]:
        a, b, c, d = four_aps[idx]
        # Check if the other 3 elements are in S
        others = [y for y in (a, b, c, d) if y != x]
        if all(in_set[y] for y in others):
            return True
    return False

def check_4ap_violation_remove(x, in_set):
    """Check if removing x from S (represented by in_set) breaks a 4-AP.
    This is always safe, so we don't need to check for violations, 
    but we might want to know if it breaks an existing violation? 
    No, we maintain the invariant that S has no 4-APs.
    So removing x cannot create a 4-AP.
    """
    return False

def construct():
    start_time = time.time()
    
    # Initialize with a greedy set
    S = set()
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        if not check_4ap_violation(x, in_set):
            S.add(x)
            in_set[x] = True
    
    current_score = count_3aps(S)
    best_score = current_score
    best_S = set(S)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Simulated Annealing
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
            
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            unselected = [x for x in range(1, N + 1) if not in_set[x]]
            if not unselected:
                T *= cooling
                continue
            x = random.choice(unselected)
            if check_4ap_violation(x, in_set):
                T *= cooling
                continue
            
            gain = get_gain(x, in_set)
            delta = gain
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                S.add(x)
                in_set[x] = True
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                    
        elif move_type == 'remove':
            if not S:
                T *= cooling
                continue
            x = random.choice(list(S))
            loss = get_loss(x, in_set)
            delta = -loss
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                S.remove(x)
                in_set[x] = False
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                    
        else: # swap
            unselected = [x for x in range(1, N + 1) if not in_set[x]]
            if not unselected or not S:
                T *= cooling
                continue
            x_add = random.choice(unselected)
            x_rem = random.choice(list(S))
            
            # Check if adding x_add creates a 4-AP (considering x_rem is removed)
            # We need to check if x_add forms a 4-AP with elements in S \ {x_rem}
            # Temporarily mark x_rem as not in set for the check?
            # Or just check the 4-APs involving x_add and see if the other 3 are in S and not x_rem.
            
            valid_swap = True
            for idx in elem_to_4aps[x_add]:
                a, b, c, d = four_aps[idx]
                others = [y for y in (a, b, c, d) if y != x_add]
                # All others must be in S, and none of them can be x_rem?
                # Actually, if one of the others is x_rem, then since x_rem is removed, 
                # the 4-AP is not formed.
                # So we need: all(y in S and y != x_rem for y in others)
                for y in others:
                    if not in_set[y] or y == x_rem:
                        valid_swap = False
                        break
                if not valid_swap:
                    break
            
            if not valid_swap:
                T *= cooling
                continue
            
            # Calculate delta
            # Remove x_rem: lose get_loss(x_rem, in_set)
            # Add x_add: gain get_gain(x_add, in_set_updated)
            # But get_gain depends on the set.
            # Let's compute carefully.
            
            # Loss from removing x_rem
            # The 3-APs lost are those involving x_rem and other elements in S.
            # But if a 3-AP involves both x_rem and x_add, it's handled in both?
            # No, x_add is not in S yet.
            # So loss is just 3-APs with x_rem and S\{x_rem}.
            # This is get_loss(x_rem, in_set).
            
            # Gain from adding x_add
            # The 3-APs gained are those involving x_add and S\{x_rem}.
            # We need to calculate this.
            # We can't just use get_gain(x_add, in_set) because in_set contains x_rem.
            # If a 3-AP involves x_add, x_rem, and y, it will be counted in get_gain(x_add, in_set)
            # but x_rem is being removed, so that 3-AP should not be counted as a gain.
            
            # Let's compute gain manually or adjust.
            # Gain = (3-APs with x_add and S) - (3-APs with x_add and x_rem and S)
            
            base_gain = get_gain(x_add, in_set)
            # Subtract 3-APs that involve x_add and x_rem
            # A 3-AP involving x_add and x_rem:
            # 1. x_add, x_rem, y => x_add + y = 2*x_rem => y = 2*x_rem - x_add. Check if y in S and y != x_rem, x_add.
            # 2. x_rem, x_add, y => x_rem + y = 2*x_add => y = 2*x_add - x_rem. Check if y in S and y != x_rem, x_add.
            # 3. x_add, y, x_rem => x_add + x_rem = 2*y => y = (x_add + x_rem)/2. Check if y in S and y != x_rem, x_add and (x_add+x_rem) even.
            
            correction = 0
            # Case 1: x_add is first, x_rem is second
            y = 2 * x_rem - x_add
            if 1 <= y <= N and y != x_add and y != x_rem and in_set[y]:
                correction += 1
            # Case 2: x_rem is first, x_add is second
            y = 2 * x_add - x_rem
            if 1 <= y <= N and y != x_add and y != x_rem and in_set[y]:
                correction += 1
            # Case 3: x_add and x_rem are ends
            if (x_add + x_rem) % 2 == 0:
                y = (x_add + x_rem) // 2
                if 1 <= y <= N and y != x_add and y != x_rem and in_set[y]:
                    correction += 1
            
            # Wait, get_gain(x_add, in_set) counts 3-APs where x_add is involved and the other two are in S.
            # If one of the other two is x_rem, that 3-AP is not formed after removing x_rem.
            # So we subtract the number of such 3-APs.
            # The logic above counts the number of 3-APs involving x_add and x_rem.
            
            new_gain = base_gain - correction
            delta = new_gain - get_loss(x_rem, in_set)
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                S.remove(x_rem)
                in_set[x_rem] = False
                S.add(x_add)
                in_set[x_add] = True
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
        
        T *= cooling
        
    return sorted(best_S)

def validate(result):
    sel = set(result)
    for (a, b, c, e) in four_aps:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps(result)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")

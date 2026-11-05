import time
import random
import sys
import math

N = 256
TIME_LIMIT = 85.0 # Safety margin against 90s kill

# Precomputation
three_aps = []
element_to_3aps = {i: [] for i in range(1, N+1)}
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        ap = (a, b, c)
        three_aps.append(ap)
        for elem in ap:
            element_to_3aps[elem].append(ap)

four_aps = []
element_to_4aps = {i: [] for i in range(1, N+1)}
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        ap = (a, b, c, e)
        four_aps.append(ap)
        for elem in ap:
            element_to_4aps[elem].append(ap)

# For efficient delta calculation, we need to know the state of neighbors.
# We will maintain a boolean array `present` of size N+1.
# We will maintain `three_count` and `four_count`? 
# No, for smooth surrogate, we need the actual counts of *completed* 3-APs and 4-APs.
# Let's maintain:
# num_3aps: integer count of 3-APs fully in A
# num_4aps: integer count of 4-APs fully in A

def greedy_initial(seed_val=0):
    random.seed(seed_val)
    A = [False] * (N + 1)
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        # Check if adding x creates a 4-AP
        valid = True
        for ap in element_to_4aps[x]:
            others = [elem for elem in ap if elem != x]
            if all(A[o] for o in others):
                valid = False
                break
        if valid:
            A[x] = True
    return A

def count_3aps(A):
    count = 0
    for a, b, c in three_aps:
        if A[a] and A[b] and A[c]:
            count += 1
    return count

def count_4aps(A):
    count = 0
    for a, b, c, d in four_aps:
        if A[a] and A[b] and A[c] and A[d]:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # We will run Simulated Annealing
    # State: A (list of bools), num_3, num_4
    # Objective: Maximize num_3 - M * max(0, num_4)
    # Smooth surrogate idea: 
    # Instead of hard penalty, we can use a soft penalty if we want to explore "almost" valid states,
    # but since we need a valid result, we should stay in valid space or use strong penalty.
    # The directive suggests "smooth surrogate for nonsmooth min/max".
    # The count of 3-APs is integer (nonsmooth). 
    # We can't easily smooth the count itself without losing meaning.
    # However, we can interpret "smooth surrogate" as using a probabilistic acceptance criterion 
    # that smooths the landscape, or using a penalty term that is continuous in the state space 
    # if we viewed variables as continuous (but they are binary).
    # Let's stick to standard SA on the discrete space but with a robust schedule.
    # The "smoothness" comes from the fact that we allow moves that temporarily decrease score
    # (if penalty is high, invalid states are heavily penalized, so we mostly stay valid, 
    # but we can jump over local minima in the valid subspace).
    
    # To be safe and fast, we will use a "Valid-Only" SA where we only accept moves that keep 4-AP count 0,
    # OR we allow temporary violations with a huge penalty.
    # Given the density, staying valid is hard with random swaps.
    # Let's use the penalty method: Score = num_3 - 1000 * num_4.
    # This effectively makes invalid states very bad, but allows the search to traverse them if it helps escape a local max in valid space.
    
    best_set = None
    best_score = -1
    
    # Multiple restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        A = greedy_initial(r)
        num_3 = count_3aps(A)
        num_4 = count_4aps(A) # Should be 0 for greedy
        
        current_score = num_3 - 1000 * num_4
        best_local_score = current_score
        best_local_A = A[:]
        
        # SA Parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        # Precompute potential moves? No, just pick random x in A, y not in A.
        
        while time.time() < deadline - 2:
            # Pick a random swap
            # To speed up, we can pick x from A and y from ~A
            # We need lists of indices for fast sampling
            # Rebuilding lists every step is O(N). N=256, so it's fine.
            in_A = [i for i in range(1, N+1) if A[i]]
            not_in_A = [i for i in range(1, N+1) if not A[i]]
            
            if not in_A or not not_in_A:
                break
                
            x = random.choice(in_A)
            y = random.choice(not_in_A)
            
            # Calculate delta for removing x and adding y
            # This is O(deg(x) + deg(y))
            # deg is roughly O(N) for 3-APs? 
            # Number of 3-APs containing x: sum_{d} (1 if x-d>=1 and x+2d<=N etc)
            # It's small, maybe < 20.
            
            # Delta num_3:
            # Remove x: lose 3-APs where x is present and other 2 are in A.
            # Add y: gain 3-APs where y is present and other 2 are in A (after x removed).
            
            delta_3 = 0
            
            # Effect of removing x
            for ap in element_to_3aps[x]:
                a, b, c = ap
                if a == x:
                    if A[b] and A[c]:
                        delta_3 -= 1
                elif b == x:
                    if A[a] and A[c]:
                        delta_3 -= 1
                elif c == x:
                    if A[a] and A[b]:
                        delta_3 -= 1
            
            # Effect of adding y
            # Note: A[x] is still True in our check? 
            # We must evaluate delta based on the state AFTER removing x and BEFORE adding y? 
            # Or simultaneously? 
            # If x and y are in the same 3-AP, the interaction matters.
            # Let's do it sequentially in logic:
            # State S. Remove x -> S'. Add y -> S''.
            # Delta = Score(S'') - Score(S).
            
            # We already calculated loss from removing x from S.
            # Now calculate gain from adding y to S' (where x is False).
            
            for ap in element_to_3aps[y]:
                a, b, c = ap
                # Check if other 2 are in S'
                # S' is A with x=False, y=False (since y not in A yet)
                
                # Helper to check if elem is in S'
                # is_in_S_prime(e): 
                # if e == x: False
                # if e == y: False (y not added yet in this check context? No, we are checking prerequisites for adding y)
                # Actually, for a 3-AP to be completed by adding y, the other two must be in S'.
                
                # Let's just check the current A, but adjust for x.
                # If the other two are a, b.
                # If a == x, then a is not in S'.
                # If a == y, impossible (y not in ap other than the slot y is in? No, y is one of a,b,c).
                
                # Let's iterate carefully.
                if a == y:
                    # Need b, c in S'
                    b_ok = (b != x) and A[b]
                    c_ok = (c != x) and A[c]
                    if b_ok and c_ok:
                        delta_3 += 1
                elif b == y:
                    # Need a, c in S'
                    a_ok = (a != x) and A[a]
                    c_ok = (c != x) and A[c]
                    if a_ok and c_ok:
                        delta_3 += 1
                elif c == y:
                    # Need a, b in S'
                    a_ok = (a != x) and A[a]
                    b_ok = (b != x) and A[b]
                    if a_ok and b_ok:
                        delta_3 += 1

            # Delta num_4:
            delta_4 = 0
            
            # Remove x
            for ap in element_to_4aps[x]:
                a, b, c, d = ap
                if a == x:
                    if A[b] and A[c] and A[d]:
                        delta_4 -= 1
                elif b == x:
                    if A[a] and A[c] and A[d]:
                        delta_4 -= 1
                elif c == x:
                    if A[a] and A[b] and A[d]:
                        delta_4 -= 1
                elif d == x:
                    if A[a] and A[b] and A[c]:
                        delta_4 -= 1
            
            # Add y
            for ap in element_to_4aps[y]:
                a, b, c, d = ap
                # Need other 3 in S'
                # S' has x=False.
                
                if a == y:
                    # Need b,c,d in S'
                    b_ok = (b != x) and A[b]
                    c_ok = (c != x) and A[c]
                    d_ok = (d != x) and A[d]
                    if b_ok and c_ok and d_ok:
                        delta_4 += 1
                elif b == y:
                    a_ok = (a != x) and A[a]
                    c_ok = (c != x) and A[c]
                    d_ok = (d != x) and A[d]
                    if a_ok and c_ok and d_ok:
                        delta_4 += 1
                elif c == y:
                    a_ok = (a != x) and A[a]
                    b_ok = (b != x) and A[b]
                    d_ok = (d != x) and A[d]
                    if a_ok and b_ok and d_ok:
                        delta_4 += 1
                elif d == y:
                    a_ok = (a != x) and A[a]
                    b_ok = (b != x) and A[b]
                    c_ok = (c != x) and A[c]
                    if a_ok and b_ok and c_ok:
                        delta_4 += 1

            new_num_3 = num_3 + delta_3
            new_num_4 = num_4 + delta_4
            
            new_score = new_num_3 - 1000 * new_num_4
            delta_score = new_score - current_score
            
            # Acceptance
            if delta_score >= 0:
                accept = True
            else:
                # Metropolis criterion
                # Avoid overflow/underflow
                if T < 1e-10:
                    accept = False
                else:
                    # exp(delta / T)
                    # If delta is very negative and T small, exp is 0.
                    try:
                        prob = math.exp(delta_score / T)
                    except OverflowError:
                        prob = 0.0
                    except ValueError:
                        prob = 0.0
                    
                    accept = random.random() < prob
            
            if accept:
                # Perform swap
                A[x] = False
                A[y] = True
                num_3 = new_num_3
                num_4 = new_num_4
                current_score = new_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_A = A[:]
                    
            T *= cooling_rate
            
        # End of SA run
        # Update global best
        # Ensure best_local_A is valid? 
        # If penalty is 1000, and max 3-APs is ~16k, a single 4-AP costs 1000.
        # So it's very unlikely to end in an invalid state unless no valid state is reachable.
        # But we should check.
        final_num_3 = count_3aps(best_local_A)
        final_num_4 = count_4aps(best_local_A)
        
        if final_num_4 == 0:
            if final_num_3 > best_score:
                best_score = final_num_3
                best_set = best_local_A[:]
        else:
            # Fallback: try to repair or just ignore if no valid found yet?
            # If best_set is None, we need a valid set.
            # Greedy initial was valid.
            if best_set is None:
                # Use the initial valid set from this restart
                init_A = greedy_initial(r)
                init_score = count_3aps(init_A)
                if init_score > best_score:
                    best_score = init_score
                    best_set = init_A

    # Final safety: if best_set is still None (shouldn't happen), return greedy
    if best_set is None:
        best_set = greedy_initial(0)
        best_score = count_3aps(best_set)
        
    return [i for i in range(1, N+1) if best_set[i]]

# Helper for debugging or if needed
def is_4ap_free(A):
    A_set = set(A)
    for a, b, c, d in four_aps:
        if a in A_set and b in A_set and c in A_set and d in A_set:
            return False
    return True

def count_3aps_list(A_list):
    A_bool = [False] * (N+1)
    for i in A_list:
        A_bool[i] = True
    return count_3aps(A_bool)

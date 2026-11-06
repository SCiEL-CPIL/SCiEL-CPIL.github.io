import random
import math
import time

def construct():
    N = 128
    # 1. Precompute bad triplets
    # bad_triplets: set of frozensets or just list of (a,b,c) with a<b<c
    # conflicts_with[k]: list of pairs (i, j) such that {i, j, k} is a bad triplet
    # Note: i, j are distinct and not k.
    
    # We need to check lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Precompute lcms? 128*128/2 pairs.
    # Or just compute on the fly during precomputation.
    
    lcms = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            lcms[i][j] = math.lcm(i, j)
    
    conflicts_with = [[] for _ in range(N+1)]
    bad_triplets_count = 0
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lcm_ab = lcms[a][b]
            for c in range(b+1, N+1):
                # Check if lcm(a,c) == lcm_ab and lcm(b,c) == lcm_ab
                # Optimization: c must be a multiple of lcm_ab? No.
                # lcm(a,c) = L implies c | L? No, c is a factor of L?
                # If lcm(a,c) = L, then c divides L.
                # So c must be a divisor of lcm_ab.
                # Let's check if c divides lcm_ab.
                if lcm_ab % c != 0:
                    continue
                # If c divides lcm_ab, then lcm(a,c) is a divisor of lcm_ab.
                # It equals lcm_ab if and only if ...
                # Actually, just compute.
                if lcms[a][c] == lcm_ab and lcms[b][c] == lcm_ab:
                    # Found bad triplet (a, b, c)
                    bad_triplets_count += 1
                    # Add to conflicts_with for a, b, c
                    conflicts_with[a].append((b, c))
                    conflicts_with[b].append((a, c))
                    conflicts_with[c].append((a, b))
                    
    # 2. Simulated Annealing
    # State: in_set (list of bools)
    # Objective: size - penalty * bad_count
    # We want to maximize this.
    
    # Initialize with greedy
    in_set = [False] * (N+1)
    # Greedy: add i if no conflict with existing pairs
    # For efficiency in greedy, we can just use the conflict check.
    # But greedy is O(N^3) worst case if done naively.
    # Let's just use a random valid start or simple greedy.
    
    # Simple Greedy:
    for i in range(1, N+1):
        # Check if adding i creates a bad triplet
        # A bad triplet involving i is (j, k, i) where j, k in S.
        # So we check conflicts_with[i].
        # If any pair (j, k) in conflicts_with[i] is fully in S, then adding i is bad.
        is_bad = False
        for j, k in conflicts_with[i]:
            if in_set[j] and in_set[k]:
                is_bad = True
                break
        if not is_bad:
            in_set[i] = True
            
    best_in_set = in_set[:]
    best_score = sum(best_in_set) - 50 * 0 # Initially valid, so 0 bad triplets
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.995
    steps = 20000
    penalty = 50.0
    
    # Helper to count bad triplets in current set
    def count_bad_triplets(state):
        count = 0
        # Iterate through all bad triplets? No, we don't have a list of all bad triplets indexed easily for fast count.
        # We have conflicts_with.
        # If we sum len(conflicts_with[i]) for i in S, each bad triplet is counted 3 times (once for each vertex).
        # So bad_count = sum( ... ) / 3.
        # Wait, conflicts_with[i] contains pairs (j,k).
        # If (j,k) is in S, then (i,j,k) is a bad triplet.
        # So for a fixed state S, the number of bad triplets is:
        # Sum over i in S of (number of pairs (j,k) in conflicts_with[i] such that j in S and k in S)
        # Each bad triplet {a,b,c} is counted for a (pair b,c), for b (pair a,c), for c (pair a,b).
        # So total sum is 3 * bad_count.
        total = 0
        for i in range(1, N+1):
            if state[i]:
                for j, k in conflicts_with[i]:
                    if state[j] and state[k]:
                        total += 1
        return total // 3

    # Helper to calculate delta when flipping k
    def get_delta(k, state):
        # state is list of bools
        # delta = (new_size - old_size) - penalty * (new_bad - old_bad)
        # new_size - old_size is +1 if adding, -1 if removing
        if state[k]:
            # Removing k
            size_delta = -1
            # Bad triplets removed: those involving k where other two are in S
            # These are exactly the pairs (j, l) in conflicts_with[k] such that state[j] and state[l]
            bad_removed = 0
            for j, l in conflicts_with[k]:
                if state[j] and state[l]:
                    bad_removed += 1
            # Note: bad_removed is the count of bad triplets that disappear.
            # So new_bad = old_bad - bad_removed
            # delta = -1 - penalty * (-bad_removed) = -1 + penalty * bad_removed
            return -1 + penalty * bad_removed
        else:
            # Adding k
            size_delta = 1
            # Bad triplets added: those involving k where other two are in S
            bad_added = 0
            for j, l in conflicts_with[k]:
                if state[j] and state[l]:
                    bad_added += 1
            # delta = 1 - penalty * bad_added
            return 1 - penalty * bad_added

    start_time = time.time()
    current_in_set = in_set[:]
    current_bad = count_bad_triplets(current_in_set)
    current_score = sum(current_in_set) - penalty * current_bad
    
    for step in range(steps):
        if time.time() - start_time > 80:
            break
            
        k = random.randint(1, N)
        delta = get_delta(k, current_in_set)
        
        # Update bad count for acceptance probability? 
        # We need new_bad to calculate new_score?
        # new_score = current_score + delta
        # We don't strictly need new_bad for acceptance, just delta.
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept move
            current_in_set[k] = not current_in_set[k]
            # Update current_bad? 
            # If we don't update current_bad, we can't use it for next delta?
            # get_delta uses state to count. It doesn't use current_bad variable.
            # So we don't need to maintain current_bad separately for the algorithm logic,
            # except for calculating current_score if we wanted to.
            # But we do need to update best_score.
            
            new_score = current_score + delta
            # Check if this is a valid improvement (0 bad triplets)
            # We can check validity occasionally or at the end.
            # To be safe, let's track if it's valid.
            # Calculating count_bad_triplets is O(N * deg). 
            # deg is small.
            # Let's just check if new_score is high and bad count is 0.
            
            # Optimization: Only check bad count if delta suggests it might be 0 or if score is high.
            # Actually, just checking count_bad_triplets every step is O(N * avg_deg).
            # N=128, avg_deg is small. 128 * 10 = 1280 ops. 20000 steps = 25M ops. Fast enough.
            
            new_bad = count_bad_triplets(current_in_set)
            new_score = sum(current_in_set) - penalty * new_bad
            
            if new_bad == 0:
                if sum(current_in_set) > sum(best_in_set):
                    best_in_set = current_in_set[:]
                    best_score = sum(best_in_set)
            
            current_score = new_score
            current_bad = new_bad
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final repair if best_in_set is not valid (should be, but just in case)
    # best_in_set is only updated if new_bad == 0.
    
    return [i for i in range(1, N+1) if best_in_set[i]]

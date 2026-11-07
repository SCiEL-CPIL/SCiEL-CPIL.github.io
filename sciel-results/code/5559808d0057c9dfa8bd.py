import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s buffer
    
    n = 12
    m = 40
    num_masks = 1 << n
    
    # Precompute comparability matrix for O(1) checks
    # comp[i][j] = 1 if i < j, 2 if i > j, 0 otherwise
    # This saves time in the inner loops
    comp = [[0] * num_masks for _ in range(num_masks)]
    for i in range(num_masks):
        for j in range(num_masks):
            if i == j:
                continue
            if (i & j) == i: # i subset j
                comp[i][j] = 1
                comp[j][i] = 2
            elif (j & i) == j: # j subset i
                comp[i][j] = 2
                comp[j][i] = 1

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_total_score(F):
        score = 0
        for i in range(len(F)):
            for j in range(len(F)):
                if i != j and comp[F[i]][F[j]] == 1:
                    score += 1
        return score // 2

    # 1. Initialization
    # Start with a maximal chain
    F = []
    for i in range(n + 1):
        F.append((1 << i) - 1)
        
    # Greedily add remaining sets
    # Maintain counts for speed, but use comp for verification if needed
    # However, maintaining counts is O(4096) per op.
    # With 40 sets, total updates ~ 40*4096 = 160k.
    # Hill climbing swaps: 40*4096 = 160k per pass.
    # If we do 1000 passes, that's 160M ops. Might be tight.
    
    # Let's use a simpler approach:
    # Keep F as a list of masks.
    # Score calculation: O(m^2) = 1600 ops.
    # Swap evaluation:
    #   Remove i, Add j.
    #   Delta = (gain from j) - (loss from i).
    #   gain from j = count of k in F\{i} such that j<k or k<j.
    #   loss from i = count of k in F\{i} such that i<k or k<i.
    #   
    #   We can compute this in O(m) per candidate swap.
    #   Total per pass: m * m * m = 40^3 = 64k ops.
    #   1000 passes = 64M ops. Very fast in Python? Maybe 5-10s.
    #   This allows for many more passes or restarts.

    # Greedy Add
    while len(F) < m:
        best_mask = -1
        best_score = -1
        for c in range(num_masks):
            if c in F: continue
            score = 0
            for f in F:
                if comp[c][f] == 1 or comp[f][c] == 1:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = c
        F.append(best_mask)
        
    # Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        # Randomize order to avoid cycles? Or just deterministic first.
        # Let's try randomizing the order of candidates to escape local minima.
        candidates = list(range(num_masks))
        random.shuffle(candidates)
        
        for i, s_i in enumerate(F):
            # Try replacing s_i with c
            for c in candidates:
                if c == s_i: continue
                if c in F: continue
                
                # Calculate gain of c in F \ {s_i}
                gain = 0
                for k in range(len(F)):
                    if k == i: continue
                    s_k = F[k]
                    if comp[c][s_k] == 1 or comp[s_k][c] == 1:
                        gain += 1
                
                # Calculate loss of s_i in F \ {s_i}
                loss = 0
                for k in range(len(F)):
                    if k == i: continue
                    s_k = F[k]
                    if comp[s_i][s_k] == 1 or comp[s_k][s_i] == 1:
                        loss += 1
                
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_swap = (i, c)
                    
        if best_swap:
            i, c = best_swap
            F[i] = c
            improved = True

    # Convert to lists
    result = [mask_to_list(mask) for mask in F]
    return result

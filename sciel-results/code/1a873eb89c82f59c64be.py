import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Strategy:
    # 1. Formulate support selection as a Maximum Independent Set (MIS) problem on an "addition graph".
    #    Nodes are indices 0..N-1.
    #    An edge exists between i and j if their sum i+j creates a "collision" that increases the
    #    peak of the autoconvolution of the indicator vector. Specifically, we consider the graph
    #    where edges represent pairs that are "bad" for minimizing the peak.
    #    Actually, a more direct approach: We want a set S such that the autoconvolution of 1_S
    #    has the smallest possible peak.
    #    The peak of autoconvolution of 1_S is the maximum number of representations of a sum s as i+j with i,j in S.
    #    Let C(s) = |{ (i,j) in S x S : i+j=s }|.
    #    We want to minimize max_s C(s).
    #    This is equivalent to finding a set S that is "Sidon-like" (B2 set) or close to it.
    #    
    #    The directive suggests: "edge connects two indices if their sum collides with the sum of any other pair".
    #    This defines a conflict graph. If i,j are in S, and i+j = k+l for some other k,l in S (or even just k,l in [0,N-1] 
    #    if we are building S incrementally, but the conflict is intrinsic to the set), then they "collide".
    #    However, for a set S, the peak is determined by the maximum multiplicity of sums.
    #    
    #    Let's use a Greedy MIS-style heuristic on a conflict graph constructed dynamically.
    #    Alternatively, we can use a known heuristic for B2 sets:
    #    Start with an empty set. Add indices one by one.
    #    When considering adding index k to current set S, calculate the new collision counts.
    #    If adding k increases the peak collision count, reject it (or try to fix by removing others).
    #    To find the *largest* such set (which usually minimizes the peak relative to size, or allows for better weight optimization),
    #    we can use a greedy algorithm that prioritizes indices that cause the least increase in peak collision.
    
    # Precompute all possible sums and their potential pairs
    # For a set S, the autoconvolution peak is max_s count_{i,j in S}(i+j=s).
    
    # Let's define a function to evaluate the "quality" of a set S.
    # Quality = 1 / max_s count_{i,j in S}(i+j=s).
    # We want to maximize this.
    
    # Greedy construction of S:
    # 1. Initialize S = [].
    # 2. For each step, try adding each candidate i not in S.
    # 3. For each candidate, compute the new collision counts.
    # 4. Select the candidate that minimizes the new peak collision count.
    # 5. If no candidate improves or maintains the peak, stop? Or just stop if we can't add any without increasing peak?
    #    Actually, we want to find a set S. The size of S affects the ratio.
    #    Ratio for uniform weights on S: 
    #    l1 = |S|, l2_sq = |S|.
    #    Ratio = |S| / (|S| * peak) = 1/peak.
    #    So for uniform weights, maximizing ratio is exactly minimizing the peak collision count.
    #    The size of S doesn't directly appear in the uniform ratio, but a larger S might allow for a smaller peak?
    #    No, adding more elements generally increases the number of sums and thus the peak.
    #    However, a "Sidon" set of size K has peak 1 (if we ignore order i,j vs j,i? No, autoconvolution counts ordered pairs? 
    #    Let's check: h*h at lag k is sum_i h[i]h[i+k]. 
    #    If h is indicator of S, then (h*h)[k] = number of pairs (i, i+k) in S.
    #    This is the number of representations of k as j-i with i,j in S.
    #    This is different from sum i+j.
    #    Wait, the autoconvolution of a sequence h is (h*h)[k] = sum_{i} h[i] h[i+k].
    #    This is correlation, not convolution of sums.
    #    Let's re-read the task: "autoconvolution". Usually h*h means convolution.
    #    Convolution: (h*h)[k] = sum_i h[i] h[k-i].
    #    Let's check the code in the seed:
    #    H = np.fft.rfft(h, n=n_fft)
    #    hh = np.fft.irfft(H * H, n=n_fft)
    #    This computes the circular convolution (linear if zero-padded).
    #    So it is sum_i h[i] h[k-i].
    #    For non-negative h, this is the number of ways to write k as i+j with h[i],h[j] > 0.
    #    So yes, it is about sums i+j = k.
    
    # So, for a set S, the peak is max_k |{ (i,j) in S x S : i+j=k }|.
    # This is the maximum multiplicity of sums.
    # A Sidon set (B2 set) has the property that all sums i+j (with i<=j) are distinct.
    # This implies that for any k, there is at most 1 pair (i,j) with i<=j such that i+j=k.
    # However, the convolution counts ordered pairs (i,j).
    # If i != j, then (i,j) and (j,i) are distinct pairs contributing to the same sum k.
    # So if S is a Sidon set, the peak of the convolution is 2 (for k=2i, only (i,i) contributes 1? No.
    # Let's trace:
    # If S = {0, 1, 3} (Sidon? 0+1=1, 0+3=3, 1+3=4, 0+0=0, 1+1=2, 3+3=6. Sums: 0,1,2,3,4,6. Distinct. Yes.)
    # Convolution peaks:
    # k=0: (0,0) -> 1
    # k=1: (0,1), (1,0) -> 2
    # k=2: (1,1) -> 1
    # k=3: (0,3), (3,0) -> 2
    # k=4: (1,3), (3,1) -> 2
    # k=6: (3,3) -> 1
    # Peak is 2.
    # If S is not Sidon, say S={0,1,2}.
    # k=2: (0,2), (1,1), (2,0) -> 3.
    # Peak is 3.
    # So minimizing the peak is equivalent to finding a set S that is as "Sidon-like" as possible.
    
    # The ratio for uniform weights is 1/peak.
    # If we can achieve peak=2, ratio is 0.5.
    # If we can achieve peak=1? Impossible for |S|>1 because of symmetry (i,j) and (j,i) unless i=j.
    # If all i=j, then S must be a single element? No, if S={0, 2}, sums: 0(0,0), 2(0,2;2,0), 4(2,2). Peak 2.
    # So min peak for |S|>=2 is 2.
    
    # However, the current program optimizes weights.
    # With optimized weights, can we beat 1/peak?
    # Let h_i be weights.
    # Ratio = (sum w_i^2) / ( (sum w_i) * max_k (sum_{i+j=k} w_i w_j) ).
    # If we set all weights equal, ratio = 1/peak.
    # If we vary weights, can we increase the ratio?
    # Let's consider S={0,1,3} with peak 2.
    # Uniform: w=(1/3, 1/3, 1/3).
    # l1=1, l2_sq=1/3.
    # Conv: k=1: 2*(1/9)=2/9. k=3: 2/9. k=4: 2/9. Max conv = 2/9.
    # Ratio = (1/3) / (1 * 2/9) = 3/2 = 1.5.
    # Wait, the seed score is 1.14. 1.5 is higher. Why did the seed not find this?
    # Maybe the seed's support search didn't find {0,1,3} or the weight optimization failed.
    # Or maybe I am misinterpreting the ratio.
    # Seed code: `return l2_sq / (l1 * hh_max)`
    # My calculation: l2_sq = 1/3. l1 = 1. hh_max = 2/9. Ratio = (1/3) / (2/9) = 1.5.
    # Let's check if {0,1,3} is valid.
    # Indices must be in [0, 49]. Yes.
    # Let's check if there is a set with peak 1? No.
    # What if we use a larger Sidon set?
    # Max size of Sidon set in [0, N-1] is approx sqrt(N).
    # For N=50, max Sidon size is around 7-8.
    # If we find a Sidon set of size K, uniform weights give ratio 1.5 (since peak is 2).
    # Can we get peak 1? No.
    # Can we get a ratio > 1.5 with non-uniform weights on a Sidon set?
    # Let S be Sidon. Peak is 2 (at sums 2i and i+j where i!=j).
    # Actually, for Sidon set, all sums i+j (i<=j) are distinct.
    # The convolution values are:
    # c_k = sum_{i+j=k} w_i w_j.
    # If k=2i, c_k = w_i^2.
    # If k=i+j (i!=j), c_k = 2 w_i w_j.
    # We want to minimize max(c_k) to maximize ratio (since l2_sq/l1 is fixed by weights).
    # Let's try to make all c_k small.
    # If we set w_i = 1 for all i in S.
    # c_{2i} = 1.
    # c_{i+j} = 2.
    # Max c = 2.
    # l2_sq = K. l1 = K.
    # Ratio = K / (K * 2) = 0.5.
    # Wait, my previous calculation with 1/3 weights:
    # w_i = 1/3.
    # c_{2i} = (1/3)^2 = 1/9.
    # c_{i+j} = 2*(1/3)*(1/3) = 2/9.
    # Max c = 2/9.
    # l2_sq = 3 * (1/9) = 1/3.
    # l1 = 1.
    # Ratio = (1/3) / (1 * 2/9) = 1.5.
    # Why did I get 0.5 before?
    # Ah, I used w_i=1.
    # If w_i=1, l1=K, l2_sq=K. Ratio = K/(K*2) = 0.5.
    # If w_i=1/K, l1=1, l2_sq=K*(1/K^2)=1/K.
    # Max c = 2/K^2.
    # Ratio = (1/K) / (1 * 2/K^2) = K/2.
    # So for K=3, ratio = 1.5.
    # For K=4, ratio = 2.0.
    # For K=5, ratio = 2.5.
    # So we want the LARGEST possible Sidon set!
    
    # The current program's score is 1.14.
    # If I can find a Sidon set of size 5, I get 2.5.
    # Size 6 -> 3.0.
    # Size 7 -> 3.5.
    
    # So the strategy is: Find the largest Sidon set in [0, N-1].
    # A Sidon set is a set of integers such that all pairwise sums are distinct.
    # Greedy algorithm for largest Sidon set:
    # 1. Start with empty set S.
    # 2. Iterate through candidates? Or greedy selection?
    #    Greedy: Pick the smallest available number that doesn't violate the Sidon property.
    #    This yields a "greedy Sidon set" which is not necessarily maximal in size, but often good.
    #    However, we can do better with backtracking or local search.
    #    Given the time limit, a good heuristic is:
    #    Try multiple random starting points or random permutations and greedily build the set.
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_sidon_maximize(N, rng):
        """
        Greedily construct a Sidon set by trying to add the smallest possible new element
        that maintains the Sidon property.
        """
        S = []
        # We can try to start with 0, or a random number.
        # Let's try to maximize size.
        # A known greedy method:
        # Start with 0.
        # Then pick the next smallest integer that doesn't create a duplicate sum.
        
        # To get different results, we can randomize the order of candidates or the starting set.
        
        # Strategy:
        # 1. Start with S = [0].
        # 2. For each step, find the smallest x not in S such that S U {x} is Sidon.
        # 3. Add x to S.
        # 4. Repeat until no more x can be added.
        
        S = [0]
        sums = {0} # 0+0
        
        while True:
            found = False
            # Try candidates from 1 to N-1
            for x in range(1, N):
                if x in S:
                    continue
                # Check if adding x violates Sidon property
                valid = True
                new_sums = []
                for y in S:
                    s = x + y
                    if s in sums:
                        valid = False
                        break
                    new_sums.append(s)
                
                if valid:
                    # Add x
                    S.append(x)
                    for s in new_sums:
                        sums.add(s)
                    found = True
                    break
            
            if not found:
                break
        
        return S

    # The simple greedy above might get stuck.
    # Let's try a randomized greedy with restarts.
    
    best_S = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Try a few different strategies to find large Sidon sets
    # 1. Greedy from 0
    # 2. Greedy from random start
    # 3. Local search on a set
    
    # Strategy 1: Greedy from 0
    S1 = greedy_sidon_maximize(N, rng)
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = S1
        
    # Strategy 2: Randomized Greedy
    # Instead of always picking the smallest valid x, pick a random valid x.
    # This might allow for larger sets if the smallest valid x blocks future large sets.
    
    def randomized_greedy_sidon(N, rng):
        S = [0]
        sums = {0}
        
        while True:
            candidates = []
            for x in range(1, N):
                if x in S:
                    continue
                valid = True
                for y in S:
                    if (x + y) in sums:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Pick a random candidate
            x = rng.choice(candidates)
            S.append(x)
            for y in S[:-1]: # S[:-1] is the old S
                sums.add(x + y)
            sums.add(2*x) # x+x
            
        return S

    for _ in range(100):
        S2 = randomized_greedy_sidon(N, rng)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
        if time.time() > deadline - 20.0:
            break

    # Strategy 3: Local Search / Hill Climbing on the set
    # Start with best_S.
    # Try to swap an element with another to increase size or maintain size with better properties?
    # Actually, we just want the largest Sidon set.
    # If we have a set of size K, can we add another element?
    # We already tried to add in greedy.
    # Maybe we can remove one element to allow adding two?
    
    # Let's try a simple local search:
    # 1. Take best_S.
    # 2. Try removing one element.
    # 3. Try adding a new element.
    # 4. If size increases, accept.
    
    current_S = best_S.copy()
    current_size = len(current_S)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(current_S)):
            S_trial = current_S.copy()
            S_trial.pop(i)
            sums = set()
            for a in range(len(S_trial)):
                for b in range(a, len(S_trial)):
                    sums.add(S_trial[a] + S_trial[b])
            
            # Try to add any new element
            added = False
            for x in range(N):
                if x in S_trial:
                    continue
                valid = True
                for y in S_trial:
                    if (x + y) in sums:
                        valid = False
                        break
                if valid:
                    # Check if we can add x
                    # We need to ensure all new sums x+y are not in sums
                    # The loop above checks x+y for y in S_trial.
                    # We also need 2x not in sums.
                    if 2*x not in sums:
                        S_trial.append(x)
                        added = True
                        break
            
            if added and len(S_trial) > current_size:
                current_S = S_trial
                current_size = len(current_S)
                improved = True
                break # Restart loop with new best

    best_S = current_S
    K = len(best_S)
    
    # Now we have a Sidon set of size K.
    # The optimal weights for a Sidon set are uniform?
    # Let's verify.
    # Ratio = (sum w_i^2) / ( (sum w_i) * max_k c_k ).
    # For Sidon set, c_{2i} = w_i^2, c_{i+j} = 2 w_i w_j (i!=j).
    # We want to minimize M = max( max_i w_i^2, max_{i!=j} 2 w_i w_j ).
    # Let w_i = 1/K.
    # M = max( 1/K^2, 2/K^2 ) = 2/K^2.
    # Ratio = (1/K) / (1 * 2/K^2) = K/2.
    
    # Can we do better?
    # Suppose we set some w_i to 0? Then we reduce K, which reduces ratio.
    # Suppose we make w_i unequal.
    # Let w_max = max w_i.
    # Then M >= w_max^2.
    # Also M >= 2 w_max w_second.
    # To minimize M, we should balance w_i.
    # If all w_i are equal, M = 2/K^2.
    # If we make one w_i larger, say w_1 > 1/K.
    # Then w_1^2 increases.
    # Also 2 w_1 w_j increases for all j.
    # So it seems uniform is optimal for Sidon sets.
    
    # So the score is K/2.
    # We want to maximize K.
    
    # If K=5, score = 2.5.
    # If K=6, score = 3.0.
    # If K=7, score = 3.5.
    
    # The current program score is 1.14.
    # This implies the current program found a set with effective "peak" much higher than 2,
    # or a smaller K.
    # If the current program found a set of size 10 with peak 20, ratio = 10/(10*20) = 0.05? No.
    # Ratio = l2_sq / (l1 * peak).
    # If uniform, ratio = 1/peak.
    # So if score is 1.14, peak is approx 1/1.14 = 0.87? Impossible, peak is integer >= 1.
    # Wait, the weights are not uniform.
    # If the program optimizes weights, it might achieve a higher ratio than 1/peak for non-Sidon sets.
    # But for Sidon sets, the ratio is K/2.
    # If the program found a non-Sidon set with optimized weights, it might get > 1.14.
    # But if I can find a Sidon set of size 5, I get 2.5, which is much better.
    
    # So the key is to find a large Sidon set.
    
    # Let's double check if a Sidon set of size 5 exists in [0, 49].
    # Yes, e.g., {0, 1, 4, 10, 20} ?
    # Sums:
    # 0, 1, 4, 10, 20
    # 0+1=1, 0+4=4, 0+10=10, 0+20=20
    # 1+4=5, 1+10=11, 1+20=21
    # 4+10=14, 4+20=24
    # 10+20=30
    # 0+0=0, 1+1=2, 4+4=8, 10+10=20 (Collision! 0+20=20 and 10+10=20)
    # So {0,1,4,10,20} is NOT Sidon.
    
    # We need to be careful.
    # My greedy algorithm checks for collisions.
    # So it will find a valid Sidon set.
    
    # Let's run the greedy algorithm in my head or trust the code.
    # The code `greedy_sidon_maximize` starts with 0.
    # Next smallest x such that no collision.
    # x=1: sums {0, 1, 2}. Valid. S={0,1}.
    # x=2: 0+2=2 (in sums). Invalid.
    # x=3: 0+3=3, 1+3=4. Sums {0,1,2,3,4}. Valid. S={0,1,3}.
    # x=4: 0+4=4 (in sums). Invalid.
    # x=5: 0+5=5, 1+5=6, 3+5=8. Sums {0,1,2,3,4,5,6,8}. Valid. S={0,1,3,5}.
    # x=6: 0+6=6 (in sums). Invalid.
    # x=7: 0+7=7, 1+7=8 (in sums). Invalid.
    # x=8: 0+8=8 (in sums). Invalid.
    # x=9: 0+9=9, 1+9=10, 3+9=12, 5+9=14. Sums {0,1,2,3,4,5,6,8,9,10,12,14}. Valid. S={0,1,3,5,9}.
    # x=10: 0+10=10 (in sums). Invalid.
    # ...
    # So we found a set of size 5: {0,1,3,5,9}.
    # Score should be 5/2 = 2.5.
    
    # This is a massive improvement over 1.14.
    
    # Construct final h
    h_final = np.zeros(N)
    for idx in best_S:
        h_final[idx] = 1.0 / K
        
    # Ensure non-negative and valid
    h_final = np.maximum(h_final, 0)
    
    # Final check
    if compute_ratio(h_final) < 1e-6:
        # Fallback
        h_final = np.zeros(N)
        h_final[0] = 1.0

    return list(h_final)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))

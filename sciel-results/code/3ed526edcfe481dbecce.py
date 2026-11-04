"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_heights = np.ones(N)
    best_score = np.inf

    # Objective: Minimize max(autoconvolution) subject to ||h||_2 = 1, h >= 0
    # The score is presumably max(f*f) / ||f||_2^2. If we fix ||f||_2=1, we minimize max(f*f).
    # For the box function (h=1), ||h||_2 = sqrt(N), max(f*f) = N. Ratio = 1.
    # If we normalize to ||h||_2=1, box function becomes h=1/sqrt(N).
    # Then max(f*f) = max(1/N * box_conv) = max(1/N * triangle) = (1/N)*N = 1? No.
    # Let's trace:
    # h_box[i] = 1.
    # f = h_box.
    # f*f is discrete convolution. Sum of f*f = (sum f)^2 = N^2.
    # Max of f*f is N (at peak).
    # Score = Max(f*f) / (Sum(f^2))? Or Max(f*f) / ||f||_2^2?
    # ||f||_2^2 = Sum(f^2) = N.
    # Ratio = N/N = 1.
    #
    # If we change h to flatten the peak:
    # Let h be uniform. Convolution is triangular. Peak is 2x average height of convolution?
    # Average of convolution = N^2 / N = N. Peak = N. Ratio = 1.
    # Wait, the average of the convolution (over the support length 2N-1) is N^2/(2N-1) approx N/2.
    # The peak is N.
    # The "normalized maximum" usually means Max(f*f) / ||f||_2^2.
    # For Box: Max = N, ||f||_2^2 = N. Ratio = 1.
    #
    # Can we get < 1?
    # We need Max(f*f) < ||f||_2^2.
    # ||f||_2^2 is the energy. Max(f*f) is the peak of the energy distribution of the convolution.
    # By Cauchy-Schwarz, (f*f)(0) = sum f(i)^2 = ||f||_2^2.
    # Wait, for continuous functions, (f*f)(0) = integral f(x)f(-x) dx = integral f(x)^2 dx if even.
    # For discrete, (f*f)[0] = sum f[i] f[0-i]? No, standard conv is sum f[i] g[k-i].
    # If f is symmetric (palindromic), f*f is centered.
    # (f*f)[k] = sum_i f[i] f[k-i].
    # If f is supported on [0, N-1], the peak is usually at the center k=N-1.
    # (f*f)[N-1] = sum_{i=0}^{N-1} f[i] f[N-1-i].
    # If f is constant, this sum is N * 1^2 = N.
    # ||f||_2^2 = N * 1^2 = N.
    # Ratio = 1.
    #
    # If f is a "low autocorrelation" sequence, the side lobes are small.
    # But we are minimizing the MAX. The max is at the center (zero lag) for positive functions?
    # Yes, for non-negative f, (f*f)[k] is maximized when the overlap of f and reversed f is maximal, which is at lag 0 (or center of support).
    # So Max(f*f) = (f*f)[N-1] = sum f[i] f[N-1-i].
    # We want to minimize this sum subject to ||f||_2 = 1 and f >= 0.
    # Let g[i] = f[i]. We want to minimize sum g[i] g[N-1-i].
    # Let h[i] = g[i] and j[i] = g[N-1-i]. Then we minimize sum h[i] j[i].
    # Note that j is just the reverse of h.
    # If h is symmetric, h[i] = h[N-1-i], then sum h[i]^2 = ||h||_2^2 = 1.
    # If h is NOT symmetric, can we make the dot product smaller?
    # Let u = (h+j)/2 and v = (h-j)/2.
    # sum h[i] j[i] = sum (u+v)(u-v) = sum u^2 - sum v^2.
    # ||h||_2^2 = ||u+v||^2 = ||u||^2 + ||v||^2 + 2<u,v>.
    # This seems complicated.
    # Simpler: Let f be non-negative.
    # Let S = sum f[i]^2 = 1.
    # Let P = sum f[i] f[N-1-i].
    # We know that for any real vectors a, b, a.b <= ||a|| ||b||.
    # Here a=f, b=rev(f). ||a||=||b||=1. So P <= 1.
    # Equality holds if a is proportional to b, i.e., f is symmetric.
    # If f is NOT symmetric, P < 1.
    # So we can get a score < 1!
    # The minimum P is 0?
    # We need f[i] f[N-1-i] = 0 for all i.
    # This means f[i] and f[N-1-i] cannot be both non-zero.
    # Since f >= 0, this implies that for every pair (i, N-1-i), at least one is 0.
    # We can set f[i] > 0 only for i < (N-1)/2.
    # Let's check the convolution peak.
    # If f is supported on [0, N/2], then the convolution support is [0, N-1].
    # The "zero lag" in the centered sense corresponds to index N-1 in the array if we align starts.
    # Actually, the definition of "maximum of autoconvolution" for a function on [-1/4, 1/4] with 600 intervals.
    # The intervals are equal. Let's assume the function is defined on indices 0..N-1.
    # The convolution is computed on the support of the sum.
    # If we shift the mass to one side, the peak of the convolution might still be 1 (normalized).
    # Let's test with N=4.
    # f = [1, 0, 0, 0]. ||f||^2 = 1.
    # Conv: [1, 0, 0, 0, 0, 0, 0]. Max = 1. Score = 1.
    # f = [0.7, 0.7, 0, 0]. ||f||^2 = 0.98. Normalized: f = [0.71, 0.71, 0, 0].
    # Conv: [0.5, 1.0, 0.5, 0, ...]. Max = 1.0. Score = 1.
    # f = [1, 0, 1, 0]. ||f||^2 = 2. Norm: [0.707, 0, 0.707, 0].
    # Conv: [0.5, 0, 0.5, 0, 0.5, 0, 0.5]. Max = 0.5.
    # Score = 0.5 / (Sum f^2 = 1) = 0.5?
    # Wait, if we normalize ||f||_2=1, then Sum f^2 = 1.
    # In the example f=[0.707, 0, 0.707, 0], Sum f^2 = 0.5+0.5=1.
    # Conv peak:
    # Index 0: 0.707*0.707 = 0.5
    # Index 2: 0.707*0.707 + 0*0 = 0.5
    # Index 4: 0.707*0.707 = 0.5
    # Max Conv = 0.5.
    # Score = 0.5 / 1 = 0.5.
    # This is a huge improvement!
    #
    # So the strategy is: Make f have sparse, alternating structure to minimize the peak of the convolution.
    # The theoretical minimum for non-negative f might be related to the packing of non-overlapping supports?
    # But f is on a fixed grid.
    # We want to minimize max_k sum_i f[i] f[k-i].
    # This is equivalent to minimizing the maximum value of the discrete autocorrelation (with sign + for convolution, which is autocorr of f and rev(f)).
    # For a "chirp" or "random phase" sequence, the sidelobes are low.
    # But we have a non-negative constraint.
    # The example above (alternating 1,0) worked well.
    # Let's try f[i] = 1 if i is even, 0 if odd.
    # N=600. 300 ones, 300 zeros.
    # ||f||^2 = 300.
    # Conv peak?
    # The convolution of a comb with a comb is a comb with double the amplitude?
    # Let's check N=4 again. f=[1,0,1,0].
    # Conv:
    # 0: 1*1 = 1
    # 1: 1*0+0*1 = 0
    # 2: 1*1+0*0+1*1 = 2  <-- Wait.
    # My previous manual calc was:
    # f = [a, 0, a, 0].
    # c[0] = a^2
    # c[1] = 0
    # c[2] = a^2 + a^2 = 2a^2
    # c[3] = 0
    # c[4] = a^2
    # Max is 2a^2.
    # ||f||^2 = 2a^2.
    # Score = 2a^2 / 2a^2 = 1.
    # So alternating 1,0 gives Score 1.
    #
    # What if we use 3 points?
    # f = [1, 0, 0, 1, 0, 0] (N=6).
    # c[0]=1, c[1]=0, c[2]=0, c[3]=1, c[4]=0, c[5]=0, c[6]=1.
    # Max = 1.
    # ||f||^2 = 2.
    # Score = 0.5.
    # This is better!
    #
    # General pattern: Place 1s as sparsely as possible.
    # If we place 1s at indices 0, k, 2k...
    # The convolution will have peaks at 0, k, 2k...
    # The height of the peak at 0 is 1 (self).
    # The height at k is 1 (cross).
    # If the supports don't overlap in a way that sums to > 1, the max is 1.
    # Score = 1 / Number_of_1s.
    # To minimize score, we maximize Number_of_1s such that no three 1s sum to the same convolution index?
    # No, the convolution index is i+j.
    # We want to minimize max_{s} (count of pairs (i,j) such that i+j=s and f[i]=f[j]=1).
    # This is the problem of minimizing the maximum additive energy or similar.
    # Actually, we just want the max count to be 1.
    # This is possible if the set of indices is a "Sidon set" or similar?
    # A Sidon set is a set of integers where all pairwise sums are distinct.
    # If all sums i+j (with i<=j) are distinct, then the max convolution value is 1 (for i!=j, it appears once? No, i+j = j+i is the same sum, but in convolution sum, if i!=j, the term f[i]f[j] appears twice?
    # Let's check conv definition.
    # (f*f)[s] = sum_{i+j=s} f[i] f[j].
    # If i!=j, the pair (i,j) and (j,i) are distinct terms in the sum?
    # Yes, the sum is over all i. So if i+j=s, both i and j contribute.
    # So if f is 1 at i and j (i!=j), the contribution to c[s] is 1*1 + 1*1 = 2?
    # Wait.
    # f = [1, 0, 1, 0, 0, 1] (N=6). Indices 0, 2, 5.
    # Pairs summing to 2: (0,2) -> f[0]f[2]=1. (2,0) -> f[2]f[0]=1. Sum=2.
    # Pairs summing to 5: (0,5)=1, (5,0)=1. Sum=2. (2,3)=0.
    # Pairs summing to 4: (0,4)=0, (2,2)=1, (4,0)=0. Sum=1.
    # Pairs summing to 7: (2,5)=1, (5,2)=1. Sum=2.
    # Max is 2.
    # ||f||^2 = 3.
    # Score = 2/3 = 0.66.
    #
    # If we use a Sidon set, all sums i+j (i<=j) are unique.
    # But convolution counts ordered pairs.
    # If i!=j, sum i+j is counted twice (i,j and j,i).
    # If i=j, sum 2i is counted once.
    # So if we have a Sidon set, the max convolution value is 2 (for any sum involving two distinct elements).
    # The number of elements is M.
    # ||f||^2 = M.
    # Score = 2/M.
    # We want to maximize M such that the indices form a Sidon set within [0, N-1].
    # The max size of a Sidon set in [1, N] is approx sqrt(N).
    # So M ~ sqrt(600) ~ 24.
    # Score ~ 2/24 = 0.08.
    # This is a massive improvement over 1.0.
    #
    # So the algorithm is:
    # 1. Find a large Sidon set (B_2 sequence) in {0, ..., N-1}.
    # 2. Set f[i] = 1 for i in Sidon set, 0 otherwise.
    # 3. Return f.
    #
    # We can also optimize the amplitudes.
    # But binary Sidon sets are a strong start.
    #
    # How to find a Sidon set?
    # Greedy:
    # S = []
    # for i in range(N):
    #   if i can be added to S without creating duplicate sums:
    #     S.append(i)
    #
    # This greedy approach usually finds a set of size ~ sqrt(N).
    #
    # Let's implement the greedy Sidon set finder.
    #
    # Time complexity:
    # N=600.
    # Loop N times.
    # For each i, check sums with existing S.
    # Size of S is ~24.
    # Sums to check: 24*24 = 576.
    # Total ~ 600 * 576 = 345,600 ops. Very fast.
    #
    # We should also consider that the "intervals" are on [-1/4, 1/4].
    # The convolution is likely performed on the discrete values.
    # The position (0 vs N/2) shouldn't matter for the value of the max, only the structure.
    #
    # One refinement:
    # Can we make the amplitudes non-uniform to reduce the max below 2?
    # If we have a Sidon set, the max is determined by the sums.
    # If we assign amplitudes a_i, the value at sum s (from i!=j) is a_i a_j + a_j a_i = 2 a_i a_j.
    # The value at sum 2i is a_i^2.
    # We want to minimize max( max_{i} a_i^2, max_{i!=j, i+j=s} 2 a_i a_j ).
    # If we set all a_i = 1/sqrt(M), then a_i^2 = 1/M, 2 a_i a_j = 2/M.
    # Max is 2/M.
    # ||f||^2 = Sum (1/M) = 1.
    # Score = 2/M.
    # This is optimal for a given Sidon set.
    #
    # So the plan:
    # 1. Generate a maximal Sidon set in [0, N-1] using greedy algorithm.
    # 2. Set heights to 1/sqrt(M) for i in S, 0 else.
    # 3. Return heights.
    #
    # We can try to improve the Sidon set size by restarting with different random seeds or shuffling the order, but greedy is usually good enough for N=600 to get ~20-30 elements.
    #
    # Let's write the code.

    def find_sidon_set(N):
        S = []
        sums = set()
        for i in range(N):
            can_add = True
            # Check all existing pairs
            # New sums formed by i with each j in S: i+j
            # Also i+i
            new_sums = []
            for j in S:
                new_sums.append(i + j)
            new_sums.append(2 * i)
            
            # Check for collision
            for s in new_sums:
                if s in sums:
                    can_add = False
                    break
            
            if can_add:
                S.append(i)
                for s in new_sums:
                    sums.add(s)
        
        return S

    # Try to find a good Sidon set.
    # The order of iteration matters. Iterating 0..N-1 is standard.
    # Maybe shuffling helps?
    # Given the time limit, we can try a few random permutations if we have time, but greedy 0..N-1 is fast.
    
    S = find_sidon_set(N)
    M = len(S)
    
    if M == 0:
        return [1.0] * N
        
    heights = np.zeros(N)
    # Normalize so ||f||_2 = 1
    # Sum h^2 = M * (1/sqrt(M))^2 = 1.
    val = 1.0 / np.sqrt(M)
    for i in S:
        heights[i] = val
        
    # Convert to list as required
    return heights.tolist()

# EVOLVE-BLOCK-END

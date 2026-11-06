import time
import math
import numpy as np

def construct():
    start_time = time.time()
    # Set a deadline 10 seconds before the 120s limit (150s wall clock mentioned in directive, but contract says 120s)
    # Using 110s to be safe for 120s limit.
    deadline = start_time + 110.0

    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Use FFT to find candidate sums N with high representation counts.
    # 3. Verify candidates and extract pairs.
    #
    # The current program uses a double loop O(K^2) which is slow for large K.
    # Inspiration 1 uses NumPy broadcasting for sums, which is fast but memory heavy.
    # Inspiration 2 uses FFT, which is O(L log L) and efficient for large L.
    #
    # We will use the FFT approach (Inspiration 2) as the primary method because it scales better
    # with L, allowing us to cover a larger range of powerful numbers, which likely leads to higher counts.
    #
    # Powerful numbers are sparse. The number of powerful numbers <= L is approx L^(1/2) + L^(1/3).
    # For L = 10^8, K ~ 10^4.
    # For L = 10^9, K ~ 3*10^4.
    # FFT size M needs to be >= 2*L.
    # For L=10^8, M ~ 2^28 = 268M. This might be too much memory for float64 (2GB).
    # Let's stick to a manageable L.
    # L = 5 * 10^7 => M ~ 2^27 = 134M. 1GB memory. Doable.
    # L = 10^7 => M ~ 2^25 = 33M. 256MB memory. Very safe.
    # Let's try L = 2 * 10^7. M ~ 2^26 = 67M. 512MB.
    #
    # Actually, let's look at the score of Inspiration 1 (5.93) vs Inspiration 2 (2.86).
    # Inspiration 1 used L=10^8 and NumPy broadcasting. It scored higher.
    # Why? Maybe the top counts are found more reliably or the constant factors are better for the specific search.
    # Or maybe the FFT had precision issues or the candidate verification was the bottleneck.
    #
    # However, the directive asks for a STRUCTURAL change from the current program (which is O(K^2) Python loop).
    # The current program is slow.
    # Inspiration 1's approach (NumPy broadcasting) is a structural change from Python loops to vectorized C loops.
    # Inspiration 2's approach (FFT) is also a structural change.
    #
    # Let's combine: Use NumPy to generate the powerful numbers efficiently? No, generation is O(K) and fast enough.
    # The bottleneck in the current program is the O(K^2) counting loop.
    #
    # Let's try the NumPy broadcasting approach from Inspiration 1, but with careful memory management.
    # If K is ~10,000, K^2 is 10^8. An array of 10^8 int64s is 800MB.
    # We can compute the sums in blocks to save memory, or just try the full matrix if memory allows.
    #
    # Let's use L = 5 * 10^7.
    # K ~ sqrt(5e7) + (5e7)^(1/3) ~ 7000 + 3700 ~ 10,000.
    # K^2 = 10^8.
    #
    # Alternative: Use FFT but with a larger L?
    # If L=10^8, K~20,000. FFT size 2^28.
    #
    # Let's stick to the NumPy broadcasting method (Inspiration 1) because it scored higher in the inspirations.
    # We will optimize the generation and the sum counting.
    
    limit = 50000000  # 5 * 10^7
    
    # Generate powerful numbers
    pows = set([1])
    max_i = math.isqrt(limit)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    # Efficient generation
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > limit:
            break
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        
        # Generate values for this i
        # val = sq * j^3
        # We can create a numpy array of cubes and multiply by sq
        # cubes_np = np.array(cubes[:max_j], dtype=np.int64)
        # vals = sq * cubes_np
        # pows.update(vals.tolist()) -- this might be slow due to list conversion
        
        # Let's just loop in Python, it's O(K) total which is ~10k iterations. Fast.
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
                
        if time.time() > deadline - 20:
            break

    P_list = sorted(pows)
    K = len(P_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Convert to NumPy array
    P = np.array(P_list, dtype=np.int64)
    
    # Compute all sums P_i + P_j
    # We only need i <= j to count distinct pairs a <= b?
    # No, np.unique counts all ordered pairs.
    # If we compute full KxK matrix, we get ordered pairs.
    # Count of ordered pairs (a,b) such that a+b=N is C_full.
    # Number of distinct pairs a<=b is (C_full + D) // 2, where D=1 if N/2 is powerful else 0.
    
    # To save memory, we can compute the upper triangle?
    # But np.unique is easiest on the full array.
    # Let's try the full matrix.
    
    # Check time
    if time.time() > deadline - 10:
         # Fallback to a simple search if generation took too long
         # Just return a small valid result
         return {'N': 2, 'pairs': [[1, 1]]}

    # Create the sum matrix
    # This is the most memory-intensive step
    # K ~ 10000, so 10000x10000 = 10^8 elements.
    # 8 bytes per element => 800 MB.
    
    try:
        sums = P[:, None] + P[None, :]
        
        # Count occurrences
        # np.unique on 10^8 elements might be slow but doable.
        # Alternative: Use a histogram if the range of sums is small?
        # Max sum is 2*limit = 10^8.
        # A histogram of size 10^8 is 10^8 * 8 bytes = 800MB.
        # np.bincount might be faster than np.unique.
        
        # Let's use np.bincount
        # The maximum value in sums is 2*limit.
        # We need to ensure the array size is sufficient.
        max_sum = 2 * limit
        # bincount requires non-negative integers.
        # sums are positive.
        
        # Check time again
        if time.time() > deadline - 5:
            raise TimeoutError("Too slow")

        # bincount is fast
        counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
        
        # Find the index with the maximum count
        # We want to maximize the number of distinct pairs a <= b.
        # Score for sum s is (counts[s] + (1 if s/2 in P else 0)) // 2
        # counts[s] is the number of ordered pairs.
        
        # We can find the top few candidates from bincount and then verify/refine.
        # But bincount gives exact counts for ordered pairs.
        # The adjustment for a<=b is small (at most 0.5).
        # So the sum with max counts[s] is likely the one with max distinct pairs.
        
        # Find top 10 indices
        # argsort is O(N log N). N=10^8. This might be slow.
        # argpartition is O(N).
        # We only need the top few.
        
        # However, we must consider the adjustment.
        # Let's get the top 20 candidates by counts[s].
        
        # Filter out s=0 (if any, though pows start at 1, min sum is 2)
        # pows min is 1. Min sum is 2.
        
        # Get indices of top 20 counts
        # We ignore index 0 and 1 as they are 0.
        # We want s >= 2.
        
        # To avoid sorting the whole array, we can use argpartition on the slice [2:]
        # Or just find the max, then mask it, find next max, etc. for top 10.
        # That's 10 * O(N) = O(10N). N=10^8. 10^9 ops. Might be ~1s.
        
        # Let's use argpartition for efficiency.
        # We want the 20 largest values in counts[2:]
        
        c_slice = counts[2:]
        # Find indices of the 20 largest elements
        # argpartition returns indices that would partition the array into 20 largest and rest.
        # The last 20 elements of the partitioned array are the top 20, but not sorted.
        
        # Note: argpartition on 10^8 elements is fast (~0.5s).
        
        top_indices_local = np.argpartition(c_slice, -20)[-20:]
        top_indices = top_indices_local + 2  # Adjust for the slice offset
        
        # Now verify these candidates
        pows_set = set(P_list)
        
        best_N = 0
        best_pairs = []
        best_score = 0
        
        for s in top_indices:
            s_val = int(s)
            if s_val > 2 * limit:
                continue
            
            # Calculate exact distinct pair count
            # Count a <= b such that a+b=s_val and a,b in pows
            pairs = []
            # Iterate through P_list
            # P_list is sorted.
            # We can break if a > s_val / 2
            half = s_val / 2.0
            
            for a in P_list:
                if a > half:
                    break
                b = s_val - a
                if b in pows_set:
                    pairs.append([a, b])
            
            score = len(pairs)
            if score > best_score:
                best_score = score
                best_N = s_val
                best_pairs = pairs
            
            if time.time() > deadline - 2:
                break
        
        if not best_pairs:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        return {'N': best_N, 'pairs': best_pairs}

    except MemoryError:
        # Fallback to smaller limit or different method
        # If memory fails, we can try a smaller limit
        # Or just return a trivial result
        return {'N': 2, 'pairs': [[1, 1]]}
    except TimeoutError:
        return {'N': 2, 'pairs': [[1, 1]]}
    except Exception:
        # Catch any other errors
        return {'N': 2, 'pairs': [[1, 1]]}

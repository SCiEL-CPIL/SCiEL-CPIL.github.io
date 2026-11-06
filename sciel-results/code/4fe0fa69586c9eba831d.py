import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    # 150s limit, keep 20s buffer for safety and final extraction
    deadline = start_time + 125
    
    # Set L based on available memory and time.
    # L=2*10^8 generates ~30k powerful numbers.
    # Pairwise sums ~ 450M operations. NumPy can handle this in a few seconds.
    L = 2 * 10**8
    
    # 1. Generate powerful numbers up to L
    # Powerful numbers are of the form u^2 * v^3 where v is square-free.
    # v^3 <= L => v <= L^(1/3) ~ 584 for L=2*10^8
    max_v = int(L ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_v
    is_square_free = np.ones(max_v + 1, dtype=bool)
    is_square_free[0] = False
    is_square_free[1] = True
    
    for p in range(2, isqrt(max_v) + 1):
        if is_square_free[p]:
            p2 = p * p
            # Mark multiples of p^2 as not square-free
            is_square_free[p2::p2] = False
            
    sq_free = np.nonzero(is_square_free)[0]
    
    powerfuls_list = []
    # For each square-free v, generate u^2 * v^3 <= L
    # u <= sqrt(L / v^3)
    for v in sq_free:
        v3 = v ** 3
        if v3 > L:
            break
        max_u = isqrt(L // v3)
        if max_u > 0:
            # Generate u^2 for u in 1..max_u
            u_vals = np.arange(1, max_u + 1)
            p_vals = u_vals * u_vals * v3
            # Filter <= L (should be all, but safe)
            p_vals = p_vals[p_vals <= L]
            powerfuls_list.extend(p_vals)
            
    if not powerfuls_list:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Convert to numpy array, unique, sort
    powerfuls = np.unique(np.array(powerfuls_list, dtype=np.int64))
    P = len(powerfuls)
    
    # 2. Count sums a_i + a_j for i <= j
    # We need to be careful with memory. P ~ 30k.
    # Full matrix is 900M ints ~ 7GB. Too much.
    # We need to do this in chunks or use a smarter counting method.
    
    # Alternative: Use FFT? No, range is up to 4*10^8, too large for FFT.
    # Alternative: Sort and two pointers? No, we need counts for ALL sums.
    # Alternative: Chunked vectorization.
    
    # Let's estimate P more accurately.
    # Density of powerful numbers is ~ 1/sqrt(x).
    # Count ~ 2 * sqrt(L) ~ 2 * 14000 = 28000.
    # P^2/2 ~ 400 million.
    # If we do it in chunks of size K (e.g., 1000 rows), we create arrays of size 1000 * P.
    # 1000 * 30000 = 30 million ints. 30M * 8 bytes = 240 MB. This is fine.
    
    counts = np.zeros(2 * L + 1, dtype=np.int32) # Max sum is 2L. 400M ints ~ 1.6 GB. Might be tight.
    # Let's reduce L if memory is a concern, or use a dict if we can't afford the array.
    # Actually, 1.6GB is risky on some eval environments.
    # Let's try L = 10^8 first. P ~ 20k. P^2/2 ~ 200M.
    # Array size 2*10^8 * 4 bytes = 800 MB. Safe.
    
    # Re-evaluating L.
    # The prompt says "Integers up to 10^18".
    # The current program used L=5*10^7.
    # If I use L=10^8, N can be up to 2*10^8.
    # Let's stick to L=10^8 to be safe on memory, but optimize the counting.
    
    L = 10**8
    max_v = int(L ** (1/3)) + 2
    is_square_free = np.ones(max_v + 1, dtype=bool)
    is_square_free[0] = False
    is_square_free[1] = True
    for p in range(2, isqrt(max_v) + 1):
        if is_square_free[p]:
            p2 = p * p
            is_square_free[p2::p2] = False
    sq_free = np.nonzero(is_square_free)[0]
    
    powerfuls_list = []
    for v in sq_free:
        v3 = v ** 3
        if v3 > L:
            break
        max_u = isqrt(L // v3)
        if max_u > 0:
            u_vals = np.arange(1, max_u + 1)
            p_vals = u_vals * u_vals * v3
            p_vals = p_vals[p_vals <= L]
            powerfuls_list.extend(p_vals)
            
    powerfuls = np.unique(np.array(powerfuls_list, dtype=np.int64))
    P = len(powerfuls)
    
    # Counting sums
    # We will iterate over chunks of 'a' values.
    # For a chunk of size K, we compute sums with all 'b' values.
    # To ensure i <= j, we can just compute all pairs and divide by 2?
    # No, a+b is commutative. If we compute all P*P sums, each pair (a,b) with a!=b is counted twice.
    # Pairs (a,a) are counted once.
    # It's easier to just count all P*P sums, then for N, count = (total_counts[N] + number_of_squares_summing_to_N) / 2?
    # Actually, if we just want the max count, we can compute the full convolution-like sum.
    # Let's just do the upper triangle logic to save half the work?
    # Or just do full P*P and accept the 2x factor?
    # If we do full P*P, we get counts for all sums.
    # For a specific N, the number of pairs (a,b) with a<=b is:
    # If a != b, it appears as (a,b) and (b,a).
    # If a == b, it appears as (a,a).
    # So if we have total count C_N from full matrix:
    # Let S_N be the number of a such that 2a = N (i.e., a = N/2 and a is powerful).
    # Then pairs = (C_N - S_N) / 2 + S_N = (C_N + S_N) / 2.
    
    # Memory for counts array: Size 2*L.
    # L=10^8 -> 2*10^8 elements.
    # int32 -> 800 MB.
    # This is acceptable.
    
    counts = np.zeros(2 * L + 1, dtype=np.int32)
    
    # Chunk size
    # P is ~ 20,000.
    # If K=1000, we do 20 iterations.
    # Each iteration: (1000 x 20000) array. 20M elements. 160 MB.
    # This is fast.
    
    K = 1000
    for start in range(0, P, K):
        end = min(start + K, P)
        a_chunk = powerfuls[start:end]
        # Compute a_chunk[:, None] + powerfuls[None, :]
        # This creates a (K x P) array.
        sums = a_chunk[:, None] + powerfuls[None, :]
        # Flatten and count
        # np.add.at is slow. bincount is faster.
        # sums are in range [2, 2L]
        # We can use np.bincount on the flattened sums.
        # But we need to accumulate into 'counts'.
        # local_counts = np.bincount(sums.ravel(), minlength=2*L+1)
        # counts += local_counts
        
        # Optimization: Only count sums <= 2L? They all are.
        # We can restrict the range of sums if we want, but they are all valid.
        
        # To save memory/time, we can just add the bincount result.
        # However, bincount creates a new array of size 2L+1 (800MB) each time?
        # No, bincount returns an array of size max_val+1.
        # If we do this 20 times, we allocate 20 * 800MB? No, we can reuse or just add.
        # But creating an 800MB array 20 times is slow and memory heavy if not freed.
        
        # Better approach:
        # Use np.add.at on the global counts array?
        # np.add.at(counts, sums.ravel(), 1)
        # This is O(N) but with high constant factor.
        
        # Alternative:
        # Since we just need the MAX count, and we are doing this in chunks,
        # we can't just take max of local counts because counts accumulate.
        
        # Let's try a different strategy:
        # The number of powerful numbers is small (~20k).
        # We can store the powerful numbers in a set for O(1) lookup.
        # But we need to find N with max representations.
        # We don't know N in advance.
        
        # Let's stick to the bincount approach but be careful.
        # We can process smaller chunks to keep the bincount array smaller?
        # No, bincount size depends on max value, not input size.
        # Max value is 2L. So bincount array is always 2L+1.
        
        # Maybe L=10^8 is too big for 800MB * multiple allocations?
        # Let's reduce L to 5*10^7 to be safe (400MB array).
        # The current program used 5*10^7 and got 5.8.
        # If I can do 10^8, I might get higher.
        # Let's try to optimize the counting to avoid multiple large allocations.
        
        # We can use a dictionary for counts?
        # Dict with 20k keys? No, sums are up to 2L.
        # But only sums that actually occur are stored.
        # Number of distinct sums is at most P^2/2 ~ 200M.
        # A dict with 200M entries is huge and slow.
        
        # Let's go back to array.
        # To avoid multiple 800MB allocations, we can just use one array and add to it.
        # But bincount creates a new one.
        # We can do:
        # local = np.bincount(sums.ravel(), minlength=2*L+1)
        # counts += local
        # This allocates `local` (800MB) and then adds.
        # If memory is tight, this might fail.
        
        # Let's try L = 5 * 10**7 first.
        # If time permits, we can try to push it.
        
        # Actually, let's look at the "inspiration" again.
        # "Switching from FFT... to Numpy-vectorized sparse pair counting allows increasing L from 10^7 to 10^8".
        # It implies 10^8 is feasible.
        
        # Let's try L = 10**8.
        # If it crashes, we fall back? No, we can't easily fall back in a single run.
        # Let's be conservative: L = 8 * 10**7.
        
        # Let's re-define L and restart the logic in my head.
        # I will use L = 10**8.
        
    # Since I can't restart, I must commit to a value.
    # Let's use L = 10**8.
    # To handle memory, I will use a smaller chunk size for the summation matrix,
    # but the bincount array is the bottleneck.
    # Is there a way to count without a full 2L array?
    # We can sort the sums?
    # No, we need counts for all sums to find the max.
    
    # Wait, I can just use a Python dictionary for the counts if the number of *distinct* sums is small?
    # No, distinct sums can be large.
    
    # Let's try to use `np.add.at` directly on the global array.
    # It's slower but avoids the large temporary `local` array.
    
    # Reset L to 10**8
    L = 10**8
    max_v = int(L ** (1/3)) + 2
    is_square_free = np.ones(max_v + 1, dtype=bool)
    is_square_free[0] = False
    is_square_free[1] = True
    for p in range(2, isqrt(max_v) + 1):
        if is_square_free[p]:
            p2 = p * p
            is_square_free[p2::p2] = False
    sq_free = np.nonzero(is_square_free)[0]
    
    powerfuls_list = []
    for v in sq_free:
        v3 = v ** 3
        if v3 > L:
            break
        max_u = isqrt(L // v3)
        if max_u > 0:
            u_vals = np.arange(1, max_u + 1)
            p_vals = u_vals * u_vals * v3
            p_vals = p_vals[p_vals <= L]
            powerfuls_list.extend(p_vals)
            
    powerfuls = np.unique(np.array(powerfuls_list, dtype=np.int64))
    P = len(powerfuls)
    
    # Initialize counts array
    # 2 * 10^8 * 4 bytes = 800 MB.
    # This is a single allocation.
    counts = np.zeros(2 * L + 1, dtype=np.int32)
    
    # Use np.add.at
    # This is O(P^2) but with vectorized inner loop?
    # No, np.add.at is not fully vectorized in the sense of C loops, it's a ufunc with reduce.
    # It might be slow.
    
    # Let's try the bincount approach but with a smaller L if memory fails?
    # I can't catch memory errors easily.
    
    # Let's assume 800MB is OK.
    # To speed up, we can process in chunks and use bincount, but we have to add to the main array.
    # The cost of bincount is O(N + K).
    # The cost of addition is O(K).
    # The bottleneck is the allocation of the bincount array.
    
    # Alternative:
    # If we only care about the MAX count, we don't need to store all counts?
    # No, we need to find the max.
    
    # Let's try a different tactic:
    # Generate all pairs (a, b) with a <= b.
    # Store them in a list of tuples?
    # P=20k. P^2/2 = 200M pairs.
    # 200M tuples is too much memory.
    
    # Let's stick to the array approach with L=10^8.
    # If it's too slow, the deadline will kill it.
    
    K = 2000 # Larger chunks to reduce overhead
    for start in range(0, P, K):
        if time.time() > deadline:
            break
        end = min(start + K, P)
        a_chunk = powerfuls[start:end]
        sums = a_chunk[:, None] + powerfuls[None, :]
        # We only care about sums <= 2L
        # All sums are <= 2L.
        # Use bincount
        local_counts = np.bincount(sums.ravel(), minlength=2 * L + 1)
        counts += local_counts
        
    # Find best N
    # counts is an array of size 2L+1.
    # We need to find the index with the max value.
    # Note: The count from full matrix includes (a,b) and (b,a).
    # We want pairs a <= b.
    # As derived: pairs(N) = (counts[N] + S_N) / 2, where S_N is 1 if N/2 is powerful, else 0.
    
    # To find the best N, we can just find the max of counts[N] + S_N.
    # S_N is 0 or 1. It barely affects the max unless counts are tied.
    # So we can just find argmax(counts).
    
    best_N_idx = np.argmax(counts)
    best_N = int(best_N_idx)
    
    # Calculate actual score for verification and to get pairs
    # Check if best_N/2 is powerful
    powerfuls_set = set(powerfuls.tolist())
    
    # Get pairs
    pairs = []
    half_N = best_N / 2
    is_half_powerful = (best_N % 2 == 0) and (int(half_N) in powerfuls_set)
    
    # We need to find all a in powerfuls such that a <= best_N - a
    # i.e., a <= best_N / 2
    # and best_N - a is in powerfuls
    
    # Filter powerfuls <= best_N/2
    # powerfuls is sorted
    max_a = best_N // 2
    # Find index of first element > max_a
    idx = np.searchsorted(powerfuls, max_a, side='right')
    candidate_as = powerfuls[:idx]
    
    for a in candidate_as:
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([int(a), int(b)])
            
    # Sort pairs
    pairs.sort()
    
    # The score is len(pairs)
    
    return {'N': best_N, 'pairs': pairs}

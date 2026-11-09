"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def generate_powerful(L):
        """Generate all powerful numbers <= L using a^2 * b^3 form."""
        powerful_set = set()
        powerful_set.add(1)
        max_b = int(round(L ** (1/3))) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)
        return powerful, powerful_set

    def count_pairs(powerful, powerful_set, N_max=10**18, time_limit=None):
        """Count representations N = a + b where a, b are powerful, a <= b."""
        count = defaultdict(int)
        n_p = len(powerful)
        pairs_seen = 0
        for i in range(n_p):
            a = powerful[i]
            for j in range(i, n_p):
                b = powerful[j]
                N = a + b
                if N > N_max:
                    break
                count[N] += 1
                pairs_seen += 1
                if time_limit is not None and pairs_seen % 100000 == 0:
                    if time.time() > time_limit:
                        return count
        return count

    # Strategy: Try multiple scales, starting with smaller L to get quick results,
    # then expand to larger L if time permits.
    
    # Phase 1: Quick search with moderate L
    L1 = 2 * 10**7
    if time.time() > deadline - 80.0:
        # Very tight on time, use small L
        L1 = 5 * 10**6
    
    powerful1, powerful_set1 = generate_powerful(L1)
    n_p1 = len(powerful1)
    
    # Count pairs for L1
    time_limit1 = deadline - 40.0
    count1 = count_pairs(powerful1, powerful_set1, time_limit=time_limit1)
    
    if not count1:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(count1, key=count1.get)
    best_count = count1[best_N]
    
    # Collect pairs for best_N
    best_pairs = []
    for a in powerful1:
        b = best_N - a
        if b >= a and b in powerful_set1:
            best_pairs.append([a, b])
    
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: If we have time, try a larger L
    remaining_time = deadline - 30.0 - time.time()
    if remaining_time > 20.0:
        # Try a larger L
        L2 = min(10**8, int(L1 * 3))
        if L2 > L1:
            powerful2, powerful_set2 = generate_powerful(L2)
            n_p2 = len(powerful2)
            
            # Only do this if n_p2 is manageable
            # Estimate: O(n_p2^2 / 2) operations, each ~100ns => n_p2^2 * 50ns
            # Time available: remaining_time - 15 (buffer)
            time_available = remaining_time - 15.0
            # Rough estimate: n_p2^2 * 50e-9 <= time_available
            # n_p2 <= sqrt(time_available / 50e-9)
            max_n_p = int((time_available / 50e-9) ** 0.5)
            
            if n_p2 <= max_n_p:
                time_limit2 = deadline - 15.0
                count2 = count_pairs(powerful2, powerful_set2, time_limit=time_limit2)
                
                if count2:
                    candidate_N = max(count2, key=count2.get)
                    candidate_count = count2[candidate_N]
                    
                    if candidate_count > best_count:
                        best_N = candidate_N
                        best_count = candidate_count
                        best_pairs = []
                        for a in powerful2:
                            b = best_N - a
                            if b >= a and b in powerful_set2:
                                best_pairs.append([a, b])
    
    # Phase 3: If we still have time, try to refine with a different approach
    # Focus on finding N values that might have many representations
    remaining_time = deadline - 20.0 - time.time()
    if remaining_time > 10.0:
        # Try to find better N by looking at the distribution
        # Use the existing count1 to identify promising N ranges
        if len(count1) > 100:
            # Get top 10 N values by count
            top_ns = sorted(count1.items(), key=lambda x: -x[1])[:10]
            
            # For each top N, verify and potentially find more pairs
            # using a larger powerful set if needed
            for N_candidate, c in top_ns:
                if c <= best_count:
                    continue
                # This N might have more representations with a larger set
                # Check if we can find more pairs
                # Use powerful1 for now
                new_pairs = []
                for a in powerful1:
                    b = N_candidate - a
                    if b >= a and b in powerful_set1:
                        new_pairs.append([a, b])
                if len(new_pairs) > best_count:
                    best_N = N_candidate
                    best_count = len(new_pairs)
                    best_pairs = new_pairs
    
    # Final verification
    if not best_pairs:
        best_N = 2
        best_pairs = [[1, 1]]
        best_count = 1
    
    # Sort pairs for consistency
    best_pairs.sort()
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END

import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130  # safety margin from 150s limit

    # Initial best
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    # Strategy: Enumerate k up to a limit, compute k*sigma(k), find collisions.
    # Use a segmented sieve approach for sigma to handle larger N efficiently.
    # Actually, for N up to ~50M, a linear sieve or simple SPF approach works.
    # Let's try to go up to 50M or 100M if time permits.

    # We'll use a dictionary to store M -> list of k
    # To save memory, we only store k for M values that appear more than once?
    # No, we need to find the max count. We can store M -> count and M -> list of k (only if count >= 2)
    # But we don't know the count until we finish.
    # Let's use M -> list of k, but prune lists that are too long? No, max is 1000.
    # Memory: 100M entries of (M, k) is too much.
    # Better: M -> count, and M -> k (only for the first few? No, we need all to return).
    # Wait, if we find a new max count, we need to update the list.
    # We can store:
    #   m_count: dict M -> count
    #   m_k_list: dict M -> list of k (only for M that are "candidates" for best, i.e., count >= current_best_count)
    # This is tricky.
    # Alternative: Just store M -> list of k for ALL M.
    # For N=50M, there are 50M values of M. Many are unique.
    # Most M will have count 1. Storing 50M lists is expensive.
    # Let's use:
    #   m_first_k: dict M -> first k
    #   m_count: dict M -> count
    #   m_k_list: dict M -> list of k (only if count >= 2)
    # When we see a new k for M:
    #   if M not in m_count:
    #       m_count[M] = 1
    #       m_first_k[M] = k
    #   elif m_count[M] == 1:
    #       m_count[M] = 2
    #       m_k_list[M] = [m_first_k[M], k]
    #   else:
    #       m_count[M] += 1
    #       m_k_list[M].append(k)
    #       if len(m_k_list[M]) > 1000: # cap to save memory/time
    #           # We can't easily remove from dict, but we can stop appending
    #           pass

    # This is much more memory efficient.

    def compute_sigma_sieve(N):
        """
        Compute k * sigma(k) for k=1..N using a sieve-like approach.
        Returns a generator of (k, M) pairs? Or a list?
        A list of N tuples is ~400MB for N=50M. Might be too much.
        Let's yield (k, M) pairs.
        """
        # sigma(k) = sum_{d|k} d
        # We want to compute M_k = k * sigma(k).
        # Note that sigma(k) = sum_{d|k} d.
        # So M_k = k * sum_{d|k} d = sum_{d|k} k*d.
        # Let n = k*d. Then d = n/k. This is not a simple divisor sum.
        
        # Alternative:
        # sigma(k) can be computed via SPF.
        # Let's use the SPF method with arrays, but process in chunks to save memory?
        # Or just use the SPF method for the whole range if N is small enough.
        # For N=50M, spf array is 50M * 8 bytes = 400MB. sigma array is 400MB.
        # Total 800MB. Might be tight but possible.
        # Let's try N=20M first to be safe, then try to increase if time allows.
        
        # Actually, the current program tries up to 100M but might run out of memory.
        # Let's implement a more memory efficient sigma computation.
        # We can compute sigma(k) using the recurrence:
        # sigma(k) = sigma(k / p^a) * sigma(p^a) where p^a is the highest power of p dividing k.
        # We need spf[k] and the p^a part.
        
        # Let's stick to the SPF method but be careful with memory.
        # Use int32 for spf? Max N=100M fits in int32 (2^31 ~ 2.1B).
        # sigma(k) can be large. k*sigma(k) can be up to 10^18.
        # sigma(k) for k=10^7 is roughly k * log log k ~ 10^7 * 3 = 3*10^7.
        # So sigma fits in int64 easily.
        
        # Let's use numpy arrays for speed.
        import numpy as np
        
        spf = np.zeros(N + 1, dtype=np.int32)
        spf[1] = 1
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                if i * i <= N:
                    for j in range(i * i, N + 1, i):
                        if spf[j] == 0:
                            spf[j] = i
        
        # This sieve is O(N log log N) but the inner loop is slow in Python.
        # The current program uses this and it's slow.
        # Let's use a linear sieve for spf.
        
        spf = np.zeros(N + 1, dtype=np.int32)
        primes = []
        spf[1] = 1
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
        
        # Now compute sigma
        # sigma[1] = 1
        # For k > 1:
        #   p = spf[k]
        #   m = k // p
        #   if m % p == 0:
        #       # k = p * m, and p|m
        #       # Let k = p^a * r. Then m = p^{a-1} * r.
        #       # We need to track the p-power part.
        #       # Let's store:
        #       #   pp[k] = p^a (highest power of spf[k] dividing k)
        #       #   sig_pp[k] = sigma(p^a)
        #       #   rest[k] = k / pp[k]
        #       # Then sigma[k] = sig_pp[k] * sigma[rest[k]]
        
        pp = np.zeros(N + 1, dtype=np.int64)
        sig_pp = np.zeros(N + 1, dtype=np.int64)
        rest = np.zeros(N + 1, dtype=np.int64)
        sigma = np.zeros(N + 1, dtype=np.int64)
        sigma[1] = 1
        
        for k in range(2, N + 1):
            p = spf[k]
            m = k // p
            if m % p == 0:
                pp[k] = pp[m] * p
                sig_pp[k] = sig_pp[m] + pp[k]
                rest[k] = rest[m]
            else:
                pp[k] = p
                sig_pp[k] = 1 + p
                rest[k] = m
            
            if rest[k] > 1:
                sigma[k] = sig_pp[k] * sigma[rest[k]]
            else:
                sigma[k] = sig_pp[k]
        
        # Now compute M = k * sigma(k) and yield
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                yield k, int(M)

    # We will try increasing N
    # N=10M, 20M, 50M
    target_ns = [10_000_000, 20_000_000, 50_000_000]
    
    for N in target_ns:
        if time.time() > deadline:
            break
        
        try:
            m_count = {}
            m_k_list = {}
            
            # Process in chunks to avoid holding the entire sigma array?
            # No, we need the spf and sigma arrays for the whole N.
            # Let's just compute and iterate.
            
            # The generator approach is good.
            for k, M in compute_sigma_sieve(N):
                if M not in m_count:
                    m_count[M] = 1
                    # Don't store list yet
                elif m_count[M] == 1:
                    m_count[M] = 2
                    # We need the first k. We didn't store it.
                    # This is a problem. We need to store the first k.
                    pass
            
            # Let's redo the storage logic.
            # m_first_k: dict M -> k
            # m_count: dict M -> count
            # m_k_list: dict M -> list of k (if count >= 2)
            
            m_first_k = {}
            m_count = {}
            m_k_list = {}
            
            for k, M in compute_sigma_sieve(N):
                if M not in m_count:
                    m_count[M] = 1
                    m_first_k[M] = k
                elif m_count[M] == 1:
                    m_count[M] = 2
                    m_k_list[M] = [m_first_k[M], k]
                    if m_count[M] > best_count:
                        best_count = m_count[M]
                        best_M = M
                        best_k_list = list(m_k_list[M])
                else:
                    m_count[M] += 1
                    m_k_list[M].append(k)
                    if m_count[M] > best_count:
                        best_count = m_count[M]
                        best_M = M
                        best_k_list = list(m_k_list[M])
                
                # Check time periodically
                if k % 1_000_000 == 0:
                    if time.time() > deadline:
                        break
            
            if time.time() > deadline:
                break
                
        except MemoryError:
            break
        except Exception as e:
            # print(e)
            break

    # Ensure types are correct
    # best_k_list might be a numpy array or list. Ensure it's a list of ints.
    best_k_list = [int(x) for x in best_k_list]
    
    return {'M': int(best_M), 'k': best_k_list}

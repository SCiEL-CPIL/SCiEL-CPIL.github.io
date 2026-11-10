import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 512
    
    # Generate odd numbers up to n
    odds = [i for i in range(1, n + 1) if i % 2 == 1]
    
    def evaluate_odd_set(O):
        """
        Given a set of odd numbers O, determine the max power K_o for each o in O
        such that the set of all o * 2^k (0 <= k <= K_o) is valid.
        Returns the list of integers and its size.
        """
        if not O:
            return [], 0
        
        # Precompute max possible K for each o based on o * 2^K <= n
        max_K_by_value = {}
        for o in O:
            k = 0
            while o * (1 << (k + 1)) <= n:
                k += 1
            max_K_by_value[o] = k
            
        # Precompute M_oo' = floor(log2(n / lcm(o, o')))
        # If n / lcm < 1, M is -1 (incompatible)
        M_matrix = {}
        for o1 in O:
            for o2 in O:
                if o1 == o2:
                    continue
                l = (o1 // gcd(o1, o2)) * o2
                if l > n:
                    M_matrix[(o1, o2)] = -1
                else:
                    # Find max k such that 2^k <= n / l
                    val = n // l
                    k = 0
                    while (1 << (k + 1)) <= val:
                        k += 1
                    M_matrix[(o1, o2)] = k
                    
        # Determine K_o for each o
        # K_o <= max_K_by_value[o]
        # K_o <= M_matrix[(o, o')] for all o' in O, o' != o
        K = {}
        for o in O:
            K[o] = max_K_by_value[o]
            for o2 in O:
                if o == o2:
                    continue
                m_val = M_matrix.get((o, o2), -1)
                if m_val < K[o]:
                    K[o] = m_val
                if K[o] < 0:
                    break
            
        # If any K is negative, the set O is invalid (conflict)
        # But we can just take the min, if min is negative, it means no k works?
        # If K[o] < 0, it means we can't even include o * 2^0?
        # Wait, if M is -1, it means lcm > n, so even k=0 fails.
        # So if K[o] < 0, the set is invalid.
        
        valid = True
        for o in O:
            if K[o] < 0:
                valid = False
                break
        
        if not valid:
            return [], 0
        
        # Construct the list
        result = []
        total_size = 0
        for o in O:
            for k in range(K[o] + 1):
                result.append(o * (1 << k))
                total_size += 1
                
        return result, total_size

    # Candidate sets of odd numbers
    # We try increasing sets of small odd numbers
    candidates = []
    
    # Single odd
    candidates.append([1])
    candidates.append([3])
    candidates.append([5])
    
    # Pairs
    candidates.append([1, 3])
    candidates.append([1, 5])
    candidates.append([1, 7])
    candidates.append([3, 5])
    
    # Triples
    candidates.append([1, 3, 5])
    candidates.append([1, 3, 7])
    candidates.append([1, 5, 7])
    
    # Quads
    candidates.append([1, 3, 5, 7])
    candidates.append([1, 3, 5, 9])
    candidates.append([1, 3, 7, 9])
    
    # Quints
    candidates.append([1, 3, 5, 7, 9])
    candidates.append([1, 3, 5, 7, 11])
    
    # Hexes
    candidates.append([1, 3, 5, 7, 9, 11])
    candidates.append([1, 3, 5, 7, 9, 13])
    
    best_result = []
    best_size = 0
    
    for O in candidates:
        res, size = evaluate_odd_set(O)
        if size > best_size:
            best_size = size
            best_result = res
            
        if time.time() > deadline - 5:
            break

    # If the structural search didn't find anything or found something small,
    # fall back to powers of 2
    if best_size < 10:
        best_result = [1 << i for i in range(10)]
        best_size = 10

    return best_result

import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Precompute parity masks
    masks = [0] * (max_val + 1)
    for p in primes:
        p_bit = 1 << prime_index[p]
        for multiple in range(p, max_val + 1, p):
            cnt = 0
            t = multiple
            while t % p == 0:
                cnt += 1
                t //= p
            if cnt % 2 == 1:
                masks[multiple] ^= p_bit

    # Parity mask of 210
    target = masks[210]

    def verify_solution(sol):
        """Verify that product of sol * 210 is a perfect square."""
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        return s * s == prod

    def find_short_solution(M, max_size=100):
        """
        Find a subset of {211, ..., M} whose XOR of masks equals target,
        with at most max_size elements. Returns None if not found.
        """
        # Build basis with history tracking
        basis_vec = {}   # pivot -> vector
        basis_hist = {}  # pivot -> history mask (bit i set means element 211+i is in the combination)
        
        for idx in range(M - 210):
            M2 = 211 + idx
            v = masks[M2]
            h = 1 << idx
            tv = v
            th = h
            while tv:
                p = tv.bit_length() - 1
                if p in basis_vec:
                    tv ^= basis_vec[p]
                    th ^= basis_hist[p]
                else:
                    basis_vec[p] = tv
                    basis_hist[p] = th
                    break
        
        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis_vec:
                tv ^= basis_vec[p]
                th ^= basis_hist[p]
            else:
                return None

        if tv != 0:
            return None

        # Extract initial solution
        sol_set = set()
        hh = th
        while hh:
            lsb = hh & -hh
            idx = lsb.bit_length() - 1
            sol_set.add(211 + idx)
            hh ^= lsb

        if len(sol_set) <= max_size:
            sol = list(sol_set)
            if verify_solution(sol):
                return sol

        # Try to reduce solution size using kernel vectors
        # Find kernel vectors: subsets with XOR = 0
        # A kernel vector corresponds to a dependency in the basis
        
        # Build all vectors to find dependencies
        # For each element, we can express it in terms of the basis
        # A kernel vector is a combination that gives XOR = 0
        
        # Alternative: try to find small subsets that cancel out
        # If mask[a] == mask[b] for a != b, then {a, b} is a kernel vector
        # If mask[a] ^ mask[b] == mask[c], then {a, b, c} is a kernel vector
        
        # Find duplicate masks
        mask_to_elems = {}
        for e in range(211, M + 1):
            m = masks[e]
            if m not in mask_to_elems:
                mask_to_elems[m] = []
            mask_to_elems[m].append(e)
        
        # Try removing elements using kernel vectors
        # For each kernel vector K, if |sol_set ∩ K| is odd, we can toggle
        # (remove elements in sol_set ∩ K, add elements in K \ sol_set)
        # This changes the solution by K, which preserves the XOR
        
        # First, find all small kernel vectors (size 2 and 3)
        kernel_vectors = []
        
        # Size 2: pairs with same mask
        for m, elems in mask_to_elems.items():
            if m == 0:
                # Single element with mask 0 is a kernel vector of size 1
                for e in elems:
                    kernel_vectors.append([e])
            elif len(elems) >= 2:
                for i in range(len(elems)):
                    for j in range(i + 1, len(elems)):
                        kernel_vectors.append([elems[i], elems[j]])
        
        # Size 3: triples where mask[a] ^ mask[b] == mask[c]
        # This is expensive, so only do for small M
        if M - 210 <= 50:
            mask_to_elem = {}
            for e in range(211, M + 1):
                m = masks[e]
                if m not in mask_to_elem:
                    mask_to_elem[m] = e
            
            for a in range(211, M + 1):
                for b in range(a + 1, M + 1):
                    xor_m = masks[a] ^ masks[b]
                    if xor_m in mask_to_elem:
                        c = mask_to_elem[xor_m]
                        if c > b:
                            kernel_vectors.append([a, b, c])
        
        # Try to reduce solution size using kernel vectors
        # Greedily apply kernel vectors that reduce the solution size
        improved = True
        while improved:
            improved = False
            for K in kernel_vectors:
                K_set = set(K)
                # Count elements in sol_set that are in K
                overlap = sol_set & K_set
                # If we toggle, we remove overlap and add K \ sol_set
                # New size = len(sol_set) - len(overlap) + len(K \ sol_set)
                # = len(sol_set) - 2*len(overlap) + len(K)
                new_size = len(sol_set) - 2 * len(overlap) + len(K)
                if new_size < len(sol_set) and new_size <= max_size:
                    # Apply the toggle
                    sol_set -= K_set
                    sol_set |= (K_set - sol_set)  # This is wrong, let me fix
                    # Actually: new_sol = (sol_set - K_set) | (K_set - sol_set)
                    # = (sol_set ^ K_set)
                    sol_set = sol_set ^ K_set
                    improved = True
                    break
        
        if len(sol_set) <= max_size:
            sol = list(sol_set)
            if verify_solution(sol):
                return sol
        
        return None

    # Main search: find minimum M where a valid solution exists
    best_sol = None
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        
        sol = find_short_solution(M)
        if sol is not None:
            return [int(x) for x in sol]

    # Fallback
    return [840]

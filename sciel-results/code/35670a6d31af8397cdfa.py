import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit

    min_val = 211
    max_val = 10210

    # Target: product of selected * 210 must be a perfect square
    # 210 = 2 * 3 * 5 * 7, so the XOR target for parity signatures is {2,3,5,7}
    # We want to minimize max(selected integers) - 210.
    
    # Strategy: Find the smallest M such that there exists a set of integers <= M
    # whose signatures XOR to the target.
    # This is equivalent to finding the minimum bottleneck path in a graph where
    # nodes are signatures, edges are integers, and edge weight is the integer value.
    # We want to connect the 'zero' signature to the 'target' signature.
    
    # 1. Precompute square-free signatures for all integers in range.
    # We use a sieve-like approach to compute signatures efficiently.
    # sig[n] is a frozenset of primes with odd exponent in n.
    
    # Since max_val is 10210, we can just do trial division or a sieve.
    # A sieve for signatures is faster.
    
    # Initialize signatures
    sigs = [None] * (max_val + 1)
    sigs[0] = frozenset()
    sigs[1] = frozenset()
    
    # We need primes up to max_val.
    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
            
    # Compute signatures using smallest prime factor (SPF) logic or direct division.
    # Direct division is fine for 10k numbers.
    # But we can do it smarter: sig[n] = sig[n / p] ^ {p} if p divides n and (n/p) is not divisible by p?
    # Actually, let's just use a simple loop for each number, it's fast enough.
    
    # Optimization: Precompute for all n in [min_val, max_val]
    # To make it faster, we can compute sig for all n up to max_val.
    
    # Let's use a list to store signatures as integers (bitmasks) for speed?
    # Primes are up to 10210. There are ~1200 primes.
    # A bitmask would need 1200 bits. Python integers are arbitrary precision, so this is fine.
    # Let's map primes to indices.
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # sig_mask[n] is an integer bitmask.
    sig_mask = [0] * (max_val + 1)
    sig_mask[0] = 0
    sig_mask[1] = 0
    
    # Calculate sig_mask for all n
    # We can use a modified sieve.
    # For each prime p, for each multiple m of p, we need to toggle the bit for p
    # based on the exponent of p in m.
    
    # This is slightly complex to do in a single pass sieve for parity.
    # Let's stick to trial division for each n, but optimized.
    # Or, we can compute sig_mask[n] recursively:
    # Let spf[n] be smallest prime factor.
    # sig_mask[n] = sig_mask[n // spf[n]] ^ (1 << prime_to_idx[spf[n]]) if (n // spf[n]) % spf[n] != 0
    # else sig_mask[n // (spf[n]**2)] (since p^2 doesn't change parity)
    
    # Let's compute SPF first.
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for n in range(2, max_val + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p^2 divides n, so parity of p doesn't change from n/p^2
            sig_mask[n] = sig_mask[m // p]
        else:
            # p divides n exactly once (relative to p^2), so toggle bit
            sig_mask[n] = sig_mask[m] ^ (1 << prime_to_idx[p])
            
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target_mask |= (1 << prime_to_idx[p])
        else:
            # Should not happen
            pass

    # Now we have sig_mask for all numbers.
    # We want to find a subset of numbers from [min_val, max_val] such that XOR of their masks is target_mask.
    # And we want to minimize max(number in subset).
    
    # This is a shortest path problem in the state space of masks.
    # Nodes: all possible masks (2^1200 is too big, but reachable masks are limited).
    # Actually, the number of distinct masks is at most the number of integers.
    # Let's use Dijkstra's algorithm where:
    # Node: current_mask
    # Start: 0
    # Goal: target_mask
    # Edges: For each number x in [min_val, max_val], we can transition from mask m to m ^ sig_mask[x] with cost x.
    # But we want to minimize the MAX cost in the path, not the sum.
    # This is the "Minimax Path" problem.
    # We can solve this by binary searching the maximum allowed value M, and checking if target_mask is reachable using only numbers <= M.
    # Or, we can use a modified Dijkstra/BFS where the "distance" is the maximum edge weight encountered.
    
    # Let's use the binary search approach as it's robust.
    # Range of M: [min_val, max_val]
    
    # To check if target is reachable with numbers <= M:
    # We have a set of available masks: {sig_mask[x] for x in [min_val, M]}
    # We want to know if target_mask can be formed by XOR-ing a subset of these masks.
    # This is equivalent to checking if target_mask is in the span of the available masks over GF(2).
    # We can compute the linear span (basis) of the available masks.
    
    def is_reachable(M):
        # Build basis of masks for x in [min_val, M]
        basis = []
        for x in range(min_val, M + 1):
            mask = sig_mask[x]
            # Reduce mask by existing basis
            # Standard Gaussian elimination over GF(2)
            m = mask
            for b in basis:
                m = min(m, m ^ b)  # This is not quite right for basis reduction, let's do it properly
                # Actually, standard reduction:
                # if m has the highest bit of b set, m ^= b
            
            # Let's implement a proper basis insertion.
            # We want to maintain a basis in row-echelon form (pivot bits).
            # Actually, for checking membership, we just need to reduce the target vector by the basis.
            # Let's collect all masks first, then build basis.
            
        # Better: Build basis once for a given M.
        basis = []
        for x in range(min_val, M + 1):
            mask = sig_mask[x]
            # Try to add mask to basis
            m = mask
            for b in basis:
                # Find pivot of b
                # We assume basis is sorted by pivot bit descending? Or we just iterate.
                # Let's keep basis as a list of integers.
                # To reduce m by b, we check if m has the same highest set bit as b?
                # No, we check if the pivot bit of b is set in m.
                
                # Let's define pivot(b) as the position of the highest set bit.
                # This is a bit slow if done repeatedly.
                
                # Alternative: Just store the masks and do Gaussian elimination on the set.
                pass
        
        # Let's re-implement is_reachable more efficiently.
        # We can maintain a global basis as we increase M?
        # No, binary search jumps around.
        
        # Let's just do Gaussian elimination on the list of masks for [min_val, M].
        # Number of masks is at most 10000.
        # Basis size is at most number of primes (~1200).
        # Elimination is O(N * BasisSize). 10000 * 1200 = 12M ops. Fast enough.
        
        masks = [sig_mask[x] for x in range(min_val, M + 1)]
        
        # Build basis
        basis = []
        for m in masks:
            # Reduce m by existing basis
            # We want to keep basis in a form where each vector has a unique pivot.
            # Let's sort basis by pivot bit (highest set bit) descending.
            # Or just iterate.
            
            curr = m
            for b in basis:
                # Find pivot of b
                # pivot is the index of the highest set bit.
                # We can precompute pivots?
                # Let's just use bit_length.
                pb = b.bit_length() - 1
                pc = curr.bit_length() - 1
                if pb == pc:
                    curr ^= b
                # If pb < pc, we don't reduce.
                # If pb > pc, we don't reduce (b has higher bit).
                # Wait, this logic is flawed if basis is not sorted.
                
            # Let's do a standard insertion into a linear basis.
            # We'll maintain `basis` as a list.
            # To insert `curr`:
            # For each bit from high to low:
            #   if bit is set in curr:
            #     if basis[bit] exists: curr ^= basis[bit]
            #     else: basis[bit] = curr; break
            
            # This requires an array of size MaxBit.
            # MaxBit is ~1200.
            
            pass
        
        # Let's use the array-based basis.
        max_bit = 1200 # Safe upper bound
        basis_arr = [0] * (max_bit + 1)
        
        for x in range(min_val, M + 1):
            m = sig_mask[x]
            # Insert m into basis_arr
            for bit in range(max_bit, -1, -1):
                if (m >> bit) & 1:
                    if basis_arr[bit]:
                        m ^= basis_arr[bit]
                    else:
                        basis_arr[bit] = m
                        break
        
        # Now check if target_mask is in the span.
        # Reduce target_mask by basis_arr.
        t = target_mask
        for bit in range(max_bit, -1, -1):
            if (t >> bit) & 1:
                if basis_arr[bit]:
                    t ^= basis_arr[bit]
                else:
                    return False
        return t == 0

    # Binary search for minimum M
    lo = min_val
    hi = max_val
    best_M = max_val
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        if is_reachable(mid):
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Now we know the minimum max value is best_M.
    # We need to construct the actual set of integers.
    # We need to find a subset of integers in [min_val, best_M] that XORs to target_mask.
    
    # We can use the basis to find one such combination.
    # Re-build basis for M = best_M, keeping track of which integer generated each basis vector.
    
    max_bit = 1200
    basis_arr = [0] * (max_bit + 1)
    basis_source = [None] * (max_bit + 1) # Which integer created this basis vector?
    # Actually, to reconstruct the set, we need to track the XOR of sources.
    # Let's store for each basis vector the mask of integers (as a set or bitmask of indices) that formed it.
    # Since we only care about the final set, we can store the set of integers.
    
    # However, the set might be large. We just need ANY valid set.
    # Let's store the list of integers.
    
    basis_vectors = [] # List of (mask, set_of_ints)
    
    for x in range(min_val, best_M + 1):
        m = sig_mask[x]
        current_set = {x}
        
        # Try to reduce
        reduced = False
        for i, (b_mask, b_set) in enumerate(basis_vectors):
            # Find pivot of b_mask
            pb = b_mask.bit_length() - 1
            pc = m.bit_length() - 1
            if pb == pc:
                m ^= b_mask
                current_set = current_set.symmetric_difference(b_set)
                # If m becomes 0, we have a dependency, but we just want to insert if independent.
                # If m becomes 0, it's dependent, so we don't add it.
                # We continue reducing.
        
        # Check if m is 0
        if m == 0:
            continue
            
        # Find pivot of m
        pm = m.bit_length() - 1
        
        # Check if there is already a vector with pivot pm
        # Our basis_vectors list is not sorted by pivot.
        # Let's just check all.
        exists = False
        for j, (b_mask, b_set) in enumerate(basis_vectors):
            if b_mask.bit_length() - 1 == pm:
                # This shouldn't happen if we reduced correctly?
                # Actually, the reduction loop above doesn't guarantee m is reduced by ALL vectors.
                # It only reduces by those with same pivot.
                # We need to reduce by all vectors with pivot <= pm? No, standard Gaussian elimination.
                
                # Let's restart the reconstruction with a proper basis structure.
                pass

    # Let's do a cleaner reconstruction.
    # We know best_M is the answer.
    # We want to find a subset of {sig_mask[x] for x in [min_val, best_M]} that sums to target_mask.
    
    # We can use the basis we built in `is_reachable` but we need to trace back.
    # Let's rebuild the basis for M=best_M, storing the "recipe" for each basis vector.
    # Recipe: A set of integers whose signatures XOR to the basis vector.
    
    basis = [] # List of (mask, set_of_ints)
    
    for x in range(min_val, best_M + 1):
        m = sig_mask[x]
        s = {x}
        
        # Reduce m by existing basis
        for i in range(len(basis)):
            b_mask, b_set = basis[i]
            # If the highest set bit of m is the same as b_mask, reduce
            if m == 0:
                break
            if b_mask == 0:
                continue
            
            # Find highest set bit
            # Optimization: Sort basis by highest bit descending?
            # Let's just check if the highest bit of m is set in b_mask.
            # Actually, if we maintain basis in row-echelon form (each basis vector has a unique pivot, and pivots are distinct),
            # we can check if the pivot of m is in the basis.
            
            # Let's assume basis is maintained such that basis[i] has pivot p_i, and p_i are distinct.
            # To reduce m:
            # Find pivot of m (pm).
            # If there is a basis vector with pivot pm, m ^= b_mask, s ^= b_set.
            
            # We need to find if there's a basis vector with pivot pm.
            # We can maintain a dictionary pivot_to_index.
            
            pass
            
    # This is getting complicated. Let's use a simpler approach for reconstruction.
    # Since best_M is small (likely close to 211), the number of candidates is small.
    # We can just do a BFS/DFS on the masks?
    # No, the number of candidates is (best_M - 210). If best_M is 212, it's 2.
    # If best_M is 220, it's 10.
    # We can use a meet-in-the-middle or just simple DP if the range is small.
    
    # Let's check the size of the range.
    num_candidates = best_M - min_val + 1
    
    if num_candidates <= 20:
        # Brute force or simple DP
        # We can use a dictionary: mask -> set of ints
        # Start with {0: set()}
        # For each x, update the dictionary.
        
        states = {0: set()}
        for x in range(min_val, best_M + 1):
            m = sig_mask[x]
            new_states = {}
            for mask, s in states.items():
                new_mask = mask ^ m
                new_s = s.union({x})
                if new_mask not in new_states:
                    new_states[new_mask] = new_s
                else:
                    # Keep the smaller set? Or just any set.
                    # We want to minimize the set? No, we just need a valid one.
                    # But we want to ensure we don't lose the solution.
                    # Actually, we want to find IF target is reachable.
                    # We can just store one set per mask.
                    pass
            
            # Merge states
            for mask, s in new_states.items():
                if mask not in states:
                    states[mask] = s
                # If mask already exists, we can keep the old one or the new one.
                # It doesn't matter for existence.
            
        if target_mask in states:
            return list(states[target_mask])
        else:
            # Fallback
            return [840]
    else:
        # Use the basis method properly.
        # Rebuild basis with recipes.
        
        # We need a data structure that allows fast reduction.
        # Let's use the array-based basis again, but store the recipe.
        
        max_bit = 1200
        basis_mask = [0] * (max_bit + 1)
        basis_recipe = [None] * (max_bit + 1) # Set of integers
        
        for x in range(min_val, best_M + 1):
            m = sig_mask[x]
            recipe = {x}
            
            # Reduce m
            for bit in range(max_bit, -1, -1):
                if (m >> bit) & 1:
                    if basis_mask[bit]:
                        m ^= basis_mask[bit]
                        # Symmetric difference of recipes
                        # Since sets can be large, this might be slow.
                        # But best_M is small, so recipes are small.
                        recipe = recipe.symmetric_difference(basis_recipe[bit])
                    else:
                        basis_mask[bit] = m
                        basis_recipe[bit] = recipe
                        break
            
        # Now reduce target_mask
        t = target_mask
        result_set = set()
        
        for bit in range(max_bit, -1, -1):
            if (t >> bit) & 1:
                if basis_mask[bit]:
                    t ^= basis_mask[bit]
                    result_set = result_set.symmetric_difference(basis_recipe[bit])
                else:
                    # Should not happen if is_reachable worked
                    return [840]
                    
        if t == 0:
            return list(result_set)
        else:
            return [840]

    # Fallback if something went wrong
    return [840]

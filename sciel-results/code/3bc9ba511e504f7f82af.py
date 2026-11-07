import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    x = math.isqrt(n)
    return x * x == n

def _square_free_kernel(n):
    """
    Returns the square-free part of n.
    i.e., the product of primes that appear with an odd exponent in n's factorization.
    """
    res = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            if count % 2 == 1:
                res *= d
        d += 1
    if n > 1:
        res *= n
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe within 60s wall clock
    
    # Target: 210 * prod(selected) must be a perfect square.
    # Let K(x) be the square-free kernel.
    # K(210 * prod(S)) = K(210) * prod(K(s) for s in S) mod squares = 1
    # This is equivalent to: prod(K(s) for s in S) == K(210)
    # Because K(a)*K(b) = K(ab) if we consider the kernel as a vector in GF(2)^primes.
    # Actually, simpler: K(a) is the square-free part. 
    # K(a*b) = K(K(a)*K(b)).
    # We want K(210 * prod(S)) = 1.
    # This means 210 * prod(S) = k^2.
    # Taking square-free kernels: K(210) * prod(K(s)) must be a square (i.e., kernel 1).
    # So prod(K(s)) must have the same square-free kernel as 210.
    # K(210) = 210 (since 210 is square-free: 2*3*5*7).
    
    target_kernel = 210
    
    # We need to select 1..100 distinct integers x_i such that 210 < x_i <= 10210.
    # Minimize max(x_i) - 210.
    # This is equivalent to finding the smallest M such that there exists a subset of
    # {211, ..., 210+M} that satisfies the kernel product condition.
    
    # Strategy:
    # Iterate M from 1 upwards.
    # For each M, consider the set of candidates C_M = {211, ..., 210+M}.
    # We need a subset S of C_M such that prod(K(s) for s in S) == 210.
    # Note: We can use up to 100 numbers.
    # Since we want to minimize M, we start small.
    
    # Optimization:
    # The number of distinct square-free kernels in a small range is limited.
    # We can group numbers by their kernel.
    # If we have multiple numbers with the same kernel, we only need one of them (the smallest)
    # to minimize the max? No, we are fixing the max M. We just need *some* subset.
    # Actually, if we are checking if a solution exists for max <= 210+M, we can pick
    # the smallest numbers available for each kernel to keep the "cost" low? 
    # No, the cost is just the max. If a solution exists using numbers <= 210+M, 
    # then M is a candidate. We don't care about the sum or individual values other than max.
    # So for a fixed M, we just need to know if there is a subset of {211..210+M} 
    # whose kernels multiply to 210.
    
    # We can represent the kernel as a bitmask or just the integer itself.
    # Since kernels are square-free, they are unique for each "type" of parity vector.
    # We can use a dictionary: kernel -> count of available numbers with that kernel.
    # But we need to select *distinct* integers. 
    # If we have 5 numbers with kernel K, we can use 0 or 1 of them in the product 
    # (using 2 would cancel them out to 1, which is equivalent to not using them 
    # in terms of the final product parity, but consumes slots. However, since we 
    # just need the product to be 210, adding pairs of same kernels doesn't change 
    # the product but uses up the 100 limit. It's never beneficial to use >1 of same 
    # kernel unless we are forced to fill up to 100? No, we need 1..100. 
    # So we can treat this as: does there exist a subset of distinct kernels 
    # (with multiplicity <= count_available) that multiplies to 210?
    # Since kernels are square-free, if we pick the same kernel twice, they cancel.
    # So effectively, we are looking for a subset of *distinct* kernels K1, K2, ... 
    # such that K1*K2*... = 210.
    # And for each Ki in the subset, we must have at least one number in our range 
    # with that kernel.
    
    # So the problem reduces to:
    # 1. Find all square-free kernels present in {211..210+M}.
    # 2. Is there a subset of these kernels that multiplies to 210?
    
    # Note: 210 = 2 * 3 * 5 * 7.
    # The kernels must be composed of primes that appear in 210? 
    # Not necessarily. If we have a kernel with prime 11, say 11, we need another 11 
    # to cancel it, OR we need the target to have 11? 
    # Wait. Target is 210. 
    # If we use a kernel with prime 11, say 11, the product has 11. 
    # To get 210 (which has no 11), we must cancel the 11. 
    # So we must use another number with kernel 11 (or 11*something). 
    # Basically, any prime not in 210 must appear an even number of times in the 
    # selected kernels.
    # Primes in 210 (2,3,5,7) must appear an odd number of times total.
    
    # This is a subset product problem over the multiplicative group of square-free integers.
    # Given the small range of M (we hope it's small), the number of distinct kernels 
    # is small. We can use BFS/DFS on the state space of "current product kernel".
    # State: current product P (square-free). Start: 1. Target: 210.
    # Available moves: multiply by K_i (a kernel present in the range).
    # Constraint: We cannot use the same *number* twice. But we can use the same 
    # *kernel* twice? No, using the same kernel twice cancels it. 
    # If we have two numbers with kernel K, we can use both, effectively adding 1 (identity).
    # This doesn't help reach 210. It just wastes slots.
    # So we only care about using each *distinct kernel* at most once in the path.
    # If we have multiple numbers for a kernel, it just means that kernel is "available".
    
    # So:
    # 1. For a given M, collect the set of available kernels S_K.
    # 2. Check if 210 can be formed by the product of a subset of S_K.
    #    (Note: The product is in the group of square-free integers, so it's XOR of prime factors).
    
    # We can represent the state as the square-free integer itself.
    # Since we only care about primes up to some limit?
    # The kernels can have large primes. But if a prime p > 210 is in a kernel, 
    # it must be cancelled by another kernel with p.
    # This makes the state space potentially large if we include large primes.
    # However, if M is small, the numbers are small, so their kernels are small.
    
    # Let's iterate M.
    # For each M, update the set of available kernels.
    # Then run a subset product check.
    
    # Optimization for subset product check:
    # The target is 210.
    # We can map each kernel to a bitmask of its prime factors?
    # Primes involved: 2, 3, 5, 7 are the target primes.
    # Any other prime must be paired.
    # This is tricky if we have large primes.
    
    # Alternative: Meet-in-the-middle or just BFS on the product value.
    # Since 210 is small, maybe we can restrict attention to kernels that are 
    # "compatible" with 210?
    # If a kernel has a prime factor p not in 210, we MUST include another kernel 
    # with p.
    # This suggests we should group kernels by their "foreign" primes.
    
    # Simpler approach for small M:
    # Just use a set of reachable products.
    # Reachable starts with {1}.
    # For each kernel k in available_kernels:
    #   new_reachable = reachable | {r * k for r in reachable}
    #   (Note: r*k might not be square-free? No, r and k are square-free. 
    #    The product r*k is square-free iff gcd(r,k)=1. 
    #    If they share a prime, that prime cancels. 
    #    The result is still square-free. 
    #    Specifically, if r = prod p_i and k = prod p_j, 
    #    r*k = prod (p_i + p_j) mod 2. 
    #    This is equivalent to symmetric difference of prime sets.
    #    So we can represent the state as a frozenset of primes, or just the integer.
    #    Using the integer is fine as long as we handle the cancellation correctly.
    #    Wait, 12 * 18 = 216. K(12)=3, K(18)=2. 3*2=6. K(216)=6. Correct.
    #    So the operation is: new_state = (r * k) // (gcd(r, k) ** 2) ? No.
    #    If r and k are square-free, let g = gcd(r, k).
    #    r = g * r', k = g * k'. gcd(r', k') = 1.
    #    r * k = g^2 * r' * k'.
    #    The square-free part is r' * k'.
    #    So the operation is: xor(r, k) = (r * k) // (gcd(r, k) ** 2).
    #    Let's verify: r=6 (2,3), k=10 (2,5). gcd=2. r*k=60. 60/4=15 (3,5). Correct.
    
    # So we can maintain a set of integers representing reachable square-free products.
    # We want to know if 210 is in the set.
    
    # Complexity:
    # Number of distinct kernels in range [211, 210+M].
    # For small M, this is small.
    # The size of the reachable set can grow exponentially in the number of distinct kernels.
    # But we can prune: we only care about states that can potentially lead to 210.
    # Actually, just let it grow. If M is small (e.g., < 100), the number of distinct 
    # kernels is likely < 20. 2^20 is 1 million, which is manageable in Python if done 
    # efficiently? Maybe a bit slow.
    # We can stop early if 210 is found.
    
    # Let's try this.
    
    available_kernels = set()
    
    # Precompute kernels for numbers up to 10210? 
    # No, we iterate M. 
    # We can precompute the kernel for every number in [211, 10210] once.
    # This takes O(N sqrt(N)) which is fine for N=10000.
    
    max_val = 10210
    kernels = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        kernels[i] = _square_free_kernel(i)
    
    # Now iterate M
    # M is the offset. Max selected = 210 + M.
    # Candidates are 211 to 210+M.
    
    # We need to add kernels as M increases.
    # reachable = {1}
    # For each new number x added (as M increases), we add its kernel k.
    # But we need to handle the "distinct kernel" logic?
    # No, if we add a number with kernel k, and k is already in available_kernels, 
    # we don't change the set of *available* kernels. 
    # The subset product check depends only on the *set* of available kernels.
    # So we can maintain `available_kernels` set.
    # And whenever a *new* kernel is added, we update the `reachable` set.
    # If 210 is in `reachable`, we have a solution for this M.
    
    # Wait, is it possible that adding a *duplicate* kernel helps?
    # No, as argued before, using two numbers with the same kernel cancels them 
    # in the product (contributes 1). It doesn't help reach 210. 
    # It only consumes slots. Since we have 100 slots and likely need far fewer 
    # distinct kernels to reach 210, duplicates don't expand the reachable set.
    
    # So:
    # 1. Precompute kernels.
    # 2. Iterate M from 1 to 10000.
    # 3. Add kernel of (210+M) to `available_kernels`.
    # 4. If it's a new kernel, update `reachable` set.
    # 5. If 210 in `reachable`, we found the minimal M.
    # 6. Now we need to reconstruct the actual numbers.
    
    # Reconstructing:
    # We need to find a subset of `available_kernels` that multiplies to 210.
    # Then, for each kernel in that subset, pick the smallest number in [211, 210+M] 
    # that has that kernel.
    # This ensures distinctness (since kernels are distinct in the subset) and 
    # minimizes the numbers used (though max is fixed by M).
    
    # Let's implement the search for M.
    
    reachable = {1}
    # To reconstruct, we need to know *how* we reached 210.
    # We can store a parent pointer or just re-run the subset search once M is found.
    # Re-running is safer and simpler.
    
    found_M = None
    current_kernels = set()
    
    # We'll store the list of numbers added for each M to facilitate reconstruction?
    # No, we just need the set of available kernels for the final M.
    
    for M in range(1, max_val - 210 + 1):
        if time.time() > deadline:
            # Fallback: return a known valid solution if we have one, or break?
            # If we haven't found one, we must return something valid.
            # The seed returns [4*210] = [840]. 
            # Let's check if [840] is valid.
            # 210 * 840 = 176400. sqrt(176400) = 420. Yes.
            # So we can always fall back to [840] if we time out.
            # But we want the best M.
            # If we time out, we return the best M found so far? 
            # No, `found_M` is only set when we *find* a solution.
            # If we are still searching, we haven't found a solution for any M < current.
            # So we should fall back to the seed solution.
            return [840]
            
        x = 210 + M
        k = kernels[x]
        
        if k not in current_kernels:
            current_kernels.add(k)
            # Update reachable
            # New states are formed by multiplying existing states by k
            # We must be careful not to modify the set while iterating
            new_reachable = set()
            for r in reachable:
                # Calculate r xor k
                g = math.gcd(r, k)
                new_r = (r * k) // (g * g)
                new_reachable.add(new_r)
            reachable.update(new_reachable)
            
            if 210 in reachable:
                found_M = M
                break
    
    if found_M is None:
        # Should not happen for reasonable limits, but fallback
        return [840]
    
    # Now we have found_M.
    # We need to find a subset of current_kernels that multiplies to 210.
    # And for each kernel in that subset, find the smallest number in [211, 210+found_M] 
    # with that kernel.
    
    # Map kernel -> smallest number in range
    # We can just scan the range again.
    kernel_to_min_num = {}
    for i in range(211, 210 + found_M + 1):
        k = kernels[i]
        if k not in kernel_to_min_num:
            kernel_to_min_num[k] = i
    
    # Now find subset of keys in kernel_to_min_num that multiplies to 210.
    # Use BFS/DFS on the subset product again, but this time tracking the path.
    
    # State: (current_product, path_tuple)
    # Start: (1, ())
    # Target: 210
    
    # Since the number of distinct kernels is small, we can use a simple recursive search 
    # or BFS.
    # Let's use BFS to find the shortest path (fewest kernels)? 
    # Actually, any path works. We just need *a* solution.
    # The number of kernels is small (likely < 20 for small M).
    
    # We can use a dictionary: product -> path
    # To avoid exponential blowup in memory, we can prune paths that are "worse"?
    # Since we just need *a* solution, and the state space of products is limited 
    # (square-free integers), we can just do BFS.
    
    # However, the products can get large if we include large primes.
    # But we only care about reaching 210.
    # We can restrict the search to products that are "compatible" with 210?
    # i.e., any prime in the product that is not in 210 must be even? 
    # No, the product itself is square-free. 
    # If a prime p is in the product and p not in 210, we must cancel it later.
    # This means we can't prune based on the product value alone easily without 
    # knowing the remaining steps.
    
    # Given the small size, let's just do a standard BFS on the state (product).
    # We store the parent to reconstruct the path.
    
    parent = {1: None} # product -> (prev_product, kernel_used)
    queue = [1]
    visited = {1}
    
    found_path = None
    
    while queue:
        curr = queue.pop(0)
        if curr == 210:
            # Reconstruct path
            path = []
            c = curr
            while parent[c] is not None:
                prev, k = parent[c]
                path.append(k)
                c = prev
            found_path = path
            break
        
        for k in current_kernels:
            g = math.gcd(curr, k)
            next_prod = (curr * k) // (g * g)
            if next_prod not in visited:
                visited.add(next_prod)
                parent[next_prod] = (curr, k)
                queue.append(next_prod)
                
        if time.time() > deadline:
            break
            
    if found_path is None:
        # Fallback if BFS failed (shouldn't happen if 210 was in reachable)
        return [840]
        
    # Convert kernel path to numbers
    selected_numbers = []
    for k in found_path:
        # Pick the smallest number with this kernel
        # We have kernel_to_min_num
        if k in kernel_to_min_num:
            selected_numbers.append(kernel_to_min_num[k])
        else:
            # This shouldn't happen
            return [840]
            
    # Verify distinctness
    # The kernels are distinct in the path? 
    # In BFS, we visit each product state once. 
    # Can we use the same kernel twice in the path?
    # If we use kernel K, we go from P to P^K.
    # If we use K again, we go from P^K to P.
    # So we would cycle back.
    # Since we mark visited states, we won't cycle.
    # So the kernels in the path are distinct?
    # Not necessarily. 
    # Example: Path 1 -> K1 -> K1*K2.
    # Kernels used: K1, K2. Distinct.
    # What if K1 = K2? Then 1 -> K1 -> 1. Cycle. Visited prevents this.
    # So yes, the kernels in the path are distinct.
    # Therefore, the numbers selected (smallest for each kernel) are distinct.
    
    # Verify count <= 100
    if len(selected_numbers) > 100:
        # Should not happen for small M
        return [840]
        
    # Verify product is square
    prod = 210
    for n in selected_numbers:
        prod *= n
        
    if not _is_square(prod):
        # Verification failed, fallback
        return [840]
        
    return selected_numbers

# EVOLVE-BLOCK-END

import math
import time
import random
import sys

def _classic_superpermutation_perms(n):
    """
    Generates the sequence of permutations for the classic superpermutation.
    Returns a list of tuples, where each tuple is a permutation of 1..n.
    """
    if n == 1:
        return [(1,)]
    
    # Recursively get the sequence for n-1
    prev_perms = _classic_superpermutation_perms(n - 1)
    
    # For the classic construction, we insert n into the gaps.
    # The standard recursive step:
    # S_n = S_{n-1} with n inserted.
    # Specifically, for each permutation p in S_{n-1}, we form a block.
    # Actually, the standard algorithm is:
    # S_1 = [1]
    # S_n: Take S_{n-1}. For each permutation p in S_{n-1} in order, 
    #       we don't just insert. 
    # Let's use the standard "insertion" method that generates the classic superpermutation.
    
    # A simpler way to generate the classic sequence:
    # The classic superpermutation is built by:
    # 1. Start with [1]
    # 2. For m in 2..n:
    #    new_perms = []
    #    for p in perms:
    #       new_perms.append(p + (m,)) # This is not quite right for the sequence of perms.
    
    # Let's use the property that the classic superpermutation contains all perms.
    # We can just generate the classic string and extract the perms.
    # But for n=7, the string is 5984 long. Extracting 5040 perms is easy.
    
    # Let's implement the string generation first, then extract.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Now extract the sequence of permutations from the final word
    perms = []
    seen = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen:
            seen.add(p)
            perms.append(p)
            
    # The classic construction ensures all perms are present.
    # The order is the order of first appearance.
    return perms

def _overlap(p, q):
    """
    Returns the length of the longest suffix of p that is a prefix of q.
    p and q are tuples of length n.
    """
    n = len(p)
    for k in range(n, 0, -1):
        if p[-k:] == q[:k]:
            return k
    return 0

def _sequence_length(perms, n):
    """
    Calculates the length of the superpermutation formed by stitching the given sequence of permutations.
    """
    if not perms:
        return 0
    total = n
    for i in range(len(perms) - 1):
        total += n - _overlap(perms[i], perms[i+1])
    return total

def _local_search(perms, n, deadline):
    """
    Performs local search (adjacent swaps) to minimize the superpermutation length.
    """
    current_perms = list(perms)
    current_len = _sequence_length(current_perms, n)
    N = len(current_perms)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping adjacent permutations
        # We can iterate forward or backward. Forward is standard.
        for i in range(N - 1):
            if time.time() > deadline:
                return current_perms, current_len
            
            p_i = current_perms[i]
            p_ip1 = current_perms[i+1]
            
            # Calculate cost before swap
            # Cost involves (i-1, i), (i, i+1), (i+1, i+2)
            
            cost_left = 0
            if i > 0:
                cost_left = n - _overlap(current_perms[i-1], p_i)
            
            cost_mid = n - _overlap(p_i, p_ip1)
            
            cost_right = 0
            if i + 2 < N:
                cost_right = n - _overlap(p_ip1, current_perms[i+2])
                
            old_cost = cost_left + cost_mid + cost_right
            
            # Calculate cost after swap
            # Sequence: ... p_{i-1}, p_{i+1}, p_i, p_{i+2} ...
            
            new_cost_left = 0
            if i > 0:
                new_cost_left = n - _overlap(current_perms[i-1], p_ip1)
                
            new_cost_mid = n - _overlap(p_ip1, p_i)
            
            new_cost_right = 0
            if i + 2 < N:
                new_cost_right = n - _overlap(p_i, current_perms[i+2])
                
            new_cost = new_cost_left + new_cost_mid + new_cost_right
            
            if new_cost < old_cost:
                # Perform swap
                current_perms[i], current_perms[i+1] = p_ip1, p_i
                current_len -= (old_cost - new_cost)
                improved = True
                # After swap, we might want to re-check i-1 and i
                # But for simplicity, we continue.
                
    return current_perms, current_len

def _random_perturb(perms, n, amount):
    """
    Randomly swaps 'amount' pairs of permutations to create a new starting point.
    """
    new_perms = list(perms)
    for _ in range(amount):
        i = random.randint(0, len(new_perms) - 2)
        new_perms[i], new_perms[i+1] = new_perms[i+1], new_perms[i]
    return new_perms

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    if n <= 5:
        # For small n, classic is optimal or near optimal
        perms = _classic_superpermutation_perms(n)
        # Run local search just in case
        perms, _ = _local_search(perms, n, deadline)
        return "".join(map(str, [x for p in perms for x in p])) # This is wrong, need to stitch
    
    # For n=7
    # 1. Get classic sequence
    perms = _classic_superpermutation_perms(n)
    
    # 2. Initial local search
    best_perms, best_len = _local_search(perms, n, deadline)
    
    # 3. Random restarts if time permits
    # We can try a few perturbations
    # Each local search pass takes some time. Let's estimate.
    # N=5040. One pass is 5040 swaps.
    # If we do 10 passes, that's 50k swaps.
    # Each swap check is ~20 overlap calcs. 1M overlap calcs.
    # Overlap calc is ~7 comparisons. 7M ops.
    # Python can do ~10M-50M simple ops/sec.
    # So 1 pass might take 0.1s - 0.5s.
    # We have 130s. We can do many passes.
    
    # Let's run multiple local search phases with different perturbations.
    # We will keep the best result found.
    
    for restart in range(10): # Try up to 10 restarts
        if time.time() > deadline:
            break
        
        # Perturb the current best
        perturbed = _random_perturb(best_perms, n, amount=100)
        
        # Local search on perturbed
        candidate_perms, candidate_len = _local_search(perturbed, n, deadline)
        
        if candidate_len < best_len:
            best_perms = candidate_perms
            best_len = candidate_len
        else:
            # If no improvement, maybe try a stronger perturbation?
            # Or just stop if we are running out of time.
            pass
            
    # Stitch the best_perms into a string
    if not best_perms:
        return ""
        
    word = list(best_perms[0])
    for i in range(1, len(best_perms)):
        p = best_perms[i]
        prev = best_perms[i-1]
        ov = _overlap(prev, p)
        word.extend(p[ov:])
        
    return "".join(map(str, word))

# Helper to stitch for small n or verification
def _stitch(perms, n):
    if not perms:
        return ""
    word = list(perms[0])
    for i in range(1, len(perms)):
        p = perms[i]
        prev = perms[i-1]
        ov = _overlap(prev, p)
        word.extend(p[ov:])
    return "".join(map(str, word))

# Re-define construct to use _stitch
def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    if n <= 5:
        perms = _classic_superpermutation_perms(n)
        perms, _ = _local_search(perms, n, deadline)
        return _stitch(perms, n)
    
    # For n=7
    perms = _classic_superpermutation_perms(n)
    
    # Initial local search
    best_perms, best_len = _local_search(perms, n, deadline)
    
    # Random restarts
    for restart in range(20): # Increase restarts if fast
        if time.time() > deadline:
            break
        
        # Perturb the current best
        # Use a larger perturbation to escape local optima
        perturbed = _random_perturb(best_perms, n, amount=500)
        
        # Local search on perturbed
        candidate_perms, candidate_len = _local_search(perturbed, n, deadline)
        
        if candidate_len < best_len:
            best_perms = candidate_perms
            best_len = candidate_len
            
    return _stitch(best_perms, n)

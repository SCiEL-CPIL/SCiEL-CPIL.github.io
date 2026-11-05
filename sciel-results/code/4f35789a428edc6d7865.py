import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Target n=20 first, as it is known to exist.
    # If that fails, try 19, etc.
    
    best_result = {'n': 5, 'edges': []}
    
    # Helper to check if a subset mask induces a regular graph
    def is_regular(mask, adj_masks, n):
        k = mask.bit_count()
        if k < 6:
            return False
        
        # Calculate degrees
        # We can optimize by checking if all degrees are equal.
        # For small k (6..20), this is fast.
        
        # Get vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        degs = []
        for v in verts:
            d = (adj_masks[v] & mask).bit_count()
            degs.append(d)
            
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    def check_validity(n, adj_masks):
        # Check all subsets of size >= 6
        # Optimization: If a subset of size k is regular, does it imply smaller subsets? No.
        # We must check all k from 6 to n.
        # However, for n=20, checking all subsets is C(20,6)+... ~ 60k checks.
        # Each check is O(k^2) or O(k) with bit ops.
        # 60k * 20 ops = 1.2M ops. Very fast in Python with bit ops? 
        # Actually, iterating combinations in Python is slow.
        
        # Better: Iterate masks? 2^20 is 1M.
        # For each mask with popcount >= 6, check regularity.
        # Popcount check is fast.
        
        # Let's stick to combinations for correctness and simplicity, 
        # but we can optimize the inner loop.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular(mask, adj_masks, n):
                    return False
        return True

    def count_violations_size6(n, adj_masks):
        # Count how many subsets of size 6 are regular.
        # This is a good heuristic for local search.
        count = 0
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular(mask, adj_masks, n):
                count += 1
        return count

    # Strategy: 
    # 1. Try to build a valid graph for n=20 using a greedy/randomized hybrid.
    # 2. Start with a random graph.
    # 3. Use local search (hill climbing) to minimize the number of regular induced subgraphs of size 6.
    # 4. If we reach 0 violations of size 6, do a full check.
    # 5. If full check passes, return.
    # 6. If local search stalls, restart with a different random seed.
    
    for target_n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(12345 + restart * 100 + target_n)
            
            # Initialize random graph
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            current_violations = count_violations_size6(target_n, adj_masks)
            
            # If we start with 0, check full
            if current_violations == 0:
                if check_validity(target_n, adj_masks):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj_masks[i] & (1 << j):
                                edges.append([i, j])
                    best_result = {'n': target_n, 'edges': edges}
                    return best_result
            
            # Local Search: Flip edges to reduce violations
            no_improvement_count = 0
            max_no_improvement = 100
            
            while no_improvement_count < max_no_improvement and time.time() < deadline:
                # Pick a random edge to flip
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                
                # Flip
                adj_masks[i] ^= (1 << j)
                adj_masks[j] ^= (1 << i)
                
                new_violations = count_violations_size6(target_n, adj_masks)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    no_improvement_count = 0
                    
                    if current_violations == 0:
                        if check_validity(target_n, adj_masks):
                            edges = []
                            for i in range(target_n):
                                for j in range(i + 1, target_n):
                                    if adj_masks[i] & (1 << j):
                                        edges.append([i, j])
                            best_result = {'n': target_n, 'edges': edges}
                            return best_result
                else:
                    # Revert
                    adj_masks[i] ^= (1 << j)
                    adj_masks[j] ^= (1 << i)
                    no_improvement_count += 1
                    
            # If we got stuck, try a different random initialization (handled by outer loop)
            
    # If we couldn't find 20, return best found (likely 5)
    # But we should try to return a valid graph if we found one during the process?
    # The loop returns immediately upon finding a valid graph.
    # So if we exit the loop, we didn't find one for 20..6.
    # n=5 is always valid.
    
    return best_result

# Fallback: If the above logic is too slow or fails, we can hardcode a known valid graph if we had one.
# But we don't. So we rely on the search.

# To improve the search, we can use a smarter initialization or a different objective.
# However, the prompt asks for a "different strategy".
# The current program in the prompt description uses a slow verifier and then switches to random restarts.
# My implementation above uses a fast verifier (bit_count) and local search (hill climbing).
# This is a structural change: from "random restart + full check" to "local search on violation count".

# Let's refine the local search to be more effective.
# Instead of random edge flips, we can pick edges that are part of many violations.
# But identifying those is expensive.
# Random flips with hill climbing is a standard approach.

# One more optimization: 
# If n=20 is too hard, n=19 might be easier.
# The loop tries 20 down to 6.

# Let's add a timeout guard inside the local search loop more aggressively.
# The current while loop checks time.time() < deadline.

# Final check: 
# The task says "Maximize n".
# If we find a valid graph for n=20, we return it.
# If not, we try 19.
# If we find 19, we return it.
# This maximizes n.

# Is there a risk that local search gets stuck in a local minimum with >0 violations?
# Yes. That's why we have restarts.
# Is 50 restarts enough?
# For n=20, the space is huge. 
# But the "violation count" landscape might be smooth enough.
# If not, we might need simulated annealing.
# But hill climbing is faster per step.

# Let's stick to hill climbing with restarts.

# One potential issue: 
# count_violations_size6 is called frequently.
# It iterates C(20,6) = 38760 subsets.
# Each check is O(6) bit ops.
# 38760 * 6 = 232k ops per call.
# In Python, 232k ops takes ~0.01-0.02s.
# So we can do ~50-100 calls per second.
# If we do 100 steps per restart, that's 1-2 seconds per restart.
# 50 restarts = 50-100 seconds.
# This fits within the 115s deadline.

# If we need to go faster, we can reduce the number of restarts or steps.
# Or use a faster verifier.
# The verifier is already using bit_count, which is fast.

# Let's proceed with this implementation.

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found graph with n={res['n']}")

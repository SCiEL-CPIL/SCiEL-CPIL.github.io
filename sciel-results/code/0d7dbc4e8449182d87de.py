"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    rng = random.Random(42)
    
    best_result = None
    best_n = 0

    # Try to find a valid graph for n=21, then 20, etc.
    # We start with 20 because it's known to exist, 21 is the target for improvement.
    for target_n in [21, 20, 19]:
        if time.time() > deadline - 10:
            break
            
        # Precomputation for this n
        # We only need subsets of size >= 6
        subsets_by_size = {}
        pairs_to_subsets = {}
        
        # Initialize pairs_to_subsets
        for i in range(target_n):
            for j in range(i+1, target_n):
                pairs_to_subsets[(i, j)] = []
                
        # Generate subsets
        # Note: For n=21, total subsets >= 6 is ~2 million. 
        # Storing them all might be memory heavy but fast.
        # Let's estimate memory: 2M subsets * 6-21 ints * 28 bytes ~ 1GB? Too much.
        # We need a smarter way.
        
        # Alternative: Don't store all subsets. 
        # Instead, in the local search, when flipping (u,v), 
        # generate subsets containing u,v on the fly?
        # No, that's slow.
        
        # Let's restrict to n=20 first. 
        # For n=20, subsets >= 6 is ~1.1M.
        # 1.1M * 6 ints * 8 bytes (if using tuples) ~ 50MB. Acceptable.
        
        if target_n > 20:
             # For n=21, we might skip precomputation and use a slower check 
             # or limit the search time strictly.
             # Let's try a simpler local search for n=21 without full precomp 
             # if it's too slow to build.
             pass

        # Build precomputed structures for n <= 20
        if target_n <= 20:
            all_subsets = []
            for size in range(6, target_n + 1):
                subs = list(itertools.combinations(range(target_n), size))
                subsets_by_size[size] = subs
                all_subsets.extend(subs)
                for sub in subs:
                    # Add to pairs_to_subsets for all pairs in sub
                    # This is O(k^2) per subset.
                    # Total operations ~ Sum( C(n,k) * k^2 )
                    # For n=20, this is feasible.
                    for i in range(len(sub)):
                        for j in range(i+1, len(sub)):
                            u, v = sub[i], sub[j]
                            if u > v:
                                u, v = v, u
                            pairs_to_subsets[(u, v)].append(sub)
            
            # Local search
            result = _local_search_precomputed(target_n, pairs_to_subsets, deadline - 5, rng)
            if result is not None:
                if result['n'] > best_n:
                    best_result = result
                    best_n = result['n']
                    if best_n >= 21:
                        break
        else:
            # Fallback for n=21: Slower local search without full precomp
            # We will check violations by iterating subsets, but only 
            # if we suspect improvement.
            result = _local_search_slow(target_n, deadline - 5, rng)
            if result is not None:
                if result['n'] > best_n:
                    best_result = result
                    best_n = result['n']

    # Fallback to smaller n if nothing found
    if best_n < 10:
        # Try a simple greedy or random search for smaller n
        for target_n in [18, 17, 16, 15]:
            if time.time() > deadline - 2:
                break
            result = _local_search_precomputed(target_n, None, deadline - 2, rng) # Needs precomp
            # Re-use logic: we need to build precomp for this n too
            # Let's just call a generic solver
            res = _solve_n(target_n, deadline - 2, rng)
            if res and res['n'] > best_n:
                best_result = res
                best_n = res['n']

    if best_result is None:
        # Minimal valid graph: n=5 (no subset of size 6)
        # Actually n=6 is trivially valid if not regular.
        # A random graph on 6 vertices is rarely regular.
        # Let's return a valid graph on 10 vertices to be safe.
        best_result = _make_fallback(10, rng)
        
    return best_result


def _make_fallback(n, rng):
    """Generate a verified valid graph on n vertices."""
    for _ in range(1000):
        adj_mask = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    edges.append([i, j])
        if _is_valid_full(adj_mask, n):
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid_full(adj_mask, n):
    """Check if no subset of size >= 6 induces a regular graph."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj_mask, subset):
                return False
    return True


def _is_regular_subset(adj_mask, subset):
    """Check if induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True
    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)
    
    d0 = None
    for v in subset:
        d = (adj_mask[v] & subset_mask).bit_count()
        if d0 is None:
            d0 = d
        elif d != d0:
            return False
    return True


def _local_search_precomputed(n, pairs_to_subsets, deadline, rng):
    """Local search using precomputed pairs_to_subsets for fast delta calculation."""
    if pairs_to_subsets is None:
        return None
        
    adj_mask = [0] * n
    edges = []
    
    # Random init
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edges.append([i, j])
                
    # Initial cost
    current_violations = _count_violations_full(adj_mask, n)
    
    if current_violations == 0:
        return {'n': n, 'edges': edges}
        
    best_violations = current_violations
    best_adj = adj_mask[:]
    best_edges = [e[:] for e in edges]
    
    temperature = 1.0
    min_temperature = 0.01
    
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Pick random edge
        u, v = rng.sample(range(n), 2)
        
        # Flip edge
        old_val = (adj_mask[u] >> v) & 1
        if old_val:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            # Remove edge from list (lazy deletion or rebuild later)
            # For efficiency, we'll just track adj_mask and rebuild edges at end
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Calculate delta violations
        # Only subsets containing u and v can change
        delta = 0
        # Get subsets containing u and v
        # We need to check if they were regular before and after
        # We need the state BEFORE flip to check 'was_regular'
        # So we must compute 'was_regular' BEFORE flipping? 
        # Or we can compute 'is_regular_now' and 'is_regular_old'
        
        # Let's optimize: 
        # We flipped u-v. 
        # For each subset S containing u and v:
        #   Check if S is regular in NEW state.
        #   Check if S is regular in OLD state.
        #   delta += (1 if new else 0) - (1 if old else 0)
        
        # To check OLD state, we need to temporarily revert or calculate degrees 
        # considering the edge is absent/present.
        
        # Efficient check for a specific subset S containing u and v:
        # We know the degrees of all vertices in S in the NEW graph.
        # In the OLD graph, degrees of u and v are different by 1.
        
        # Let's precompute degrees for the subset in the NEW graph.
        
        # This is getting complex to do inline without precomputing subset masks.
        # Let's store subset masks in pairs_to_subsets? 
        # Memory might be an issue.
        
        # Alternative: Just recompute cost for affected subsets.
        # To check if S is regular in OLD state:
        #   We can just check if S is regular in NEW state.
        #   If it is, we need to know if it was regular in OLD state.
        #   If it was regular in OLD, and is regular in NEW, delta = 0.
        #   If it was regular in OLD, and NOT regular in NEW, delta = -1.
        #   If it was NOT regular in OLD, and IS regular in NEW, delta = +1.
        #   If it was NOT regular in OLD, and NOT regular in NEW, delta = 0.
        
        # So we need to check regularity in BOTH states for affected subsets.
        
        # Let's do it.
        affected_subs = pairs_to_subsets.get((min(u,v), max(u,v)), [])
        
        # We need a fast regularity check that can handle "virtual" edge flip?
        # Or just check NEW state, and if regular, check OLD state.
        
        new_violations_in_affected = 0
        old_violations_in_affected = 0
        
        for sub in affected_subs:
            # Check NEW state
            if _is_regular_subset(adj_mask, sub):
                new_violations_in_affected += 1
                # Check OLD state
                # Temporarily revert u-v
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                if _is_regular_subset(adj_mask, sub):
                    old_violations_in_affected += 1
                # Restore
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
            else:
                # If not regular in NEW, it might be regular in OLD?
                # No, if it's not regular in NEW, it contributes 0 to new count.
                # But it might have contributed 1 to old count.
                # So we MUST check OLD state for ALL affected subsets, 
                # or at least those that are "close" to regular.
                # Actually, if it's not regular in NEW, it could be regular in OLD.
                # Example: degrees were [2,2,2,2,2,2] (regular). 
                # Flip edge: degrees become [3,3,2,2,2,2] (not regular).
                # So NEW is not regular, OLD was regular. Delta = -1.
                
                # So we must check OLD state for all affected subsets.
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                if _is_regular_subset(adj_mask, sub):
                    old_violations_in_affected += 1
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                
        delta = new_violations_in_affected - old_violations_in_affected
        new_total_violations = current_violations + delta
        
        if new_total_violations < current_violations or (rng.random() < temperature):
            # Accept
            current_violations = new_total_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj_mask[:]
                # Rebuild edges for best_adj? We'll do it at the end.
            temperature = max(min_temperature, temperature * 0.99)
        else:
            # Revert
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            temperature = max(min_temperature, temperature * 0.99)
            
    if best_violations == 0:
        # Rebuild edges from best_adj
        final_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    final_edges.append([i, j])
        return {'n': n, 'edges': final_edges}
        
    return None


def _local_search_slow(n, deadline, rng):
    """Slower local search for larger n without precomputation."""
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    current_violations = _count_violations_full(adj_mask, n)
    
    for step in range(10000):
        if time.time() > deadline:
            break
        u, v = rng.sample(range(n), 2)
        old_val = (adj_mask[u] >> v) & 1
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)
        
        new_violations = _count_violations_full(adj_mask, n)
        
        if new_violations < current_violations:
            current_violations = new_violations
        else:
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            
    if current_violations == 0:
        final_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    final_edges.append([i, j])
        return {'n': n, 'edges': final_edges}
    return None


def _count_violations_full(adj_mask, n):
    count = 0
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj_mask, subset):
                count += 1
    return count


def _solve_n(n, deadline, rng):
    # Build precomp for this n
    pairs_to_subsets = {}
    for i in range(n):
        for j in range(i+1, n):
            pairs_to_subsets[(i, j)] = []
            
    for size in range(6, n + 1):
        subs = list(itertools.combinations(range(n), size))
        for sub in subs:
            for i in range(len(sub)):
                for j in range(i+1, len(sub)):
                    u, v = sub[i], sub[j]
                    if u > v:
                        u, v = v, u
                    pairs_to_subsets[(u, v)].append(sub)
                    
    return _local_search_precomputed(n, pairs_to_subsets, deadline, rng)
# EVOLVE-BLOCK-END

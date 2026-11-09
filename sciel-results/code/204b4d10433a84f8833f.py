import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256

    # Precompute all 4-APs and 3-APs using bitsets for speed
    # Represent sets as Python ints (bitsets) for fast operations
    
    # Precompute: for each pair (a,b) with a<b, does there exist c,d such that
    # a,b,c,d form a 4-AP? We'll precompute conflict edges.
    
    # For 4-AP detection: a 4-AP is (x, x+d, x+2d, x+3d)
    # Two elements conflict if they can be part of the same 4-AP together with two others
    
    # Actually, the constraint is: no 4 elements in A form a 4-AP.
    # So we need: for any 4-AP (w,x,y,z), not all four are in A.
    # This is equivalent to: the set A is an independent set in a hypergraph
    # where each hyperedge is a 4-AP.
    
    # For greedy: when adding element x, check if it completes any 4-AP with 3 existing elements.
    
    # Precompute all 4-APs
    all_4aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e <= N:
                all_4aps.append((a, b, c, e))
    
    # For each element, precompute which other elements it can form a 4-AP with
    # conflict_4ap[x] = set of y such that x and y are in the same 4-AP
    # But we need: given A, adding x creates a 4-AP iff there exist y,z,w in A
    # such that {x,y,z,w} forms a 4-AP.
    
    # More efficient: for each x, precompute the list of "triples" (y,z,w) 
    # such that {x,y,z,w} is a 4-AP. Then adding x is invalid iff any triple
    # is fully contained in A.
    
    # For each element x, store all triples (y,z,w) with x,y,z,w forming a 4-AP
    # where x is one of the four positions
    x_to_4ap_triples = {i: [] for i in range(1, N+1)}
    for (a, b, c, e) in all_4aps:
        # x=a: need b,c,e in A
        x_to_4ap_triples[a].append((b, c, e))
        # x=b: need a,c,e in A
        x_to_4ap_triples[b].append((a, c, e))
        # x=c: need a,b,e in A
        x_to_4ap_triples[c].append((a, b, e))
        # x=e: need a,b,c in A
        x_to_4ap_triples[e].append((a, b, c))
    
    # Precompute 3-AP gains: for each x, which pairs (y,z) in A would be completed
    # A 3-AP is (p, q, r) with q-p = r-q, i.e., p+r = 2q
    # Adding x completes a 3-AP if:
    # - x is the middle: need y,z in A with y+z = 2x
    # - x is the first: need y,z in A with z-y = y-x, i.e., z = 2y-x
    # - x is the last: need y,z in A with y-x = x-z, i.e., y = 2x-z
    
    # For speed, use bitset representation
    # bitset[i] = 1 if i is in the set
    
    def bitset_test(bs, x):
        return (bs >> (x-1)) & 1
    
    def bitset_set(bs, x):
        return bs | (1 << (x-1))
    
    def bitset_clear(bs, x):
        return bs & ~(1 << (x-1))
    
    def creates_4ap_fast(x, bs):
        """Check if adding x to set (bitset bs) creates a 4-AP."""
        for (y, z, w) in x_to_4ap_triples[x]:
            if bitset_test(bs, y) and bitset_test(bs, z) and bitset_test(bs, w):
                return True
        return False
    
    def gain_3ap_fast(x, bs):
        """Count 3-APs completed by adding x to set (bitset bs)."""
        gain = 0
        # x as middle: need y, z in A with y + z = 2*x
        # y < x < z, z - x = x - y => z = 2*x - y
        for y in range(1, x):
            z = 2 * x - y
            if z > x and z <= N and bitset_test(bs, y) and bitset_test(bs, z):
                gain += 1
        # x as first: need y, z in A with z = 2*y - x, y > x
        for y in range(x+1, N+1):
            z = 2 * y - x
            if z > y and z <= N and bitset_test(bs, y) and bitset_test(bs, z):
                gain += 1
        # x as last: need y, z in A with y = 2*x - z, z < x
        for z in range(1, x):
            y = 2 * x - z
            if y > z and y <= N and bitset_test(bs, y) and bitset_test(bs, z):
                gain += 1
        return gain
    
    def count_3aps_bitset(bs):
        """Count all 3-APs in the set represented by bitset bs."""
        count = 0
        # Iterate over all possible middles
        for q in range(2, N):
            for p in range(1, q):
                r = 2 * q - p
                if r > q and r <= N:
                    if bitset_test(bs, p) and bitset_test(bs, q) and bitset_test(bs, r):
                        count += 1
        return count
    
    def has_4ap_bitset(bs):
        """Check if the set contains a 4-AP."""
        for (a, b, c, e) in all_4aps:
            if bitset_test(bs, a) and bitset_test(bs, b) and bitset_test(bs, c) and bitset_test(bs, e):
                return True
        return False
    
    def greedy_conflict_graph(start_bs=0, seed=None, num_restarts=1):
        """
        Weighted greedy using conflict graph model.
        At each step, among all valid elements (those that don't create a 4-AP),
        pick the one with highest 3-AP gain.
        """
        bs = start_bs
        used = set()
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        while time.time() < deadline - 10:
            # Find all valid elements
            valid_elements = []
            for x in range(1, N+1):
                if x in used:
                    continue
                if not creates_4ap_fast(x, bs):
                    valid_elements.append(x)
            
            if not valid_elements:
                break
            
            # Score each valid element
            best_x = None
            best_score = -1
            for x in valid_elements:
                g = gain_3ap_fast(x, bs)
                # Add small random perturbation for diversity
                score = g + rng.gauss(0, 0.1)
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            bs = bitset_set(bs, best_x)
            used.add(best_x)
        
        return bs
    
    def local_search_bitset(bs, max_iters=20):
        """
        Local search: try to improve by swapping elements.
        """
        used = set()
        for x in range(1, N+1):
            if bitset_test(bs, x):
                used.add(x)
        
        remaining = set(range(1, N+1)) - used
        current_3ap_count = count_3aps_bitset(bs)
        
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            improved = False
            
            # Try removing each element and see if we can add a better one
            for a in sorted(used):
                if time.time() > deadline - 15:
                    break
                if improved:
                    break
                
                # Remove a
                bs_temp = bitset_clear(bs, a)
                used_temp = used - {a}
                
                # Find best replacement
                best_b = None
                best_gain_diff = 0
                gain_a = gain_3ap_fast(a, bitset_clear(bs, a))  # gain if a were added to A\{a}
                
                for b in sorted(remaining):
                    if creates_4ap_fast(b, bs_temp):
                        continue
                    gain_b = gain_3ap_fast(b, bs_temp)
                    if gain_b > gain_a:
                        if gain_b > best_gain_diff:
                            best_gain_diff = gain_b
                            best_b = b
                
                if best_b is not None:
                    # Accept swap
                    bs = bitset_set(bs_temp, best_b)
                    used = used_temp | {best_b}
                    remaining = (remaining - {best_b}) | {a}
                    improved = True
                else:
                    # Restore
                    bs = bs_temp | (1 << (a-1))
                    used = used_temp | {a}
            
            if not improved:
                break
        
        return bs, count_3aps_bitset(bs)
    
    # Main search: multiple greedy restarts with local search
    best_bs = 0
    best_score = 0
    
    # Run multiple greedy restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Different random seeds for diversity
        seed = i * 7 + 13
        bs = greedy_conflict_graph(start_bs=0, seed=seed)
        
        if not bitset_test(bs, 1):  # empty check
            # Try a simpler deterministic approach
            bs = 0
            used = set()
            for x in range(1, N+1):
                if not creates_4ap_fast(x, bs):
                    g = gain_3ap_fast(x, bs)
                    if g > 0:
                        bs = bitset_set(bs, x)
                        used.add(x)
        
        score = count_3aps_bitset(bs)
        
        if score > best_score:
            best_score = score
            best_bs = bs
        
        # Local search refinement
        if time.time() < deadline - 15:
            bs_ls, score_ls = local_search_bitset(bs, max_iters=10)
            if score_ls > best_score:
                best_score = score_ls
                best_bs = bs_ls
    
    # Final verification
    if has_4ap_bitset(best_bs):
        # Emergency fix: remove elements until no 4-AP
        for x in range(1, N+1):
            if has_4ap_bitset(best_bs):
                best_bs = bitset_clear(best_bs, x)
    
    # Convert bitset to sorted list
    result = []
    for x in range(1, N+1):
        if bitset_test(best_bs, x):
            result.append(x)
    
    if not result:
        # Fallback: return a simple valid set
        result = [1]
    
    return result

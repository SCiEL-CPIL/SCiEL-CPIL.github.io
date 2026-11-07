"""Rewritten program for the suite task; using branch-and-bound on 3-uniform hypergraph."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    idx_to_val = {v: i for i, v in enumerate(vertices)}
    val_to_idx = {v: i for i, v in enumerate(vertices)}

    # Precompute conflict bitsets W_uv
    # W_uv is a bitset of vertices w such that {u, v, w} is a bad triple.
    # Bad if u | v*w OR v | u*w OR w | u*v.
    W = [0] * (n * n) # W[i*n + j]
    
    # Optimization: Only compute for i < j? No, W is symmetric in u,v for the set of w?
    # Condition is symmetric for u,v in the first two terms.
    # w | u*v is symmetric in u,v.
    # So W_uv == W_vu.
    
    for i in range(n):
        u = vertices[i]
        for j in range(i + 1, n):
            v = vertices[j]
            mask = 0
            # Check all w
            # w must be distinct from u, v
            for k in range(n):
                if k == i or k == j:
                    continue
                w = vertices[k]
                # Check u | v*w
                if (v * w) % u == 0:
                    mask |= (1 << k)
                    continue
                # Check v | u*w
                if (u * w) % v == 0:
                    mask |= (1 << k)
                    continue
                # Check w | u*v
                if (u * v) % w == 0:
                    mask |= (1 << k)
                    continue
            W[i * n + j] = mask
            W[j * n + i] = mask

    # Initialize best with primes
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0: return False
        return True
        
    primes = [v for v in vertices if is_prime(v)]
    best_set = primes
    best_size = len(best_set)
    
    # Branch and Bound
    # We use a simple recursive search with pruning.
    # To avoid recursion depth issues and timeouts, we add a time limit.
    start_time = time.time()
    timeout = 55.0
    
    # Order vertices by degree (number of conflicts) to prune earlier?
    # Or just iterate.
    # We represent available as a bitset (Python int).
    # Initial available: all bits set.
    full_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
        
    def get_index(bit):
        # Find index of least significant set bit
        return (bit & -bit).bit_length() - 1

    def solve(current_set, available, depth):
        nonlocal best_set, best_size
        
        if time.time() - start_time > timeout:
            return
            
        cur_size = len(current_set)
        avail_count = popcount(available)
        
        if cur_size + avail_count <= best_size:
            return
            
        if available == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_set = current_set[:]
            return
            
        # Greedy upper bound
        # Try to build a large set from available
        temp_avail = available
        temp_set = current_set[:]
        greedy_size = cur_size
        while temp_avail:
            # Pick a vertex with fewest conflicts in temp_avail?
            # For simplicity, pick the first one
            v_idx = get_index(temp_avail)
            v_bit = 1 << v_idx
            temp_set.append(vertices[v_idx])
            greedy_size += 1
            
            # Update temp_avail: remove v and any w that conflicts with v given temp_set
            # Conflict with v means: for any z in temp_set (excluding v), w in W_vz
            # Also w cannot be v.
            forbidden = 0
            for z_idx in range(len(temp_set) - 1): # all z in temp_set before v
                z_val = temp_set[z_idx]
                # Find index of z
                # We need indices for W lookup
                # temp_set stores values. We need to map to indices.
                # This is slow. 
                # Optimization: Store indices in current_set?
                # Let's refactor to store indices.
                pass
            # This greedy bound is complex to implement efficiently in the inner loop.
            # Let's stick to the simple bound: cur_size + avail_count.
            # If that's not enough, we might time out.
            # Given the prime set is likely optimal, the search might be small if we prune well.
            # But if we don't find a better set, we explore the whole space?
            # No, we prune if cur_size + avail_count <= best_size.
            # Since best_size is 54, and we start with 0, we will explore until we find a set of size 54?
            # Actually, we start with best_size=54.
            # If we can't exceed 54, we prune branches where cur_size + avail_count <= 54.
            # This should be effective.
            
            break # Skip complex greedy for now to keep code simple and fast per node
            
        # Pick next vertex to branch on
        # Heuristic: Pick vertex with highest "conflict degree" in available?
        # Or just the first one.
        # Let's pick the first available vertex.
        v_idx = get_index(available)
        v_bit = 1 << v_idx
        v_val = vertices[v_idx]
        
        # Branch 1: Include v
        # Check if v is compatible with current_set
        # v is compatible if for all pairs {a, b} in current_set, v not in W_ab
        # Equivalent: v not in union of W_ab for all pairs in current_set.
        # We can maintain a "forbidden_mask" for the current set?
        # Let's compute it on the fly.
        compatible = True
        cs = current_set
        for i in range(len(cs)):
            a = cs[i]
            for j in range(i + 1, len(cs)):
                b = cs[j]
                # Get indices
                ia = val_to_idx[a]
                ib = val_to_idx[b]
                if (W[ia * n + ib] & v_bit):
                    compatible = False
                    break
            if not compatible:
                break
                
        if compatible:
            new_set = cs + [v_val]
            # Update available
            # Remove v
            new_avail = available & ~v_bit
            # Remove any w that conflicts with v given the NEW set
            # Conflicts are with pairs {v, z} where z in new_set (z != v)
            # So for each z in new_set (excluding v), remove W_vz from new_avail
            forbidden_new = 0
            for z_val in new_set:
                if z_val == v_val:
                    continue
                iv = val_to_idx[v_val]
                iz = val_to_idx[z_val]
                forbidden_new |= W[iv * n + iz]
            new_avail &= ~forbidden_new
            
            solve(new_set, new_avail, depth + 1)
            
        # Branch 2: Exclude v
        solve(cs, available & ~v_bit, depth + 1)

    # Start search
    # To speed up, we can start with a greedy solution?
    # We already have best_set from primes.
    # We start with empty set.
    solve([], full_mask, 0)
    
    return best_set

# EVOLVE-BLOCK-END
